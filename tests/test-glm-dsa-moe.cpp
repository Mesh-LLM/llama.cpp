#include "test-glm-dsa-moe.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

struct moe_shape {
    ggml_type type;
    int64_t   n_input;
    int64_t   n_output;
};

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static float fixture_value(uint64_t index, uint32_t salt) {
    uint32_t value = static_cast<uint32_t>(index) ^ salt;
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return static_cast<float>(value & 0xffffU) / 32768.0f - 1.0f;
}

static std::vector<float> make_values(size_t count, uint32_t salt) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = fixture_value(i, salt);
    }
    return values;
}

static std::vector<uint8_t> make_quantized_weights(const moe_shape & shape) {
    const ggml_type_traits * traits = ggml_get_type_traits(shape.type);
    require(traits && traits->from_float_ref && traits->to_float, "missing quantization traits");

    const size_t         row_size = ggml_row_size(shape.type, shape.n_input);
    std::vector<uint8_t> weights(row_size * shape.n_output);
    std::vector<float>   row(shape.n_input);
    for (int64_t output = 0; output < shape.n_output; ++output) {
        for (int64_t input = 0; input < shape.n_input; ++input) {
            row[input] = fixture_value(static_cast<uint64_t>(output) * shape.n_input + input, 0x51a7d3e2U);
        }
        traits->from_float_ref(row.data(), weights.data() + output * row_size, shape.n_input);
    }
    return weights;
}

static std::vector<float> run_mul_mat_id(ggml_backend_t               backend,
                                         const moe_shape &            shape,
                                         const std::vector<uint8_t> & weights,
                                         const std::vector<float> &   input) {
    ggml_init_params params = {
        /* .mem_size   = */ 2 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr context(ggml_init(params));
    require(context != nullptr, "failed to create MoE precision context");

    ggml_tensor * matrix = ggml_new_tensor_3d(context.get(), shape.type, shape.n_input, shape.n_output, 1);
    ggml_tensor * vector = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, shape.n_input, 1, 1);
    ggml_tensor * ids    = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * output = ggml_mul_mat_id(context.get(), matrix, vector, ids);

    ggml_cgraph * graph = ggml_new_graph_custom(context.get(), 32, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_backend_supports_op(backend, output), "backend does not support GLM MoE matvec");

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
    require(buffer != nullptr, "failed to allocate MoE precision tensors");
    require(ggml_nbytes(matrix) == weights.size(), "quantized MoE weight size differs");
    ggml_backend_tensor_set(matrix, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(vector, input.data(), 0, input.size() * sizeof(float));
    const int32_t expert = 0;
    ggml_backend_tensor_set(ids, &expert, 0, sizeof(expert));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "GLM MoE precision graph failed");
    std::vector<float> values(ggml_nelements(output));
    ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    return values;
}

static std::vector<float> precise_reference(const moe_shape &            shape,
                                            const std::vector<uint8_t> & weights,
                                            const std::vector<float> &   input) {
    const ggml_type_traits * traits   = ggml_get_type_traits(shape.type);
    const size_t             row_size = ggml_row_size(shape.type, shape.n_input);
    std::vector<float>       row(shape.n_input);
    std::vector<float>       output(shape.n_output);
    for (int64_t i = 0; i < shape.n_output; ++i) {
        traits->to_float(weights.data() + i * row_size, row.data(), shape.n_input);
        double sum = 0.0;
        for (int64_t j = 0; j < shape.n_input; ++j) {
            sum += static_cast<double>(row[j]) * input[j];
        }
        output[i] = static_cast<float>(sum);
    }
    return output;
}

static double normalized_mean_squared_error(const std::vector<float> & reference, const std::vector<float> & actual) {
    require(reference.size() == actual.size(), "MoE precision result sizes differ");
    double squared_error = 0.0;
    double reference_sum = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = static_cast<double>(reference[i]) - actual[i];
        squared_error += difference * difference;
        reference_sum += static_cast<double>(reference[i]) * reference[i];
    }
    return squared_error / reference_sum;
}

static void test_shape(ggml_backend_t cpu, ggml_backend_t metal, const moe_shape & shape) {
    const std::vector<uint8_t> weights   = make_quantized_weights(shape);
    const std::vector<float>   input     = make_values(shape.n_input, 0x9c6d812fU);
    const std::vector<float>   reference = precise_reference(shape, weights, input);
    const std::vector<float>   cpu_out   = run_mul_mat_id(cpu, shape, weights, input);
    const std::vector<float>   metal_out = run_mul_mat_id(metal, shape, weights, input);

    const double cpu_nmse       = normalized_mean_squared_error(reference, cpu_out);
    const double metal_nmse     = normalized_mean_squared_error(reference, metal_out);
    const double cpu_metal_nmse = normalized_mean_squared_error(cpu_out, metal_out);
    std::printf("GLM MoE %s [%lldx%lld]: CPU/reference %.3e, Metal/reference %.3e, CPU/Metal %.3e\n",
                ggml_type_name(shape.type), static_cast<long long>(shape.n_output),
                static_cast<long long>(shape.n_input), cpu_nmse, metal_nmse, cpu_metal_nmse);

    require(metal_nmse <= 2e-4, "Metal GLM MoE matvec exceeds the numerical parity threshold");
    require(metal_nmse <= cpu_nmse, "Metal GLM MoE matvec is less accurate than the CPU Q8_K path");
}

}  // namespace

void test_glm_dsa_moe_precision() {
    ggml_backend_dev_t metal_device = ggml_backend_dev_by_name("MTL0");
    if (!metal_device) {
        return;
    }

    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    ggml_backend_ptr metal(ggml_backend_dev_init(metal_device, nullptr));
    require(cpu != nullptr && metal != nullptr, "failed to initialize MoE precision backends");

    test_shape(cpu.get(), metal.get(), { GGML_TYPE_Q2_K, 6144, 2048 });
    test_shape(cpu.get(), metal.get(), { GGML_TYPE_Q3_K, 2048, 6144 });
}
