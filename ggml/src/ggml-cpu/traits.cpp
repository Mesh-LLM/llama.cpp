#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

static bool ggml_cpu_traits_trace_enabled() {
    const char * path = getenv("GGML_CPU_TRAITS_TRACE_LOG");
    return path != NULL && path[0] != '\0';
}

static void ggml_cpu_traits_trace_event(
        const char * phase,
        const struct ggml_tensor * op,
        int extra_index,
        const ggml::cpu::tensor_traits * tensor_traits,
        bool handled) {
    if (!ggml_cpu_traits_trace_enabled()) {
        return;
    }
    if (op == NULL || (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID)) {
        return;
    }
    const char * path = getenv("GGML_CPU_TRAITS_TRACE_LOG");
    FILE * fp = fopen(path, "a");
    if (fp == NULL) {
        return;
    }
    fprintf(fp,
        "ggml-cpu-traits: pid=%d phase=%s op=%s op_name=%s extra_index=%d has_traits=%d trait=%s handled=%d src0_name=%s src0_buffer=%s\n",
        (int) getpid(),
        phase ? phase : "",
        ggml_op_name(op->op),
        op->name,
        extra_index,
        tensor_traits != nullptr ? 1 : 0,
        tensor_traits != nullptr ? tensor_traits->debug_name() : "(null)",
        handled ? 1 : 0,
        op->src[0] ? op->src[0]->name : "(null)",
        (op->src[0] && op->src[0]->buffer) ? ggml_backend_buffer_name(op->src[0]->buffer) : "(null)"
    );
    fclose(fp);
}

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    int extra_index = 0;
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ggml_cpu_traits_trace_event("compute_forward_check", op, extra_index, tensor_traits, false);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ggml_cpu_traits_trace_event("compute_forward_handled", op, extra_index, tensor_traits, true);
                return true;
            }
        }
        extra_index++;
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    int extra_index = 0;
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ggml_cpu_traits_trace_event("work_size_check", op, extra_index, tensor_traits, false);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ggml_cpu_traits_trace_event("work_size_handled", op, extra_index, tensor_traits, true);
                return true;
            }
        }
        extra_index++;
    }
    return false;
}
