// moe-components: extract topology-independent MoE components and assemble shards
//
// Subcommands:
//   extract-trunk   -m model.gguf -o trunk.gguf
//   extract-expert  -m model.gguf --expert N -o expert-N.gguf
//   assemble        --trunk trunk.gguf --expert-file expert-0.gguf ... -o shard.gguf

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdarg>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

enum class moe_components_mode {
    extract_trunk,
    extract_expert,
    assemble,
};

struct moe_components_params {
    moe_components_mode mode = moe_components_mode::extract_trunk;
    std::string model;
    std::string trunk;
    std::vector<std::string> expert_files;
    std::string output;
    int expert_id = -1;
};

static void print_usage(const char * prog) {
    printf("\n");
    printf("usage:\n");
    printf("  %s extract-trunk  -m model.gguf -o trunk.gguf\n", prog);
    printf("  %s extract-expert -m model.gguf --expert 42 -o expert-042.gguf\n", prog);
    printf("  %s assemble --trunk trunk.gguf --expert-file expert-000.gguf --expert-file expert-031.gguf -o shard.gguf\n", prog);
    printf("\n");
}

static bool is_expert_tensor(const char * name) {
    return strstr(name, "ffn_gate_exps") != nullptr
        || strstr(name, "ffn_up_exps")   != nullptr
        || strstr(name, "ffn_down_exps") != nullptr
        || strstr(name, "exp_probs_b")   != nullptr;
}

static bool is_shared_expert(const char * name) {
    return strstr(name, "_shexp") != nullptr;
}

static bool is_router_gate(const char * name) {
    return strstr(name, "ffn_gate_inp") != nullptr && !is_shared_expert(name);
}

static void zeros(std::ofstream & f, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; i++) {
        f.write(&zero, 1);
    }
}

struct input_shard {
    std::string path;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta;
};

[[noreturn]] static void fail(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

static void parse_args(int argc, const char ** argv, moe_components_params & params) {
    if (argc < 2) {
        print_usage(argv[0]);
        exit(1);
    }

    std::string subcommand = argv[1];
    if (subcommand == "extract-trunk") {
        params.mode = moe_components_mode::extract_trunk;
    } else if (subcommand == "extract-expert") {
        params.mode = moe_components_mode::extract_expert;
    } else if (subcommand == "assemble") {
        params.mode = moe_components_mode::assemble;
    } else if (subcommand == "-h" || subcommand == "--help") {
        print_usage(argv[0]);
        exit(0);
    } else {
        fail("error: unknown subcommand: %s", subcommand.c_str());
    }

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) fail("error: missing value for %s", arg.c_str());
            params.model = argv[i];
        } else if (arg == "--trunk") {
            if (++i >= argc) fail("error: missing value for --trunk");
            params.trunk = argv[i];
        } else if (arg == "--expert-file") {
            if (++i >= argc) fail("error: missing value for --expert-file");
            params.expert_files.push_back(argv[i]);
        } else if (arg == "--expert") {
            if (++i >= argc) fail("error: missing value for --expert");
            params.expert_id = std::atoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) fail("error: missing value for %s", arg.c_str());
            params.output = argv[i];
        } else {
            fail("error: unknown argument: %s", arg.c_str());
        }
    }

    switch (params.mode) {
        case moe_components_mode::extract_trunk:
            if (params.model.empty() || params.output.empty()) {
                fail("error: extract-trunk requires --model and --output");
            }
            break;
        case moe_components_mode::extract_expert:
            if (params.model.empty() || params.output.empty() || params.expert_id < 0) {
                fail("error: extract-expert requires --model, --expert, and --output");
            }
            break;
        case moe_components_mode::assemble:
            if (params.trunk.empty() || params.output.empty() || params.expert_files.empty()) {
                fail("error: assemble requires --trunk, at least one --expert-file, and --output");
            }
            break;
    }
}

static std::vector<input_shard> load_input_shards(const std::string & input_path) {
    std::vector<input_shard> shards;

    auto load_one = [&shards](const std::string & path) {
        struct ggml_context * ctx_meta = nullptr;
        struct gguf_init_params init_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx_meta,
        };

        auto * ctx_gguf = gguf_init_from_file(path.c_str(), init_params);
        if (!ctx_gguf) {
            fail("error: failed to load %s", path.c_str());
        }

        shards.push_back({ path, ctx_gguf, ctx_meta });
    };

    load_one(input_path);

    int split_count_idx = gguf_find_key(shards[0].ctx_gguf, "split.count");
    if (split_count_idx < 0) {
        return shards;
    }

    int n_split = gguf_get_val_u16(shards[0].ctx_gguf, split_count_idx);
    if (n_split <= 1) {
        return shards;
    }

    std::vector<char> split_prefix(PATH_MAX, 0);
    if (!llama_split_prefix(split_prefix.data(), split_prefix.size(), input_path.c_str(), 0, n_split)) {
        fail("error: unexpected split input file name: %s", input_path.c_str());
    }

    for (int i_split = 1; i_split < n_split; i_split++) {
        std::vector<char> split_path(PATH_MAX, 0);
        int ret = llama_split_path(
            split_path.data(),
            split_path.size(),
            split_prefix.data(),
            i_split,
            n_split);
        if (ret <= 0) {
            fail("error: failed to derive split path for shard %d/%d", i_split + 1, n_split);
        }
        load_one(split_path.data());
    }

    return shards;
}

static const input_shard * find_input_shard_for_tensor(
        const std::vector<input_shard> & input_shards,
        const char * name,
        int * shard_index_out,
        int * tensor_index_out) {
    for (int shard_index = 0; shard_index < (int) input_shards.size(); shard_index++) {
        int tensor_index = gguf_find_tensor(input_shards[shard_index].ctx_gguf, name);
        if (tensor_index >= 0) {
            if (shard_index_out) {
                *shard_index_out = shard_index;
            }
            if (tensor_index_out) {
                *tensor_index_out = tensor_index;
            }
            return &input_shards[shard_index];
        }
    }

    return nullptr;
}

static std::string architecture_name(struct gguf_context * ctx) {
    int arch_key = gguf_find_key(ctx, "general.architecture");
    if (arch_key < 0) {
        return "";
    }
    return gguf_get_val_str(ctx, arch_key);
}

static int expert_count(struct gguf_context * ctx) {
    std::string arch_name = architecture_name(ctx);
    std::string expert_key = arch_name + ".expert_count";
    int ec_idx = gguf_find_key(ctx, expert_key.c_str());
    if (ec_idx < 0) {
        fail("error: no expert count found (key: %s). Is this a MoE model?", expert_key.c_str());
    }
    return gguf_get_val_u32(ctx, ec_idx);
}

static void remove_split_metadata(struct gguf_context * ctx) {
    gguf_remove_key(ctx, "split.no");
    gguf_remove_key(ctx, "split.count");
    gguf_remove_key(ctx, "split.tensors.count");
}

static void update_expert_metadata(struct gguf_context * ctx, int new_expert_count) {
    std::string arch_name = architecture_name(ctx);
    std::string expert_count_key = arch_name + ".expert_count";
    gguf_set_val_u32(ctx, expert_count_key.c_str(), new_expert_count);

    std::string expert_used_key = arch_name + ".expert_used_count";
    int expert_used_idx = gguf_find_key(ctx, expert_used_key.c_str());
    if (expert_used_idx >= 0) {
        int expert_used = gguf_get_val_u32(ctx, expert_used_idx);
        if (expert_used > new_expert_count) {
            gguf_set_val_u32(ctx, expert_used_key.c_str(), new_expert_count);
        }
    }
}

static struct ggml_context * make_tensor_context(size_t tensor_count_hint) {
    size_t ctx_size = (tensor_count_hint + 64) * ggml_tensor_overhead() + 4096;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    return ggml_init(ctx_params);
}

static int expanded_component_dims(
        const struct ggml_tensor * tensor,
        int expert_count,
        int64_t ne_out[GGML_MAX_DIMS]) {
    int n_dims = ggml_n_dims(tensor);
    if (n_dims <= 0) {
        fail("error: tensor %s has invalid dimension count %d", tensor->name, n_dims);
    }

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        ne_out[d] = 1;
    }
    for (int d = 0; d < n_dims; d++) {
        ne_out[d] = tensor->ne[d];
    }

    if (tensor->ne[n_dims - 1] == 1) {
        ne_out[n_dims - 1] = expert_count;
        return n_dims;
    }

    if (n_dims >= GGML_MAX_DIMS) {
        fail("error: tensor %s has no room to restore expert dimension", tensor->name);
    }

    ne_out[n_dims] = expert_count;
    return n_dims + 1;
}

static void write_metadata(struct gguf_context * ctx_out, std::ofstream & f_out) {
    size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size);
    gguf_get_meta_data(ctx_out, meta_data.data());
    f_out.write((const char *) meta_data.data(), meta_size);
}

static void write_tensor_passthrough(
        std::ofstream & f_out,
        const std::vector<input_shard> & input_shards,
        const char * name) {
    int shard_index = -1;
    int tensor_index = -1;
    const input_shard * source_shard =
        find_input_shard_for_tensor(input_shards, name, &shard_index, &tensor_index);
    if (!source_shard) {
        fail("error: tensor %s not found in input", name);
    }

    struct ggml_tensor * t_in = ggml_get_tensor(source_shard->ctx_meta, name);
    size_t n_bytes = ggml_nbytes(t_in);
    size_t offset =
        gguf_get_data_offset(source_shard->ctx_gguf) +
        gguf_get_tensor_offset(source_shard->ctx_gguf, tensor_index);

    std::ifstream f_in(source_shard->path, std::ios::binary);
    if (!f_in.is_open()) {
        fail("error: cannot open %s", source_shard->path.c_str());
    }
    std::vector<uint8_t> buf(n_bytes);
    f_in.seekg(offset);
    f_in.read((char *) buf.data(), n_bytes);
    f_out.write((const char *) buf.data(), n_bytes);
    zeros(f_out, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
}

static void extract_trunk(const moe_components_params & params) {
    auto input_shards = load_input_shards(params.model);
    auto * ctx_in = input_shards[0].ctx_gguf;
    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    remove_split_metadata(ctx_out);

    size_t tensor_count_hint = 0;
    for (const auto & shard : input_shards) {
        int n_tensors = gguf_get_n_tensors(shard.ctx_gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(shard.ctx_gguf, i);
            if (is_expert_tensor(name) || is_router_gate(name)) {
                continue;
            }
            tensor_count_hint++;
        }
    }

    auto * ctx_new = make_tensor_context(tensor_count_hint);
    for (const auto & shard : input_shards) {
        int n_tensors = gguf_get_n_tensors(shard.ctx_gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(shard.ctx_gguf, i);
            if (is_expert_tensor(name) || is_router_gate(name)) {
                continue;
            }
            struct ggml_tensor * t = ggml_get_tensor(shard.ctx_meta, name);
            struct ggml_tensor * t_new = ggml_dup_tensor(ctx_new, t);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);
        }
    }

    std::ofstream f_out(params.output, std::ios::binary);
    if (!f_out.is_open()) {
        fail("error: cannot open output file %s: %s", params.output.c_str(), strerror(errno));
    }
    f_out.exceptions(std::ofstream::failbit);

    write_metadata(ctx_out, f_out);
    for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
        write_tensor_passthrough(f_out, input_shards, gguf_get_tensor_name(ctx_out, i));
    }

    f_out.close();
    gguf_free(ctx_out);
    ggml_free(ctx_new);
    for (const auto & shard : input_shards) {
        gguf_free(shard.ctx_gguf);
        ggml_free(shard.ctx_meta);
    }
}

static void extract_expert(const moe_components_params & params) {
    auto input_shards = load_input_shards(params.model);
    auto * ctx_in = input_shards[0].ctx_gguf;
    int n_expert = expert_count(ctx_in);
    if (params.expert_id < 0 || params.expert_id >= n_expert) {
        fail("error: expert %d out of range [0, %d)", params.expert_id, n_expert);
    }

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    remove_split_metadata(ctx_out);
    update_expert_metadata(ctx_out, 1);

    size_t tensor_count_hint = 0;
    for (const auto & shard : input_shards) {
        int n_tensors = gguf_get_n_tensors(shard.ctx_gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(shard.ctx_gguf, i);
            if (is_expert_tensor(name) || is_router_gate(name)) {
                tensor_count_hint++;
            }
        }
    }

    auto * ctx_new = make_tensor_context(tensor_count_hint);
    for (const auto & shard : input_shards) {
        int n_tensors = gguf_get_n_tensors(shard.ctx_gguf);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(shard.ctx_gguf, i);
            if (!(is_expert_tensor(name) || is_router_gate(name))) {
                continue;
            }

            struct ggml_tensor * t = ggml_get_tensor(shard.ctx_meta, name);
            int n_dims = ggml_n_dims(t);
            int expert_dim = n_dims - 1;
            if (t->ne[expert_dim] != n_expert) {
                fail("error: tensor %s has unexpected expert dimension %lld (expected %d)",
                    name, (long long) t->ne[expert_dim], n_expert);
            }

            int64_t ne_new[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (int d = 0; d < n_dims; d++) {
                ne_new[d] = t->ne[d];
            }
            ne_new[expert_dim] = 1;
            struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims, ne_new);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);
        }
    }

    std::ofstream f_out(params.output, std::ios::binary);
    if (!f_out.is_open()) {
        fail("error: cannot open output file %s: %s", params.output.c_str(), strerror(errno));
    }
    f_out.exceptions(std::ofstream::failbit);

    write_metadata(ctx_out, f_out);
    for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
        const char * name = gguf_get_tensor_name(ctx_out, i);
        int shard_index = -1;
        int tensor_index = -1;
        const input_shard * source_shard =
            find_input_shard_for_tensor(input_shards, name, &shard_index, &tensor_index);
        if (!source_shard) {
            fail("error: tensor %s not found in input", name);
        }

        struct ggml_tensor * t_in = ggml_get_tensor(source_shard->ctx_meta, name);
        size_t bytes_per_expert = ggml_nbytes(t_in) / n_expert;
        size_t offset =
            gguf_get_data_offset(source_shard->ctx_gguf) +
            gguf_get_tensor_offset(source_shard->ctx_gguf, tensor_index) +
            (size_t) params.expert_id * bytes_per_expert;

        std::ifstream f_in(source_shard->path, std::ios::binary);
        if (!f_in.is_open()) {
            fail("error: cannot open %s", source_shard->path.c_str());
        }
        std::vector<uint8_t> buf(bytes_per_expert);
        f_in.seekg(offset);
        f_in.read((char *) buf.data(), bytes_per_expert);
        f_out.write((const char *) buf.data(), bytes_per_expert);
        zeros(f_out, GGML_PAD(bytes_per_expert, GGUF_DEFAULT_ALIGNMENT) - bytes_per_expert);
    }

    f_out.close();
    gguf_free(ctx_out);
    ggml_free(ctx_new);
    for (const auto & shard : input_shards) {
        gguf_free(shard.ctx_gguf);
        ggml_free(shard.ctx_meta);
    }
}

static void assemble(const moe_components_params & params) {
    auto trunk_shards = load_input_shards(params.trunk);
    if (trunk_shards.size() != 1) {
        fail("error: assembled trunk input must be a single GGUF");
    }
    auto * ctx_trunk = trunk_shards[0].ctx_gguf;
    std::vector<input_shard> expert_shards;
    for (const auto & path : params.expert_files) {
        auto shards = load_input_shards(path);
        if (shards.size() != 1) {
            fail("error: expert component %s must be a single GGUF", path.c_str());
        }
        expert_shards.push_back(shards[0]);
    }

    if (expert_shards.empty()) {
        fail("error: at least one expert file is required");
    }

    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_trunk);
    remove_split_metadata(ctx_out);
    update_expert_metadata(ctx_out, (int) expert_shards.size());

    int trunk_tensor_count = 0;
    int expert_tensor_count = 0;
    for (int i = 0; i < gguf_get_n_tensors(ctx_trunk); i++) {
        const char * name = gguf_get_tensor_name(ctx_trunk, i);
        if (!(is_expert_tensor(name) || is_router_gate(name))) {
            trunk_tensor_count++;
        }
    }
    for (int i = 0; i < gguf_get_n_tensors(expert_shards[0].ctx_gguf); i++) {
        const char * name = gguf_get_tensor_name(expert_shards[0].ctx_gguf, i);
        if (is_expert_tensor(name) || is_router_gate(name)) {
            expert_tensor_count++;
        }
    }

    auto * ctx_new = make_tensor_context(trunk_tensor_count + expert_tensor_count);
    for (int i = 0; i < gguf_get_n_tensors(ctx_trunk); i++) {
        const char * name = gguf_get_tensor_name(ctx_trunk, i);
        if (is_expert_tensor(name) || is_router_gate(name)) {
            continue;
        }
        struct ggml_tensor * t = ggml_get_tensor(trunk_shards[0].ctx_meta, name);
        struct ggml_tensor * t_new = ggml_dup_tensor(ctx_new, t);
        ggml_set_name(t_new, name);
        gguf_add_tensor(ctx_out, t_new);
    }

    int n_experts_selected = (int) expert_shards.size();
    for (int i = 0; i < gguf_get_n_tensors(expert_shards[0].ctx_gguf); i++) {
        const char * name = gguf_get_tensor_name(expert_shards[0].ctx_gguf, i);
        if (!(is_expert_tensor(name) || is_router_gate(name))) {
            continue;
        }
        struct ggml_tensor * t = ggml_get_tensor(expert_shards[0].ctx_meta, name);
        int64_t ne_new[GGML_MAX_DIMS] = {1, 1, 1, 1};
        int n_dims_new = expanded_component_dims(t, n_experts_selected, ne_new);
        struct ggml_tensor * t_new = ggml_new_tensor(ctx_new, t->type, n_dims_new, ne_new);
        ggml_set_name(t_new, name);
        gguf_add_tensor(ctx_out, t_new);
    }

    std::ofstream f_out(params.output, std::ios::binary);
    if (!f_out.is_open()) {
        fail("error: cannot open output file %s: %s", params.output.c_str(), strerror(errno));
    }
    f_out.exceptions(std::ofstream::failbit);

    write_metadata(ctx_out, f_out);
    for (int i = 0; i < gguf_get_n_tensors(ctx_out); i++) {
        const char * name = gguf_get_tensor_name(ctx_out, i);
        if (!(is_expert_tensor(name) || is_router_gate(name))) {
            write_tensor_passthrough(f_out, trunk_shards, name);
            continue;
        }

        size_t bytes_per_expert = 0;
        for (int e = 0; e < (int) expert_shards.size(); e++) {
            int tensor_index = gguf_find_tensor(expert_shards[e].ctx_gguf, name);
            if (tensor_index < 0) {
                fail("error: expert file %s missing tensor %s", expert_shards[e].path.c_str(), name);
            }
            struct ggml_tensor * t_in = ggml_get_tensor(expert_shards[e].ctx_meta, name);
            size_t n_bytes = ggml_nbytes(t_in);
            if (bytes_per_expert == 0) {
                bytes_per_expert = n_bytes;
            } else if (bytes_per_expert != n_bytes) {
                fail("error: inconsistent tensor size for %s across expert files", name);
            }

            size_t offset =
                gguf_get_data_offset(expert_shards[e].ctx_gguf) +
                gguf_get_tensor_offset(expert_shards[e].ctx_gguf, tensor_index);
            std::ifstream f_in(expert_shards[e].path, std::ios::binary);
            if (!f_in.is_open()) {
                fail("error: cannot open %s", expert_shards[e].path.c_str());
            }
            std::vector<uint8_t> buf(n_bytes);
            f_in.seekg(offset);
            f_in.read((char *) buf.data(), n_bytes);
            f_out.write((const char *) buf.data(), n_bytes);
        }
        size_t total_out_bytes = bytes_per_expert * expert_shards.size();
        zeros(f_out, GGML_PAD(total_out_bytes, GGUF_DEFAULT_ALIGNMENT) - total_out_bytes);
    }

    f_out.close();
    gguf_free(ctx_out);
    ggml_free(ctx_new);
    for (const auto & shard : trunk_shards) {
        gguf_free(shard.ctx_gguf);
        ggml_free(shard.ctx_meta);
    }
    for (const auto & shard : expert_shards) {
        gguf_free(shard.ctx_gguf);
        ggml_free(shard.ctx_meta);
    }
}

int main(int argc, const char ** argv) {
    moe_components_params params;
    parse_args(argc, argv, params);

    switch (params.mode) {
        case moe_components_mode::extract_trunk:
            extract_trunk(params);
            break;
        case moe_components_mode::extract_expert:
            extract_expert(params);
            break;
        case moe_components_mode::assemble:
            assemble(params);
            break;
    }

    return 0;
}
