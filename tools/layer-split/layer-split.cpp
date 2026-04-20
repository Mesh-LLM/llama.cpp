#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

struct layer_split_params {
    std::string input;
    std::string output;
    std::string output_prefix;
    int layer_start = -1;
    int layer_end   = -1;
    int n_stages    = 0;
    bool include_output = false;
    bool smoke_load = false;
    bool dry_run = false;
};

struct input_shard {
    std::string path;
    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_meta;
};

static void print_usage(const char * prog) {
    printf("\n");
    printf("usage: %s [options]\n", prog);
    printf("\n");
    printf("Write runnable layer-range GGUF shards.\n");
    printf("\n");
    printf("options:\n");
    printf("  -m, --model FILE          input GGUF file\n");
    printf("  --layer-start N           start of half-open layer range [start, end)\n");
    printf("  --layer-end N             end of half-open layer range [start, end)\n");
    printf("  --stages N                split the full model into N contiguous layer shards\n");
    printf("  --include-output          include output head tensors even on non-final shards\n");
    printf("  --smoke-load              load the written shard into a real llama context\n");
    printf("  -o, --output FILE         output file for a single range\n");
    printf("  --output-prefix PREFIX    output prefix for multi-shard mode (PREFIX-N.gguf)\n");
    printf("  --dry-run                 show plan without writing\n");
    printf("  -h, --help                show this help\n");
    printf("\n");
}

static void zeros(std::ofstream & f, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        f.write(&zero, 1);
    }
}

static void parse_args(int argc, const char ** argv, layer_split_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            params.input = argv[++i];
        } else if (arg == "--layer-start") {
            params.layer_start = std::atoi(argv[++i]);
        } else if (arg == "--layer-end") {
            params.layer_end = std::atoi(argv[++i]);
        } else if (arg == "--stages") {
            params.n_stages = std::atoi(argv[++i]);
        } else if (arg == "--include-output") {
            params.include_output = true;
        } else if (arg == "--smoke-load") {
            params.smoke_load = true;
        } else if (arg == "-o" || arg == "--output") {
            params.output = argv[++i];
        } else if (arg == "--output-prefix") {
            params.output_prefix = argv[++i];
        } else if (arg == "--dry-run") {
            params.dry_run = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            std::exit(1);
        }
    }

    if (params.input.empty()) {
        fprintf(stderr, "error: --model is required\n");
        std::exit(1);
    }

    const bool single_range = params.layer_start >= 0 || params.layer_end >= 0;
    const bool multi_stage = params.n_stages > 0;
    if (single_range == multi_stage) {
        fprintf(stderr, "error: choose exactly one of --layer-start/--layer-end or --stages\n");
        std::exit(1);
    }

    if (single_range) {
        if (params.layer_start < 0 || params.layer_end < 0 || params.layer_start >= params.layer_end) {
            fprintf(stderr, "error: invalid layer range [%d, %d)\n", params.layer_start, params.layer_end);
            std::exit(1);
        }
        if (params.output.empty()) {
            fprintf(stderr, "error: --output is required for single-range mode\n");
            std::exit(1);
        }
    } else {
        if (params.n_stages <= 0) {
            fprintf(stderr, "error: --stages must be > 0\n");
            std::exit(1);
        }
        if (params.output_prefix.empty()) {
            fprintf(stderr, "error: --output-prefix is required with --stages\n");
            std::exit(1);
        }
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
            fprintf(stderr, "error: failed to load %s\n", path.c_str());
            std::exit(1);
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
        fprintf(stderr, "error: unexpected split input file name: %s\n", input_path.c_str());
        std::exit(1);
    }

    for (int i_split = 1; i_split < n_split; ++i_split) {
        std::vector<char> split_path(PATH_MAX, 0);
        int ret = llama_split_path(split_path.data(), split_path.size(), split_prefix.data(), i_split, n_split);
        if (ret <= 0) {
            fprintf(stderr, "error: failed to derive split path for shard %d/%d\n", i_split + 1, n_split);
            std::exit(1);
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
    for (int shard_index = 0; shard_index < (int) input_shards.size(); ++shard_index) {
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

static bool parse_layer_id(const char * name, int * layer_id) {
    int parsed = -1;
    if (std::sscanf(name, "blk.%d.", &parsed) == 1 && parsed >= 0) {
        if (layer_id) {
            *layer_id = parsed;
        }
        return true;
    }
    return false;
}

static bool is_output_norm_tensor(const char * name) {
    return std::strcmp(name, "output_norm.weight") == 0;
}

static bool is_output_tensor(const char * name) {
    return std::strcmp(name, "output.weight") == 0;
}

static bool should_include_tensor(const char * name, int layer_start, int layer_end, bool include_output) {
    int layer_id = -1;
    if (parse_layer_id(name, &layer_id)) {
        return layer_id >= layer_start && layer_id < layer_end;
    }

    if ((is_output_norm_tensor(name) || is_output_tensor(name)) && !include_output) {
        return false;
    }

    return true;
}

static int get_block_count(struct gguf_context * ctx) {
    int arch_key = gguf_find_key(ctx, "general.architecture");
    if (arch_key < 0) {
        fprintf(stderr, "error: missing general.architecture\n");
        std::exit(1);
    }

    const std::string arch_name = gguf_get_val_str(ctx, arch_key);
    const std::string block_count_key = arch_name + ".block_count";
    int block_count_idx = gguf_find_key(ctx, block_count_key.c_str());
    if (block_count_idx < 0) {
        fprintf(stderr, "error: missing %s\n", block_count_key.c_str());
        std::exit(1);
    }

    return (int) gguf_get_val_u32(ctx, block_count_idx);
}

static std::vector<std::pair<int, int>> make_stage_ranges(int n_layer, int n_stages) {
    std::vector<std::pair<int, int>> ranges;
    ranges.reserve((size_t) n_stages);

    const int base = n_layer / n_stages;
    const int rem  = n_layer % n_stages;

    int start = 0;
    for (int i = 0; i < n_stages; ++i) {
        const int width = base + (i < rem ? 1 : 0);
        const int end = start + width;
        ranges.emplace_back(start, end);
        start = end;
    }

    return ranges;
}

static void write_layer_shard(
        const std::vector<input_shard> & input_shards,
        int layer_start,
        int layer_end,
        bool include_output,
        const std::string & output_path,
        bool dry_run) {
    auto * ctx_in = input_shards[0].ctx_gguf;
    auto * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);

    gguf_remove_key(ctx_out, "split.no");
    gguf_remove_key(ctx_out, "split.count");
    gguf_remove_key(ctx_out, "split.tensors.count");

    gguf_set_val_bool(ctx_out, "mesh.layer_split", true);
    gguf_set_val_i32(ctx_out, "mesh.layer_start", layer_start);
    gguf_set_val_i32(ctx_out, "mesh.layer_end", layer_end);
    gguf_set_val_bool(ctx_out, "mesh.include_output", include_output);

    int n_tensors_total = 0;
    for (const auto & input_shard : input_shards) {
        n_tensors_total += gguf_get_n_tensors(input_shard.ctx_gguf);
    }

    const size_t ctx_size = (size_t) (n_tensors_total + 64) * ggml_tensor_overhead() + 4096;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_new = ggml_init(ctx_params);

    size_t included_tensor_count = 0;
    size_t included_bytes = 0;
    size_t skipped_layer_bytes = 0;

    for (const auto & input_shard : input_shards) {
        const int n_tensors = gguf_get_n_tensors(input_shard.ctx_gguf);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(input_shard.ctx_gguf, i);
            struct ggml_tensor * t = ggml_get_tensor(input_shard.ctx_meta, name);

            if (!should_include_tensor(name, layer_start, layer_end, include_output)) {
                skipped_layer_bytes += ggml_nbytes(t);
                continue;
            }

            struct ggml_tensor * t_new = ggml_dup_tensor(ctx_new, t);
            ggml_set_name(t_new, name);
            gguf_add_tensor(ctx_out, t_new);

            included_tensor_count++;
            included_bytes += ggml_nbytes(t);
        }
    }

    printf("\n--- shard [%d, %d) -> %s ---\n", layer_start, layer_end, output_path.c_str());
    printf("  include_output: %s\n", include_output ? "true" : "false");
    printf("  included tensors: %zu\n", included_tensor_count);
    printf("  included bytes: %.2f MiB\n", included_bytes / (1024.0 * 1024.0));
    printf("  skipped bytes: %.2f MiB\n", skipped_layer_bytes / (1024.0 * 1024.0));

    if (dry_run) {
        printf("  (dry run, not writing)\n");
        gguf_free(ctx_out);
        ggml_free(ctx_new);
        return;
    }

    std::ofstream f_out(output_path, std::ios::binary);
    if (!f_out.is_open()) {
        fprintf(stderr, "error: cannot open output file %s\n", output_path.c_str());
        std::exit(1);
    }
    f_out.exceptions(std::ofstream::failbit);

    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size);
    gguf_get_meta_data(ctx_out, meta_data.data());
    f_out.write((const char *) meta_data.data(), meta_size);

    std::vector<uint8_t> read_buf;
    std::vector<std::ifstream> f_inputs;
    f_inputs.reserve(input_shards.size());
    for (const auto & input_shard : input_shards) {
        f_inputs.emplace_back(input_shard.path, std::ios::binary);
        if (!f_inputs.back().is_open()) {
            fprintf(stderr, "error: cannot open %s\n", input_shard.path.c_str());
            std::exit(1);
        }
    }

    for (int i = 0; i < gguf_get_n_tensors(ctx_out); ++i) {
        const char * name = gguf_get_tensor_name(ctx_out, i);

        int shard_index = -1;
        int i_in = -1;
        const input_shard * source_shard = find_input_shard_for_tensor(input_shards, name, &shard_index, &i_in);
        if (!source_shard) {
            fprintf(stderr, "error: tensor %s not found in input\n", name);
            std::exit(1);
        }

        struct ggml_tensor * t_in = ggml_get_tensor(source_shard->ctx_meta, name);
        const size_t offset_in = gguf_get_data_offset(source_shard->ctx_gguf) +
            gguf_get_tensor_offset(source_shard->ctx_gguf, i_in);
        const size_t n_bytes = ggml_nbytes(t_in);

        read_buf.resize(n_bytes);
        std::ifstream & f_in = f_inputs[(size_t) shard_index];
        f_in.seekg(offset_in);
        f_in.read((char *) read_buf.data(), n_bytes);
        f_out.write((const char *) read_buf.data(), n_bytes);
        zeros(f_out, GGML_PAD(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes);
    }

    f_out.close();
    gguf_free(ctx_out);
    ggml_free(ctx_new);
}

static bool smoke_load_shard(const std::string & path, int layer_start, int layer_end) {
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 32;
    ctx_params.n_ubatch = 32;

    llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "error: smoke load failed to load model: %s\n", path.c_str());
        return false;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: smoke load failed to create context: %s\n", path.c_str());
        llama_model_free(model);
        return false;
    }

    llama_set_compute_range(ctx, layer_start, layer_end);

    llama_free(ctx);
    llama_model_free(model);
    return true;
}

int main(int argc, const char ** argv) {
    layer_split_params params;
    parse_args(argc, argv, params);

    auto input_shards = load_input_shards(params.input);
    const int n_layer = get_block_count(input_shards[0].ctx_gguf);

    printf("Model: %s\n", params.input.c_str());
    printf("Block count: %d\n", n_layer);

    if (params.smoke_load) {
        llama_backend_init();
    }

    if (params.n_stages > 0) {
        if (params.n_stages > n_layer) {
            fprintf(stderr, "error: --stages %d exceeds block count %d\n", params.n_stages, n_layer);
            return 1;
        }

        const auto ranges = make_stage_ranges(n_layer, params.n_stages);
        for (int i = 0; i < (int) ranges.size(); ++i) {
            const auto [start, end] = ranges[(size_t) i];
            char path_buf[1024];
            std::snprintf(path_buf, sizeof(path_buf), "%s-%d.gguf", params.output_prefix.c_str(), i);
            const bool include_output = params.include_output || end == n_layer;
            write_layer_shard(input_shards, start, end, include_output, path_buf, params.dry_run);
            if (!params.dry_run && params.smoke_load) {
                const bool ok = smoke_load_shard(path_buf, start, end);
                printf("  smoke load: %s\n", ok ? "ok" : "failed");
                if (!ok) {
                    return 1;
                }
            }
        }
    } else {
        if (params.layer_end > n_layer) {
            fprintf(stderr, "error: layer range [%d, %d) exceeds block count %d\n", params.layer_start, params.layer_end, n_layer);
            return 1;
        }
        const bool include_output = params.include_output || params.layer_end == n_layer;
        write_layer_shard(input_shards, params.layer_start, params.layer_end, include_output, params.output, params.dry_run);
        if (!params.dry_run && params.smoke_load) {
            const bool ok = smoke_load_shard(params.output, params.layer_start, params.layer_end);
            printf("  smoke load: %s\n", ok ? "ok" : "failed");
            if (!ok) {
                return 1;
            }
        }
    }

    for (const auto & input_shard : input_shards) {
        gguf_free(input_shard.ctx_gguf);
        ggml_free(input_shard.ctx_meta);
    }

    if (params.smoke_load) {
        llama_backend_free();
    }

    printf("\nDone.\n");
    return 0;
}
