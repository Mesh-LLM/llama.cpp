#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"
#include "llama-model.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <list>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#endif

struct ranked_expert {
    int expert_id = -1;
    double gate_mass = 0.0;
};

struct tensor_source {
    uint64_t file_offset = 0;
    size_t bytes = 0;
    size_t slice_bytes = 0;
    bool is_lazy_expert = false;
};

struct manual_model_source {
    std::string model_path;
    std::ifstream file;
    struct gguf_context * metadata = nullptr;
    struct ggml_context * meta_ctx = nullptr;
    int32_t n_expert = 0;
    std::vector<int> hot_experts;
    std::unordered_map<std::string, tensor_source> tensors;
};

struct slice_range {
    const char * tensor_name = nullptr;
    uint8_t * data = nullptr;
    size_t bytes = 0;
};

struct callback_data {
    std::vector<uint8_t> scratch;
    bool collect = true;
    int top_k = 0;
    int n_expert = 0;
    const std::set<int> * hot_experts = nullptr;
    std::set<int> * resident_cold = nullptr;
    const std::unordered_map<int, std::vector<slice_range>> * ranges = nullptr;
    manual_model_source * source = nullptr;
    std::set<int> step_selected;
    std::set<int> step_newly_fetched;
};

struct tensor_region {
    const char * tensor_name = nullptr;
    uint8_t * data = nullptr;
    size_t bytes = 0;
};

struct step_metrics {
    std::string phase;
    int token_index = -1;
    double elapsed_s = 0.0;
    std::vector<int> selected_experts;
    std::vector<int> newly_fetched_experts;
    uint64_t resident_expert_bytes_before_eviction = 0;
    uint64_t resident_expert_bytes_after_eviction = 0;
};

struct experiment_stats {
    int hot_expert_count = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    uint64_t expert_buffer_bytes = 0;
    uint64_t initial_resident_expert_bytes = 0;
    uint64_t final_resident_expert_bytes = 0;
    uint64_t min_resident_expert_bytes = 0;
    uint64_t max_resident_expert_bytes = 0;
    int cold_activation_count = 0;
    int unique_cold_experts = 0;
    uint64_t estimated_fetch_bytes = 0;
    int fetch_event_count = 0;
    double fetch_stall_upper_bound_s = 0.0;
    int prefill_cold_activation_count = 0;
    int decode_cold_activation_count = 0;
    int prefill_unique_cold_experts = 0;
    int decode_unique_cold_experts = 0;
    uint64_t prefill_fetch_bytes = 0;
    uint64_t decode_fetch_bytes = 0;
    int prefill_fetch_event_count = 0;
    int decode_fetch_event_count = 0;
    double prefill_fetch_stall_upper_bound_s = 0.0;
    double decode_fetch_stall_upper_bound_s = 0.0;
    std::vector<int> post_prefill_probe_expert_ids;
    std::vector<uint64_t> post_prefill_probe_resident_bytes;
    long minor_faults_delta = 0;
    long major_faults_delta = 0;
    std::vector<step_metrics> steps;
    std::string output_text;
};

struct owned_tensor_backing {
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    uint8_t * data = nullptr;
    size_t buffer_size = 0;
    size_t alloc_size = 0;
};

static void madvise_region(uint8_t * ptr, size_t len, int advice);
static size_t page_size();

static std::string ranking_file_path;
static std::string json_out_path;
static int hot_expert_count = 0;
static int cold_cache_expert_count = 0;
static bool random_hot = false;
static uint32_t hot_seed = 0;
static bool decode_only_measurement = false;
static bool hard_evict = false;
static bool custom_expert_backing = false;
static uint64_t hard_evict_attempt_count = 0;
static uint64_t hard_evict_success_count = 0;

static bool is_lazy_expert_tensor_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strstr(name, ".ffn_gate_exps") != nullptr ||
           std::strstr(name, ".ffn_down_exps") != nullptr ||
           std::strstr(name, ".ffn_up_exps") != nullptr ||
           std::strstr(name, ".ffn_gate_up_exps") != nullptr ||
           std::strstr(name, ".ffn_exp_probs_b") != nullptr;
}

static void read_bytes(std::ifstream & file, uint64_t offset, void * dst, size_t len) {
    file.clear();
    file.seekg((std::streamoff) offset, std::ios::beg);
    if (!file) {
        throw std::runtime_error("failed to seek model file");
    }
    file.read(reinterpret_cast<char *>(dst), (std::streamsize) len);
    if (!file) {
        throw std::runtime_error("failed to read model file bytes");
    }
}

static uint8_t * allocate_host_backing(size_t size, size_t * alloc_size_out) {
#if defined(__APPLE__)
    const size_t ps = page_size();
    const size_t alloc_size = ((size + ps - 1) / ps) * ps;
    vm_address_t address = 0;
    kern_return_t rc = vm_allocate((vm_map_t) mach_task_self(), &address, alloc_size, VM_FLAGS_ANYWHERE);
    if (rc != KERN_SUCCESS) {
        throw std::runtime_error("failed to vm_allocate host backing");
    }
    *alloc_size_out = alloc_size;
    return reinterpret_cast<uint8_t *>(address);
#else
    void * ptr = nullptr;
    const size_t alignment = std::max<size_t>(page_size(), 64);
    if (posix_memalign(&ptr, alignment, size) != 0 || ptr == nullptr) {
        throw std::runtime_error("failed to allocate host backing");
    }
    *alloc_size_out = size;
    return reinterpret_cast<uint8_t *>(ptr);
#endif
}

static void free_host_backing(uint8_t * ptr, size_t alloc_size) {
    if (!ptr) {
        return;
    }
#if defined(__APPLE__)
    vm_deallocate((vm_map_t) mach_task_self(), (vm_address_t) ptr, alloc_size);
#else
    free(ptr);
#endif
}

static manual_model_source build_manual_model_source(
        const std::string & model_path,
        const std::vector<int> & hot_experts) {
    manual_model_source source;
    source.model_path = model_path;
    source.hot_experts = hot_experts;

    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx = */ &source.meta_ctx,
    };
    source.metadata = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!source.metadata) {
        throw std::runtime_error("failed to read GGUF metadata from " + model_path);
    }

    source.file.open(model_path, std::ios::binary);
    if (!source.file.is_open()) {
        throw std::runtime_error("failed to open model file: " + model_path);
    }

    const size_t data_offset = gguf_get_data_offset(source.metadata);
    for (ggml_tensor * cur = ggml_get_first_tensor(source.meta_ctx); cur; cur = ggml_get_next_tensor(source.meta_ctx, cur)) {
        const char * name = ggml_get_name(cur);
        const int n_dims = ggml_n_dims(cur);
        if (source.n_expert == 0 && is_lazy_expert_tensor_name(name) && n_dims > 0) {
            source.n_expert = (int32_t) cur->ne[n_dims - 1];
        }

        const int64_t tensor_id = gguf_find_tensor(source.metadata, name);
        if (tensor_id < 0) {
            throw std::runtime_error(std::string("tensor metadata not found: ") + name);
        }

        size_t bytes = ggml_nbytes(cur);
        tensor_source tensor;
        tensor.file_offset = data_offset + gguf_get_tensor_offset(source.metadata, tensor_id);
        tensor.bytes = bytes;
        tensor.is_lazy_expert = is_lazy_expert_tensor_name(name) && source.n_expert > 0 && cur->ne[n_dims - 1] == source.n_expert;
        tensor.slice_bytes = tensor.is_lazy_expert ? bytes / (size_t) source.n_expert : 0;
        source.tensors.emplace(name, tensor);
    }

    if (source.n_expert <= 0) {
        throw std::runtime_error("failed to determine expert count from model metadata");
    }

    return source;
}

static void zero_tensor(ggml_tensor * tensor, std::vector<uint8_t> & scratch) {
    const size_t n_bytes = ggml_nbytes(tensor);
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        std::memset(tensor->data, 0, n_bytes);
        return;
    }
    scratch.assign(n_bytes, 0);
    ggml_backend_tensor_set(tensor, scratch.data(), 0, n_bytes);
}

static void set_tensor_from_source(ggml_tensor * tensor, void * userdata) {
    auto * source = reinterpret_cast<manual_model_source *>(userdata);
    const std::string name = ggml_get_name(tensor);
    auto it = source->tensors.find(name);
    if (it == source->tensors.end()) {
        throw std::runtime_error("tensor not found in manual source: " + name);
    }

    const tensor_source & info = it->second;
    const size_t n_bytes = ggml_nbytes(tensor);
    if (n_bytes != info.bytes) {
        throw std::runtime_error("tensor size mismatch for " + name);
    }

    std::vector<uint8_t> device_buffer;
    uint8_t * dst = nullptr;
    if (ggml_backend_buffer_is_host(tensor->buffer)) {
        dst = reinterpret_cast<uint8_t *>(tensor->data);
    } else {
        device_buffer.assign(n_bytes, 0);
        dst = device_buffer.data();
    }

    if (!info.is_lazy_expert) {
        if (ggml_backend_buffer_is_host(tensor->buffer)) {
            std::memset(dst, 0, n_bytes);
        }
        read_bytes(source->file, info.file_offset, dst, n_bytes);
    } else {
        for (int expert_id : source->hot_experts) {
            if (expert_id < 0 || expert_id >= source->n_expert) {
                continue;
            }
            read_bytes(
                source->file,
                info.file_offset + (uint64_t) info.slice_bytes * (uint64_t) expert_id,
                dst + (size_t) expert_id * info.slice_bytes,
                info.slice_bytes);
        }
    }

    if (!ggml_backend_buffer_is_host(tensor->buffer)) {
        ggml_backend_tensor_set(tensor, dst, 0, n_bytes);
    }
}

static void load_cold_expert_into_ranges(
        manual_model_source & source,
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int expert_id) {
    auto it = ranges.find(expert_id);
    if (it == ranges.end()) {
        return;
    }

    for (const auto & range : it->second) {
#if defined(__APPLE__) && defined(MADV_FREE_REUSE)
        madvise_region(range.data, range.bytes, MADV_FREE_REUSE);
#endif
        auto source_it = source.tensors.find(range.tensor_name);
        if (source_it == source.tensors.end()) {
            throw std::runtime_error(std::string("missing tensor source for ") + range.tensor_name);
        }
        const tensor_source & tensor = source_it->second;
        read_bytes(
            source.file,
            tensor.file_offset + (uint64_t) tensor.slice_bytes * (uint64_t) expert_id,
            range.data,
            range.bytes);
    }
}

static bool moe_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = reinterpret_cast<callback_data *>(user_data);

    if (!cb->collect) {
        return false;
    }

    if (ask) {
        if (strstr(t->name, "ffn_moe_probs") != nullptr &&
            strstr(t->name, "masked") == nullptr &&
            strstr(t->name, "biased") == nullptr) {
            return true;
        }
        return false;
    }

    if (strstr(t->name, "ffn_moe_probs") == nullptr) {
        return true;
    }
    if (strstr(t->name, "masked") != nullptr || strstr(t->name, "biased") != nullptr) {
        return true;
    }

    const char * dash = strrchr(t->name, '-');
    int layer_id = dash ? std::atoi(dash + 1) : -1;
    (void) layer_id;

    int n_expert = (int) t->ne[0];
    int n_tokens = (int) t->ne[1];

    float * data_ptr = nullptr;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        data_ptr = reinterpret_cast<float *>(t->data);
    } else {
        size_t n_bytes = ggml_nbytes(t);
        cb->scratch.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->scratch.data(), 0, n_bytes);
        data_ptr = reinterpret_cast<float *>(cb->scratch.data());
    }

    for (int tok = 0; tok < n_tokens; ++tok) {
        const float * probs = data_ptr + tok * n_expert;
        std::vector<int> ids(n_expert);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(
            ids.begin(),
            ids.begin() + std::min(cb->top_k, n_expert),
            ids.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });

        for (int i = 0; i < std::min(cb->top_k, n_expert); ++i) {
            int expert_id = ids[i];
            cb->step_selected.insert(expert_id);
            if (cb->hot_experts && cb->hot_experts->count(expert_id) > 0) {
                continue;
            }
            if (cb->resident_cold && cb->resident_cold->count(expert_id) > 0) {
                continue;
            }
            if (cb->source && cb->ranges) {
                load_cold_expert_into_ranges(*cb->source, *cb->ranges, expert_id);
            }
            if (cb->resident_cold) {
                cb->resident_cold->insert(expert_id);
            }
            cb->step_newly_fetched.insert(expert_id);
        }
    }
    return true;
}

static std::vector<ranked_expert> load_ranking(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open ranking file: " + path);
    }

    std::vector<ranked_expert> ranking;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        int expert_id = -1;
        double gate_mass = 0.0;
        if (std::sscanf(line.c_str(), "%d,%lf", &expert_id, &gate_mass) == 2) {
            ranking.push_back({expert_id, gate_mass});
        }
    }

    std::sort(ranking.begin(), ranking.end(), [](const ranked_expert & a, const ranked_expert & b) {
        return a.gate_mass > b.gate_mass;
    });
    return ranking;
}

static std::vector<int> choose_hot_experts(const std::vector<ranked_expert> & ranking) {
    if (hot_expert_count <= 0) {
        throw std::runtime_error("--hot-experts must be > 0");
    }
    if ((size_t) hot_expert_count > ranking.size()) {
        throw std::runtime_error("requested more hot experts than available in ranking");
    }

    std::vector<int> chosen;
    chosen.reserve(hot_expert_count);

    if (!random_hot) {
        for (int i = 0; i < hot_expert_count; ++i) {
            chosen.push_back(ranking[i].expert_id);
        }
        std::sort(chosen.begin(), chosen.end());
        return chosen;
    }

    std::vector<int> pool;
    pool.reserve(ranking.size());
    for (const auto & item : ranking) {
        pool.push_back(item.expert_id);
    }

    std::mt19937 rng(hot_seed);
    std::shuffle(pool.begin(), pool.end(), rng);
    chosen.assign(pool.begin(), pool.begin() + hot_expert_count);
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

static size_t page_size() {
#if defined(__unix__) || defined(__APPLE__)
    long size = sysconf(_SC_PAGESIZE);
    return size > 0 ? (size_t) size : 4096;
#else
    return 4096;
#endif
}

static void page_align(uint8_t * ptr, size_t len, uint8_t ** start_out, size_t * len_out) {
    const size_t ps = page_size();
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t start = addr & ~(uintptr_t(ps - 1));
    uintptr_t end = (addr + len + ps - 1) & ~(uintptr_t(ps - 1));
    *start_out = reinterpret_cast<uint8_t *>(start);
    *len_out = end - start;
}

static bool is_page_aligned_region(uint8_t * ptr, size_t len) {
    const size_t ps = page_size();
    if (len == 0) {
        return false;
    }
    return (reinterpret_cast<uintptr_t>(ptr) % ps) == 0 && (len % ps) == 0;
}

static bool page_aligned_interior(uint8_t * ptr, size_t len, uint8_t ** start_out, size_t * len_out) {
    const size_t ps = page_size();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = start + len;
    uintptr_t aligned_start = (start + ps - 1) & ~(uintptr_t(ps - 1));
    uintptr_t aligned_end = end & ~(uintptr_t(ps - 1));
    if (aligned_end <= aligned_start) {
        *start_out = nullptr;
        *len_out = 0;
        return false;
    }
    *start_out = reinterpret_cast<uint8_t *>(aligned_start);
    *len_out = aligned_end - aligned_start;
    return true;
}

static void madvise_region(uint8_t * ptr, size_t len, int advice) {
#if defined(__unix__) || defined(__APPLE__)
    uint8_t * aligned = nullptr;
    size_t aligned_len = 0;
    page_align(ptr, len, &aligned, &aligned_len);
    if (aligned_len == 0) {
        return;
    }
#if defined(__APPLE__)
    if (advice == MADV_DONTNEED) {
        // On Darwin, MADV_DONTNEED alone often leaves pages resident until memory pressure.
        // For this experiment we want a stronger best-effort eviction signal for cold experts.
        (void) madvise(aligned, aligned_len, MADV_FREE_REUSABLE);
        (void) madvise(aligned, aligned_len, MADV_FREE);
#ifdef MADV_PAGEOUT
        (void) madvise(aligned, aligned_len, MADV_PAGEOUT);
#endif
        return;
    }
#endif
    if (madvise(aligned, aligned_len, advice) != 0) {
        LOG_WRN("madvise failed on %p (%zu bytes): %s\n", (void *) aligned, aligned_len, std::strerror(errno));
    }
#else
    (void) ptr;
    (void) len;
    (void) advice;
#endif
}

static bool hard_evict_region(uint8_t * ptr, size_t len) {
#if defined(__APPLE__)
    uint8_t * region_ptr = ptr;
    size_t region_len = len;
    if (!is_page_aligned_region(region_ptr, region_len) &&
        !page_aligned_interior(ptr, len, &region_ptr, &region_len)) {
        return false;
    }
    vm_address_t address = (vm_address_t) region_ptr;
    kern_return_t rc = vm_allocate(
        (vm_map_t) mach_task_self(),
        &address,
        region_len,
        VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE);
    if (rc != KERN_SUCCESS) {
        return false;
    }
    madvise_region(region_ptr, region_len, MADV_FREE_REUSABLE);
    madvise_region(region_ptr, region_len, MADV_FREE);
#ifdef MADV_PAGEOUT
    madvise_region(region_ptr, region_len, MADV_PAGEOUT);
#endif
    return true;
#else
    (void) ptr;
    (void) len;
    return false;
#endif
}

static void touch_region(uint8_t * ptr, size_t len) {
    const size_t ps = page_size();
    volatile uint8_t sink = 0;
    for (size_t offset = 0; offset < len; offset += ps) {
        sink ^= ptr[offset];
    }
    if (len > 0) {
        sink ^= ptr[len - 1];
    }
    (void) sink;
}

static size_t expert_slice_bytes(const ggml_tensor * tensor, int32_t n_expert) {
    if (!tensor || !tensor->data || n_expert <= 0) {
        return 0;
    }
    size_t total = ggml_nbytes(tensor);
    return total / (size_t) n_expert;
}

static void add_tensor_ranges(
        std::unordered_map<int, std::vector<slice_range>> & ranges,
        ggml_tensor * tensor,
        int32_t n_expert) {
    if (!tensor || !tensor->data || !tensor->buffer) {
        return;
    }

    const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
    if (!buffer_name || std::strncmp(buffer_name, "CPU", 3) != 0) {
        return;
    }

    size_t slice_bytes = expert_slice_bytes(tensor, n_expert);
    if (slice_bytes == 0) {
        return;
    }

    auto * base = reinterpret_cast<uint8_t *>(tensor->data);
    for (int expert_id = 0; expert_id < n_expert; ++expert_id) {
        ranges[expert_id].push_back(
            slice_range{
                ggml_get_name(tensor),
                base + slice_bytes * (size_t) expert_id,
                slice_bytes,
            }
        );
    }
}

static void add_tensor_region(
        std::vector<tensor_region> & regions,
        std::set<uint8_t *> & seen_bases,
        ggml_tensor * tensor) {
    if (!tensor || !tensor->data || !tensor->buffer) {
        return;
    }

    const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
    if (!buffer_name || std::strncmp(buffer_name, "CPU", 3) != 0) {
        return;
    }

    auto * base = reinterpret_cast<uint8_t *>(tensor->data);
    if (!seen_bases.insert(base).second) {
        return;
    }

    regions.push_back(
        tensor_region{
            ggml_get_name(tensor),
            base,
            ggml_nbytes(tensor),
        }
    );
}

static std::unordered_map<int, std::vector<slice_range>> collect_expert_ranges(llama_model * model) {
    std::unordered_map<int, std::vector<slice_range>> ranges;
    int32_t n_expert = (int32_t) model->hparams.n_expert;

    for (auto & layer : model->layers) {
        add_tensor_ranges(ranges, layer.ffn_gate_exps, n_expert);
        add_tensor_ranges(ranges, layer.ffn_down_exps, n_expert);
        add_tensor_ranges(ranges, layer.ffn_up_exps, n_expert);
        add_tensor_ranges(ranges, layer.ffn_gate_up_exps, n_expert);
        add_tensor_ranges(ranges, layer.ffn_gate_exps_b, n_expert);
        add_tensor_ranges(ranges, layer.ffn_down_exps_b, n_expert);
        add_tensor_ranges(ranges, layer.ffn_up_exps_b, n_expert);
        add_tensor_ranges(ranges, layer.ffn_gate_up_exps_b, n_expert);
        add_tensor_ranges(ranges, layer.ffn_exp_probs_b, n_expert);
        add_tensor_ranges(ranges, layer.ffn_gate_exps_s, n_expert);
        add_tensor_ranges(ranges, layer.ffn_down_exps_s, n_expert);
        add_tensor_ranges(ranges, layer.ffn_up_exps_s, n_expert);
        add_tensor_ranges(ranges, layer.ffn_gate_exps_in_s, n_expert);
        add_tensor_ranges(ranges, layer.ffn_down_exps_in_s, n_expert);
        add_tensor_ranges(ranges, layer.ffn_up_exps_in_s, n_expert);
    }

    return ranges;
}

static std::vector<tensor_region> collect_expert_tensor_regions(llama_model * model) {
    std::vector<tensor_region> regions;
    std::set<uint8_t *> seen_bases;

    for (auto & layer : model->layers) {
        add_tensor_region(regions, seen_bases, layer.ffn_gate_exps);
        add_tensor_region(regions, seen_bases, layer.ffn_down_exps);
        add_tensor_region(regions, seen_bases, layer.ffn_up_exps);
        add_tensor_region(regions, seen_bases, layer.ffn_gate_up_exps);
        add_tensor_region(regions, seen_bases, layer.ffn_gate_exps_b);
        add_tensor_region(regions, seen_bases, layer.ffn_down_exps_b);
        add_tensor_region(regions, seen_bases, layer.ffn_up_exps_b);
        add_tensor_region(regions, seen_bases, layer.ffn_gate_up_exps_b);
        add_tensor_region(regions, seen_bases, layer.ffn_exp_probs_b);
        add_tensor_region(regions, seen_bases, layer.ffn_gate_exps_s);
        add_tensor_region(regions, seen_bases, layer.ffn_down_exps_s);
        add_tensor_region(regions, seen_bases, layer.ffn_up_exps_s);
        add_tensor_region(regions, seen_bases, layer.ffn_gate_exps_in_s);
        add_tensor_region(regions, seen_bases, layer.ffn_down_exps_in_s);
        add_tensor_region(regions, seen_bases, layer.ffn_up_exps_in_s);
    }

    return regions;
}

static uint64_t bytes_for_expert(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int expert_id) {
    auto it = ranges.find(expert_id);
    if (it == ranges.end()) {
        return 0;
    }

    uint64_t total = 0;
    for (const auto & range : it->second) {
        total += (uint64_t) range.bytes;
    }
    return total;
}

static uint64_t total_expert_buffer_bytes(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int32_t n_expert) {
    uint64_t total = 0;
    for (int expert_id = 0; expert_id < n_expert; ++expert_id) {
        total += bytes_for_expert(ranges, expert_id);
    }
    return total;
}

static uint64_t resident_bytes_for_region(const tensor_region & region) {
#if defined(__unix__) || defined(__APPLE__)
    if (!region.data || region.bytes == 0) {
        return 0;
    }

    uint8_t * aligned = nullptr;
    size_t aligned_len = 0;
    page_align(region.data, region.bytes, &aligned, &aligned_len);
    if (aligned_len == 0) {
        return 0;
    }

    const size_t ps = page_size();
    const size_t n_pages = aligned_len / ps;
    std::vector<unsigned char> vec(n_pages);
    if (mincore(aligned, aligned_len, reinterpret_cast<char *>(vec.data())) != 0) {
        LOG_WRN("mincore failed on %p (%zu bytes): %s\n", (void *) aligned, aligned_len, std::strerror(errno));
        return 0;
    }

    uint64_t resident = 0;
    for (size_t i = 0; i < n_pages; ++i) {
        if (vec[i] & 1) {
            resident += ps;
        }
    }
    return resident;
#else
    (void) region;
    return 0;
#endif
}

static uint64_t resident_bytes_for_expert_regions(const std::vector<tensor_region> & regions) {
    uint64_t total = 0;
    for (const auto & region : regions) {
        total += resident_bytes_for_region(region);
    }
    return total;
}

static uint64_t resident_bytes_for_expert(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int expert_id) {
    auto it = ranges.find(expert_id);
    if (it == ranges.end()) {
        return 0;
    }

    uint64_t total = 0;
    for (const auto & range : it->second) {
        total += resident_bytes_for_region(
            tensor_region{
                range.tensor_name,
                range.data,
                range.bytes,
            }
        );
    }
    return total;
}

static void load_hot_slices_into_backing(
        manual_model_source & source,
        ggml_tensor * tensor,
        uint8_t * dst) {
    const std::string name = ggml_get_name(tensor);
    auto it = source.tensors.find(name);
    if (it == source.tensors.end()) {
        throw std::runtime_error("tensor not found in manual source: " + name);
    }
    const tensor_source & info = it->second;
    const size_t n_bytes = ggml_nbytes(tensor);
    if (!info.is_lazy_expert) {
        std::memset(dst, 0, n_bytes);
        read_bytes(source.file, info.file_offset, dst, n_bytes);
        return;
    }
#if !defined(__APPLE__)
    std::memset(dst, 0, n_bytes);
#endif
    for (int expert_id : source.hot_experts) {
        if (expert_id < 0 || expert_id >= source.n_expert) {
            continue;
        }
        read_bytes(
            source.file,
            info.file_offset + (uint64_t) info.slice_bytes * (uint64_t) expert_id,
            dst + (size_t) expert_id * info.slice_bytes,
            info.slice_bytes);
    }
}

static void maybe_rebind_lazy_tensor(
        ggml_tensor * tensor,
        manual_model_source & source,
        std::vector<owned_tensor_backing> & owned,
        std::set<ggml_tensor *> & seen) {
    if (!tensor || !seen.insert(tensor).second) {
        return;
    }
    const std::string name = ggml_get_name(tensor);
    auto it = source.tensors.find(name);
    if (it == source.tensors.end() || !it->second.is_lazy_expert) {
        return;
    }

    const size_t n_bytes = ggml_nbytes(tensor);
    size_t alloc_size = 0;
    uint8_t * data = allocate_host_backing(n_bytes, &alloc_size);
    load_hot_slices_into_backing(source, tensor, data);
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, n_bytes);
    if (!buffer) {
        free_host_backing(data, alloc_size);
        throw std::runtime_error("failed to create CPU_Mapped buffer from custom expert backing");
    }

    tensor->buffer = buffer;
    tensor->data = data;
    owned.push_back(
        owned_tensor_backing{
            tensor,
            buffer,
            data,
            n_bytes,
            alloc_size,
        }
    );
}

static void rebind_lazy_expert_tensors_to_custom_backing(
        llama_model * model,
        manual_model_source & source,
        std::vector<owned_tensor_backing> & owned) {
    std::set<ggml_tensor *> seen;
    for (auto & layer : model->layers) {
        maybe_rebind_lazy_tensor(layer.ffn_gate_exps, source, owned, seen);
        maybe_rebind_lazy_tensor(layer.ffn_down_exps, source, owned, seen);
        maybe_rebind_lazy_tensor(layer.ffn_up_exps, source, owned, seen);
        maybe_rebind_lazy_tensor(layer.ffn_gate_up_exps, source, owned, seen);
        maybe_rebind_lazy_tensor(layer.ffn_exp_probs_b, source, owned, seen);
    }
}

static void free_owned_backings(std::vector<owned_tensor_backing> & owned_backings) {
    for (auto & backing : owned_backings) {
        if (backing.buffer) {
            ggml_backend_buffer_free(backing.buffer);
        }
        free_host_backing(backing.data, backing.alloc_size);
    }
    owned_backings.clear();
}

static void madvise_expert(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int expert_id,
        int advice);

static void madvise_expert(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        int expert_id,
        int advice) {
    auto it = ranges.find(expert_id);
    if (it == ranges.end()) {
        return;
    }
    for (const auto & range : it->second) {
        if (hard_evict && advice == MADV_DONTNEED) {
            hard_evict_attempt_count++;
            if (hard_evict_region(range.data, range.bytes)) {
                hard_evict_success_count++;
                continue;
            }
        }
        madvise_region(range.data, range.bytes, advice);
    }
}

static void apply_hot_cold_state(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        const std::set<int> & hot_experts,
        int32_t n_expert) {
    for (int expert_id = 0; expert_id < n_expert; ++expert_id) {
        auto it = ranges.find(expert_id);
        if (it == ranges.end()) {
            continue;
        }
        if (hot_experts.count(expert_id) > 0) {
            for (const auto & range : it->second) {
#if defined(__APPLE__) && defined(MADV_FREE_REUSE)
                madvise_region(range.data, range.bytes, MADV_FREE_REUSE);
#endif
                madvise_region(range.data, range.bytes, MADV_WILLNEED);
                touch_region(range.data, range.bytes);
            }
        } else {
            madvise_expert(ranges, expert_id, MADV_DONTNEED);
        }
    }
}

static void update_cold_cache_state(
        const std::unordered_map<int, std::vector<slice_range>> & ranges,
        const std::set<int> & hot_experts,
        const std::vector<int> & accessed_cold_experts,
        std::list<int> & resident_order,
        std::set<int> & resident_cold) {
    for (int expert_id : accessed_cold_experts) {
        resident_order.remove(expert_id);
        resident_order.push_front(expert_id);
        resident_cold.insert(expert_id);
    }

    while ((int) resident_order.size() > cold_cache_expert_count) {
        int evicted = resident_order.back();
        resident_order.pop_back();
        resident_cold.erase(evicted);
        if (hot_experts.count(evicted) == 0) {
            madvise_expert(ranges, evicted, MADV_DONTNEED);
        }
    }

    if (cold_cache_expert_count == 0) {
        for (int expert_id : accessed_cold_experts) {
            if (hot_experts.count(expert_id) == 0) {
                madvise_expert(ranges, expert_id, MADV_DONTNEED);
            }
        }
        resident_order.clear();
        resident_cold.clear();
    }
}

static long current_minor_faults() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_minflt;
#else
    return 0;
#endif
}

static long current_major_faults() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_majflt;
#else
    return 0;
#endif
}

static void write_json_string(std::ofstream & out, const std::string & value) {
    out << '"';
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if ((unsigned char) c < 0x20) {
                    out << '?';
                } else {
                    out << c;
                }
        }
    }
    out << '"';
}

static void write_result_json(
        const std::string & path,
        const std::vector<int> & hot_experts,
        const experiment_stats & stats) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open json output: " + path);
    }

    out << "{\n";
    out << "  \"hot_experts\": [";
    for (size_t i = 0; i < hot_experts.size(); ++i) {
        if (i > 0) out << ", ";
        out << hot_experts[i];
    }
    out << "],\n";
    out << "  \"hot_expert_count\": " << stats.hot_expert_count << ",\n";
    out << "  \"cold_cache_expert_count\": " << cold_cache_expert_count << ",\n";
    out << "  \"prompt_tokens\": " << stats.prompt_tokens << ",\n";
    out << "  \"generated_tokens\": " << stats.generated_tokens << ",\n";
    out << "  \"decode_only_measurement\": " << (decode_only_measurement ? "true" : "false") << ",\n";
    out << "  \"hard_evict\": " << (hard_evict ? "true" : "false") << ",\n";
    out << "  \"custom_expert_backing\": " << (custom_expert_backing ? "true" : "false") << ",\n";
    out << "  \"hard_evict_attempt_count\": " << hard_evict_attempt_count << ",\n";
    out << "  \"hard_evict_success_count\": " << hard_evict_success_count << ",\n";
    out << "  \"expert_buffer_bytes\": " << stats.expert_buffer_bytes << ",\n";
    out << "  \"initial_resident_expert_bytes\": " << stats.initial_resident_expert_bytes << ",\n";
    out << "  \"final_resident_expert_bytes\": " << stats.final_resident_expert_bytes << ",\n";
    out << "  \"min_resident_expert_bytes\": " << stats.min_resident_expert_bytes << ",\n";
    out << "  \"max_resident_expert_bytes\": " << stats.max_resident_expert_bytes << ",\n";
    out << "  \"cold_activation_count\": " << stats.cold_activation_count << ",\n";
    out << "  \"unique_cold_experts\": " << stats.unique_cold_experts << ",\n";
    out << "  \"estimated_fetch_bytes\": " << stats.estimated_fetch_bytes << ",\n";
    out << "  \"fetch_event_count\": " << stats.fetch_event_count << ",\n";
    out << "  \"fetch_stall_upper_bound_s\": " << stats.fetch_stall_upper_bound_s << ",\n";
    out << "  \"prefill_cold_activation_count\": " << stats.prefill_cold_activation_count << ",\n";
    out << "  \"decode_cold_activation_count\": " << stats.decode_cold_activation_count << ",\n";
    out << "  \"prefill_unique_cold_experts\": " << stats.prefill_unique_cold_experts << ",\n";
    out << "  \"decode_unique_cold_experts\": " << stats.decode_unique_cold_experts << ",\n";
    out << "  \"prefill_fetch_bytes\": " << stats.prefill_fetch_bytes << ",\n";
    out << "  \"decode_fetch_bytes\": " << stats.decode_fetch_bytes << ",\n";
    out << "  \"prefill_fetch_event_count\": " << stats.prefill_fetch_event_count << ",\n";
    out << "  \"decode_fetch_event_count\": " << stats.decode_fetch_event_count << ",\n";
    out << "  \"prefill_fetch_stall_upper_bound_s\": " << stats.prefill_fetch_stall_upper_bound_s << ",\n";
    out << "  \"decode_fetch_stall_upper_bound_s\": " << stats.decode_fetch_stall_upper_bound_s << ",\n";
    out << "  \"post_prefill_probe_expert_ids\": [";
    for (size_t i = 0; i < stats.post_prefill_probe_expert_ids.size(); ++i) {
        if (i) out << ", ";
        out << stats.post_prefill_probe_expert_ids[i];
    }
    out << "],\n";
    out << "  \"post_prefill_probe_resident_bytes\": [";
    for (size_t i = 0; i < stats.post_prefill_probe_resident_bytes.size(); ++i) {
        if (i) out << ", ";
        out << stats.post_prefill_probe_resident_bytes[i];
    }
    out << "],\n";
    out << "  \"minor_faults_delta\": " << stats.minor_faults_delta << ",\n";
    out << "  \"major_faults_delta\": " << stats.major_faults_delta << ",\n";
    out << "  \"output_text\": ";
    write_json_string(out, stats.output_text);
    out << ",\n";
    out << "  \"steps\": [\n";
    for (size_t i = 0; i < stats.steps.size(); ++i) {
        const auto & step = stats.steps[i];
        out << "    {\n";
        out << "      \"phase\": ";
        write_json_string(out, step.phase);
        out << ",\n";
        out << "      \"token_index\": " << step.token_index << ",\n";
        out << "      \"elapsed_s\": " << step.elapsed_s << ",\n";
        out << "      \"selected_experts\": [";
        for (size_t j = 0; j < step.selected_experts.size(); ++j) {
            if (j > 0) out << ", ";
            out << step.selected_experts[j];
        }
        out << "],\n";
        out << "      \"newly_fetched_experts\": [";
        for (size_t j = 0; j < step.newly_fetched_experts.size(); ++j) {
            if (j > 0) out << ", ";
            out << step.newly_fetched_experts[j];
        }
        out << "],\n";
        out << "      \"resident_expert_bytes_before_eviction\": " << step.resident_expert_bytes_before_eviction << ",\n";
        out << "      \"resident_expert_bytes_after_eviction\": " << step.resident_expert_bytes_after_eviction << "\n";
        out << "    }";
        if (i + 1 < stats.steps.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 64;

    std::vector<const char *> filtered_argv;
    filtered_argv.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ranking-file") == 0 && i + 1 < argc) {
            ranking_file_path = argv[++i];
        } else if (std::strcmp(argv[i], "--hot-experts") == 0 && i + 1 < argc) {
            hot_expert_count = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cold-cache-experts") == 0 && i + 1 < argc) {
            cold_cache_expert_count = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json-out") == 0 && i + 1 < argc) {
            json_out_path = argv[++i];
        } else if (std::strcmp(argv[i], "--random-hot") == 0) {
            random_hot = true;
        } else if (std::strcmp(argv[i], "--hot-seed") == 0 && i + 1 < argc) {
            hot_seed = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--decode-only-measurement") == 0) {
            decode_only_measurement = true;
        } else if (std::strcmp(argv[i], "--hard-evict") == 0) {
            hard_evict = true;
        } else if (std::strcmp(argv[i], "--custom-expert-backing") == 0) {
            custom_expert_backing = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    if (!common_params_parse((int) filtered_argv.size(), const_cast<char **>(filtered_argv.data()), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (ranking_file_path.empty()) {
        LOG_ERR("--ranking-file is required\n");
        return 1;
    }
    if (hot_expert_count <= 0) {
        LOG_ERR("--hot-experts is required\n");
        return 1;
    }
    if (cold_cache_expert_count < 0) {
        LOG_ERR("--cold-cache-experts must be >= 0\n");
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("prompt is required (use --prompt or --file)\n");
        return 1;
    }

    auto ranking = load_ranking(ranking_file_path);
    auto hot_experts_vec = choose_hot_experts(ranking);
    std::set<int> hot_experts(hot_experts_vec.begin(), hot_experts_vec.end());
    hard_evict_attempt_count = 0;
    hard_evict_success_count = 0;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data;
    params.cb_eval = moe_callback;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    auto model_params = common_model_params_to_llama(params);
    auto ctx_params = common_context_params_to_llama(params);

    manual_model_source source = build_manual_model_source(params.model.path, hot_experts_vec);
    llama_model * model = llama_model_init_from_user(source.metadata, set_tensor_from_source, &source, model_params);
    if (!model) {
        LOG_ERR("failed to load model through manual tensor source\n");
        return 1;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("failed to initialize context from model\n");
        llama_model_free(model);
        return 1;
    }

    if (model->hparams.n_expert == 0) {
        LOG_ERR("model is not MoE\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (params.tensor_buft_overrides.empty()) {
        LOG_WRN("running without explicit CPU MoE tensor overrides; expert pages may not be host-resident for eviction\n");
    }

    const int32_t n_expert = (int32_t) model->hparams.n_expert;
    const int top_k = (int) model->hparams.n_expert_used;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    auto ranges = collect_expert_ranges(model);
    if (ranges.empty()) {
        LOG_ERR("no host-resident expert tensor ranges found; try --cpu-moe\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    auto expert_regions = collect_expert_tensor_regions(model);
    if (expert_regions.empty()) {
        LOG_ERR("no host-resident expert tensor regions found; try --cpu-moe\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::vector<owned_tensor_backing> owned_backings;
    if (custom_expert_backing) {
        auto old_ranges = ranges;
        rebind_lazy_expert_tensors_to_custom_backing(model, source, owned_backings);
        ranges = collect_expert_ranges(model);
        expert_regions = collect_expert_tensor_regions(model);
        for (int expert_id = 0; expert_id < n_expert; ++expert_id) {
            madvise_expert(old_ranges, expert_id, MADV_DONTNEED);
        }
    }

    cb_data.top_k = top_k;
    cb_data.n_expert = n_expert;
    cb_data.hot_experts = &hot_experts;
    cb_data.ranges = &ranges;
    cb_data.source = &source;

    apply_hot_cold_state(ranges, hot_experts, n_expert);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos);

    experiment_stats stats;
    stats.hot_expert_count = hot_expert_count;
    stats.prompt_tokens = (int) prompt_tokens.size();
    stats.expert_buffer_bytes = total_expert_buffer_bytes(ranges, n_expert);
    stats.initial_resident_expert_bytes = resident_bytes_for_expert_regions(expert_regions);
    stats.min_resident_expert_bytes = stats.initial_resident_expert_bytes;
    stats.max_resident_expert_bytes = stats.initial_resident_expert_bytes;

    std::set<int> ever_fetched_cold;
    std::set<int> prefill_fetched_cold;
    std::set<int> decode_fetched_cold;
    std::set<int> resident_cold;
    std::list<int> resident_order;
    cb_data.resident_cold = &resident_cold;

    auto update_step = [&](const std::string & phase, int token_index, double elapsed_s) {
        step_metrics step;
        step.phase = phase;
        step.token_index = token_index;
        step.elapsed_s = elapsed_s;
        step.selected_experts.assign(cb_data.step_selected.begin(), cb_data.step_selected.end());
        step.resident_expert_bytes_before_eviction = resident_bytes_for_expert_regions(expert_regions);

        std::vector<int> accessed_cold;
        for (int expert_id : cb_data.step_selected) {
            if (hot_experts.count(expert_id) == 0) {
                stats.cold_activation_count++;
                if (phase == "prefill") {
                    stats.prefill_cold_activation_count++;
                } else {
                    stats.decode_cold_activation_count++;
                }
                accessed_cold.push_back(expert_id);
            }
        }

        for (int expert_id : cb_data.step_newly_fetched) {
            if (hot_experts.count(expert_id) == 0) {
                uint64_t fetch_bytes = bytes_for_expert(ranges, expert_id);
                stats.estimated_fetch_bytes += fetch_bytes;
                if (phase == "prefill") {
                    stats.prefill_fetch_bytes += fetch_bytes;
                    prefill_fetched_cold.insert(expert_id);
                } else {
                    stats.decode_fetch_bytes += fetch_bytes;
                    decode_fetched_cold.insert(expert_id);
                }
                ever_fetched_cold.insert(expert_id);
            }
        }

        if (!cb_data.step_newly_fetched.empty()) {
            stats.fetch_event_count++;
            stats.fetch_stall_upper_bound_s += elapsed_s;
            if (phase == "prefill") {
                stats.prefill_fetch_event_count++;
                stats.prefill_fetch_stall_upper_bound_s += elapsed_s;
            } else {
                stats.decode_fetch_event_count++;
                stats.decode_fetch_stall_upper_bound_s += elapsed_s;
            }
            step.newly_fetched_experts.assign(cb_data.step_newly_fetched.begin(), cb_data.step_newly_fetched.end());
        }

        update_cold_cache_state(ranges, hot_experts, accessed_cold, resident_order, resident_cold);
        step.resident_expert_bytes_after_eviction = resident_bytes_for_expert_regions(expert_regions);
        stats.min_resident_expert_bytes = std::min(stats.min_resident_expert_bytes, step.resident_expert_bytes_after_eviction);
        stats.max_resident_expert_bytes = std::max(stats.max_resident_expert_bytes, step.resident_expert_bytes_before_eviction);
        stats.max_resident_expert_bytes = std::max(stats.max_resident_expert_bytes, step.resident_expert_bytes_after_eviction);
        stats.steps.push_back(std::move(step));
    };

    long minor_faults_before = current_minor_faults();
    long major_faults_before = current_major_faults();

    llama_memory_clear(llama_get_memory(ctx), true);

    cb_data.step_selected.clear();
    cb_data.step_newly_fetched.clear();
    cb_data.collect = true;
    double t0 = ggml_time_us() / 1e6;
    if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size()))) {
        LOG_ERR("prefill failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    double t1 = ggml_time_us() / 1e6;
    if (!decode_only_measurement) {
        update_step("prefill", -1, t1 - t0);
    } else {
        if (custom_expert_backing) {
            auto old_owned_backings = std::move(owned_backings);
            rebind_lazy_expert_tensors_to_custom_backing(model, source, owned_backings);
            free_owned_backings(old_owned_backings);
            ranges = collect_expert_ranges(model);
            expert_regions = collect_expert_tensor_regions(model);
            cb_data.ranges = &ranges;
        }
        apply_hot_cold_state(ranges, hot_experts, n_expert);
        resident_order.clear();
        resident_cold.clear();
        cb_data.step_selected.clear();
        cb_data.step_newly_fetched.clear();

        stats.generated_tokens = 0;
        stats.cold_activation_count = 0;
        stats.unique_cold_experts = 0;
        stats.estimated_fetch_bytes = 0;
        stats.fetch_event_count = 0;
        stats.fetch_stall_upper_bound_s = 0.0;
        stats.prefill_cold_activation_count = 0;
        stats.decode_cold_activation_count = 0;
        stats.prefill_unique_cold_experts = 0;
        stats.decode_unique_cold_experts = 0;
        stats.prefill_fetch_bytes = 0;
        stats.decode_fetch_bytes = 0;
        stats.prefill_fetch_event_count = 0;
        stats.decode_fetch_event_count = 0;
        stats.prefill_fetch_stall_upper_bound_s = 0.0;
        stats.decode_fetch_stall_upper_bound_s = 0.0;
        stats.steps.clear();
        ever_fetched_cold.clear();
        prefill_fetched_cold.clear();
        decode_fetched_cold.clear();
        minor_faults_before = current_minor_faults();
        major_faults_before = current_major_faults();
        stats.initial_resident_expert_bytes = resident_bytes_for_expert_regions(expert_regions);
        stats.min_resident_expert_bytes = stats.initial_resident_expert_bytes;
        stats.max_resident_expert_bytes = stats.initial_resident_expert_bytes;

        std::vector<int> probe_ids;
        for (int expert_id : hot_experts_vec) {
            if ((int) probe_ids.size() >= 2) {
                break;
            }
            probe_ids.push_back(expert_id);
        }
        for (const auto & item : ranking) {
            if ((int) probe_ids.size() >= 10) {
                break;
            }
            if (hot_experts.count(item.expert_id) == 0) {
                probe_ids.push_back(item.expert_id);
            }
        }
        stats.post_prefill_probe_expert_ids = probe_ids;
        for (int expert_id : probe_ids) {
            stats.post_prefill_probe_resident_bytes.push_back(resident_bytes_for_expert(ranges, expert_id));
        }
    }

    std::string output_text;

    for (int i = 0; i < params.n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token next = 0;
        float max_logit = -INFINITY;
        for (int token_id = 0; token_id < n_vocab; ++token_id) {
            if (logits[token_id] > max_logit) {
                max_logit = logits[token_id];
                next = token_id;
            }
        }

        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }

        output_text += common_token_to_piece(ctx, next, false);
        stats.generated_tokens++;

        cb_data.step_selected.clear();
        cb_data.step_newly_fetched.clear();
        double step_t0 = ggml_time_us() / 1e6;
        if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
            LOG_ERR("decode failed at token %d\n", i);
            break;
        }
        double step_t1 = ggml_time_us() / 1e6;
        update_step("decode", i, step_t1 - step_t0);
    }

    long minor_faults_after = current_minor_faults();
    long major_faults_after = current_major_faults();
    stats.minor_faults_delta = minor_faults_after - minor_faults_before;
    stats.major_faults_delta = major_faults_after - major_faults_before;
    stats.unique_cold_experts = (int) ever_fetched_cold.size();
    stats.prefill_unique_cold_experts = (int) prefill_fetched_cold.size();
    stats.decode_unique_cold_experts = (int) decode_fetched_cold.size();
    stats.final_resident_expert_bytes = resident_bytes_for_expert_regions(expert_regions);
    stats.min_resident_expert_bytes = std::min(stats.min_resident_expert_bytes, stats.final_resident_expert_bytes);
    stats.max_resident_expert_bytes = std::max(stats.max_resident_expert_bytes, stats.final_resident_expert_bytes);
    stats.output_text = output_text;

    if (!json_out_path.empty()) {
        write_result_json(json_out_path, hot_experts_vec, stats);
        // This experiment binary aggressively swaps custom tensor backing mid-run.
        // Exiting immediately after writing results avoids teardown and pipe-write
        // hangs that would otherwise leave the Python harness waiting even though
        // the measurement data is already complete.
        std::_Exit(0);
    }

    LOG_INF("hot experts: %d\n", hot_expert_count);
    LOG_INF("cold cache experts: %d\n", cold_cache_expert_count);
    LOG_INF("generated tokens: %d\n", stats.generated_tokens);
    LOG_INF("unique cold experts: %d\n", stats.unique_cold_experts);
    LOG_INF("estimated fetch bytes: %llu\n", (unsigned long long) stats.estimated_fetch_bytes);
    LOG_INF("fetch events: %d\n", stats.fetch_event_count);
    LOG_INF("fetch stall upper bound: %.3fs\n", stats.fetch_stall_upper_bound_s);
    LOG_INF("minor faults delta: %ld\n", stats.minor_faults_delta);
    LOG_INF("major faults delta: %ld\n", stats.major_faults_delta);

    llama_free(ctx);
    llama_model_free(model);
    free_owned_backings(owned_backings);
    gguf_free(source.metadata);
    ggml_free(source.meta_ctx);
    llama_backend_free();
    return 0;
}
