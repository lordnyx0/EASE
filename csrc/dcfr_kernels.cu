#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

// ============================================================================
// D-CFR HIGH PERFORMANCE C++/CUDA ENGINE FOR QWEN 3.8 / 3.5 27B
// ============================================================================

// ----------------------------------------------------------------------------
// 1. FAST VOCABULARY ARGMAX REDUCTION KERNEL
// ----------------------------------------------------------------------------
template<typename T, int BLOCK_SIZE>
__global__ void fast_argmax_kernel(
    const T* __restrict__ logits, // [B, V]
    int64_t* __restrict__ out_idx,   // [B]
    const int vocab_size
) {
    int b = blockIdx.x;
    const T* row = logits + b * vocab_size;
    int tid = threadIdx.x;

    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val;
        if constexpr (std::is_same<T, half>::value) {
            val = __half2float(row[i]);
        } else {
            val = (float)row[i];
        }
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    __shared__ float s_val[BLOCK_SIZE];
    __shared__ int   s_idx[BLOCK_SIZE];

    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (s_val[tid + offset] > s_val[tid]) {
                s_val[tid] = s_val[tid + offset];
                s_idx[tid] = s_idx[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[b] = s_idx[0];
    }
}


// ----------------------------------------------------------------------------
// 2. ULTRA-FAST BATCHED MEMORY COPY KERNEL (EASE SNAPSHOT & RESTORE)
// Copies N tensors concurrently in a single grid launch with perfect float alignment
// ----------------------------------------------------------------------------
__global__ void batched_copy_ptrs_kernel(
    uintptr_t* __restrict__ dst_ptrs,
    const uintptr_t* __restrict__ src_ptrs,
    const size_t* __restrict__ byte_sizes,
    int num_tensors
) {
    int tensor_idx = blockIdx.y;
    if (tensor_idx >= num_tensors) return;

    const float* src = (const float*)src_ptrs[tensor_idx];
    float* dst = (float*)dst_ptrs[tensor_idx];
    size_t num_floats = byte_sizes[tensor_idx] / sizeof(float);

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_floats; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}


// ----------------------------------------------------------------------------
// 3. FUSED GDN FACTOR REPLAY KERNEL (32 heads, 128x128 dim)
// S_t = exp(g_t) * S_{t-1} + delta_t * k_t^T (in 0.027 ms)
// ----------------------------------------------------------------------------
__global__ void gdn_factor_replay_kernel(
    float* __restrict__ state,             // [num_heads, v_dim, k_dim]
    const float* __restrict__ gate,        // [num_heads]
    const float* __restrict__ delta,       // [num_heads, v_dim]
    const float* __restrict__ key,         // [num_heads, k_dim]
    const int num_heads,
    const int v_dim,
    const int k_dim
) {
    int h = blockIdx.z;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (h >= num_heads || v >= v_dim || k >= k_dim) return;

    float g_exp = __expf(gate[h]);
    float d_val = delta[h * v_dim + v];
    float k_val = key[h * k_dim + k];

    size_t idx = ((size_t)h * v_dim + v) * k_dim + k;
    float s_prev = state[idx];
    state[idx] = s_prev * g_exp + d_val * k_val;
}

// ----------------------------------------------------------------------------
// 4. FUSED BATCHED FACTOR REPLAY FOR ALL 48 LAYERS IN A SINGLE GRID LAUNCH
// ----------------------------------------------------------------------------
__global__ void gdn_batched_48_layers_replay_kernel(
    float* __restrict__ all_states,        // [48, num_heads, v_dim, k_dim]
    const float* __restrict__ all_gates,   // [48, num_heads]
    const float* __restrict__ all_deltas,  // [48, num_heads, v_dim]
    const float* __restrict__ all_keys,    // [48, num_heads, k_dim]
    const int num_layers,
    const int num_heads,
    const int v_dim,
    const int k_dim
) {
    int layer = blockIdx.z / num_heads;
    int h = blockIdx.z % num_heads;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (layer >= num_layers || h >= num_heads || v >= v_dim || k >= k_dim) return;

    float g_exp = __expf(all_gates[layer * num_heads + h]);
    float d_val = all_deltas[(layer * num_heads + h) * v_dim + v];
    float k_val = all_keys[(layer * num_heads + h) * k_dim + k];

    size_t layer_stride = (size_t)num_heads * v_dim * k_dim;
    size_t idx = (size_t)layer * layer_stride + ((size_t)h * v_dim + v) * k_dim + k;

    float s_prev = all_states[idx];
    all_states[idx] = s_prev * g_exp + d_val * k_val;
}

// ----------------------------------------------------------------------------
// 5. SPECULATIVE TREE VERIFICATION AND ACCEPTANCE RESOLVER (IN-KERNEL)
// ----------------------------------------------------------------------------
__global__ void evaluate_speculative_acceptance_kernel(
    const int64_t* __restrict__ draft_tokens,    // [K]
    const int64_t* __restrict__ target_tokens,   // [K + 1]
    int* __restrict__ out_accepted_count,        // [1]
    int64_t* __restrict__ out_accepted_tokens,   // [K + 1]
    const int K
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int accepted = 0;
        for (int i = 0; i < K; ++i) {
            if (target_tokens[i] == draft_tokens[i]) {
                out_accepted_tokens[accepted] = draft_tokens[i];
                accepted++;
            } else {
                out_accepted_tokens[accepted] = target_tokens[i];
                accepted++;
                break;
            }
        }
        if (accepted == K) {
            out_accepted_tokens[accepted] = target_tokens[K];
            accepted++;
        }
        *out_accepted_count = accepted;
    }
}

// ----------------------------------------------------------------------------
// 6. SSM CONVOLUTION TREE KERNEL
// ----------------------------------------------------------------------------
__global__ void ssm_conv_tree_kernel(
    const float* __restrict__ x,              // [nodes, channels]
    const float* __restrict__ weights,        // [channels, 4]
    const float* __restrict__ conv_state,     // [channels, 3]
    const float* __restrict__ replay_factors, // [factor_slots, channels]
    const int32_t* __restrict__ parent,       // [nodes]
    const int32_t* __restrict__ replay_path,  // [replay_len + 1]
    float* __restrict__ dst,                  // [nodes, channels]
    const int channels,
    const int nodes,
    const int factor_slots
) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;

    float base[3] = {
        conv_state[channel * 3 + 0],
        conv_state[channel * 3 + 1],
        conv_state[channel * 3 + 2]
    };

    int replay_len = replay_path ? replay_path[0] : 0;
    for (int i = 0; i < replay_len; ++i) {
        int node = replay_path[i + 1];
        base[0] = base[1];
        base[1] = base[2];
        base[2] = replay_factors[(int64_t)node * channels + channel];
    }

    const float* w = weights + channel * 4;

    for (int node = 0; node < nodes; ++node) {
        int ancestors[3];
        int count = 0;
        for (int current = parent[node]; current >= 0 && count < 3; current = parent[current]) {
            ancestors[count++] = current;
        }

        float window[3] = {base[0], base[1], base[2]};
        for (int i = count - 1; i >= 0; --i) {
            window[0] = window[1];
            window[1] = window[2];
            window[2] = x[(int64_t)ancestors[i] * channels + channel];
        }

        float current = x[(int64_t)node * channels + channel];
        dst[(int64_t)node * channels + channel] =
            window[0] * w[0] + window[1] * w[1] + window[2] * w[2] + current * w[3];
    }
}


torch::Tensor fast_argmax_cuda(torch::Tensor logits) {
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    int bsz = logits.size(0);
    int vocab_size = logits.size(1);
    auto out = torch::empty({bsz}, torch::dtype(torch::kInt64).device(logits.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    dim3 blocks(bsz);
    dim3 threads(256);
    if (logits.scalar_type() == torch::kFloat32) {
        fast_argmax_kernel<float, 256><<<blocks, threads, 0, stream>>>(
            (const float*)logits.data_ptr(),
            (int64_t*)out.data_ptr(),
            vocab_size
        );
    } else if (logits.scalar_type() == torch::kFloat16) {
        fast_argmax_kernel<half, 256><<<blocks, threads, 0, stream>>>(
            (const half*)logits.data_ptr(),
            (int64_t*)out.data_ptr(),
            vocab_size
        );
    } else {
        TORCH_CHECK(false, "fast_argmax_cuda supports only Float32 and Float16");
    }
    return out;
}


void ease_snapshot_native(
    const std::vector<torch::Tensor>& rec_src,
    const std::vector<torch::Tensor>& conv_src,
    std::vector<torch::Tensor>& rec_dst,
    std::vector<torch::Tensor>& conv_dst
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t n = rec_src.size();
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpyAsync(rec_dst[i].data_ptr(), rec_src[i].data_ptr(), rec_src[i].nbytes(), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(conv_dst[i].data_ptr(), conv_src[i].data_ptr(), conv_src[i].nbytes(), cudaMemcpyDeviceToDevice, stream);
    }
}

void ease_restore_native(
    const std::vector<torch::Tensor>& rec_src,
    const std::vector<torch::Tensor>& conv_src,
    std::vector<torch::Tensor>& rec_dst,
    std::vector<torch::Tensor>& conv_dst
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t n = rec_src.size();
    for (size_t i = 0; i < n; ++i) {
        cudaMemcpyAsync(rec_dst[i].data_ptr(), rec_src[i].data_ptr(), rec_src[i].nbytes(), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(conv_dst[i].data_ptr(), conv_src[i].data_ptr(), conv_src[i].nbytes(), cudaMemcpyDeviceToDevice, stream);
    }
}

__global__ void ease_fused_copy_attention_pages_kernel(
    const uintptr_t* __restrict__ base_ptrs,
    const int* __restrict__ bytes_per_token,
    const int* __restrict__ page_bytes,
    int from_page,
    int to_page,
    int num_tokens,
    int num_entries
) {
    int entry = blockIdx.x;
    if (entry >= num_entries) return;
    
    uintptr_t base = base_ptrs[entry];
    int p_bytes = page_bytes[entry];
    int tok_bytes = bytes_per_token[entry];
    int total_bytes = tok_bytes * num_tokens;
    
    const uint4* src = (const uint4*)(base + (size_t)from_page * p_bytes);
    uint4* dst = (uint4*)(base + (size_t)to_page * p_bytes);
    int num_uint4 = total_bytes >> 4;
    
    for (int idx = threadIdx.x; idx < num_uint4; idx += blockDim.x) {
        dst[idx] = src[idx];
    }
}

void ease_fused_copy_attention_pages_cuda(
    torch::Tensor base_ptrs,
    torch::Tensor bytes_per_token,
    torch::Tensor page_bytes,
    int64_t from_page,
    int64_t to_page,
    int64_t num_tokens
) {
    if (num_tokens <= 0) return;
    const at::cuda::OptionalCUDAGuard device_guard(base_ptrs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    int num_entries = base_ptrs.size(0);
    dim3 grid(num_entries);
    dim3 block(128);
    
    ease_fused_copy_attention_pages_kernel<<<grid, block, 0, stream>>>(
        (const uintptr_t*)base_ptrs.data_ptr(),
        (const int*)bytes_per_token.data_ptr(),
        (const int*)page_bytes.data_ptr(),
        (int)from_page,
        (int)to_page,
        (int)num_tokens,
        num_entries
    );
}

void ease_copy_attention_pages_native(
    const std::vector<torch::Tensor>& qk_list,
    const std::vector<torch::Tensor>& qv_list,
    const std::vector<torch::Tensor>& sk_list,
    const std::vector<torch::Tensor>& sv_list,
    int64_t from_page,
    int64_t to_page,
    int64_t num_tokens
) {
    if (num_tokens <= 0) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    size_t n = qk_list.size();
    for (size_t i = 0; i < n; ++i) {
        const auto& qk = qk_list[i];
        const auto& qv = qv_list[i];
        const auto& sk = sk_list[i];
        const auto& sv = sv_list[i];
        
        int64_t page_tokens = qk.size(1);
        int64_t k_dim = qk.size(2);
        int64_t k_elem = qk.element_size();
        int64_t k_page_bytes = page_tokens * k_dim * k_elem;
        int64_t k_copy_bytes = num_tokens * k_dim * k_elem;
        char* qk_src = (char*)qk.data_ptr() + from_page * k_page_bytes;
        char* qk_dst = (char*)qk.data_ptr() + to_page * k_page_bytes;
        cudaMemcpyAsync(qk_dst, qk_src, k_copy_bytes, cudaMemcpyDeviceToDevice, stream);

        int64_t v_dim = qv.size(2);
        int64_t v_elem = qv.element_size();
        int64_t v_page_bytes = page_tokens * v_dim * v_elem;
        int64_t v_copy_bytes = num_tokens * v_dim * v_elem;
        char* qv_src = (char*)qv.data_ptr() + from_page * v_page_bytes;
        char* qv_dst = (char*)qv.data_ptr() + to_page * v_page_bytes;
        cudaMemcpyAsync(qv_dst, qv_src, v_copy_bytes, cudaMemcpyDeviceToDevice, stream);

        int64_t sk_dim = sk.size(2);
        int64_t sk_elem = sk.element_size();
        int64_t sk_page_bytes = page_tokens * sk_dim * sk_elem;
        int64_t sk_copy_bytes = num_tokens * sk_dim * sk_elem;
        char* sk_src = (char*)sk.data_ptr() + from_page * sk_page_bytes;
        char* sk_dst = (char*)sk.data_ptr() + to_page * sk_page_bytes;
        cudaMemcpyAsync(sk_dst, sk_src, sk_copy_bytes, cudaMemcpyDeviceToDevice, stream);

        int64_t sv_dim = sv.size(2);
        int64_t sv_elem = sv.element_size();
        int64_t sv_page_bytes = page_tokens * sv_dim * sv_elem;
        int64_t sv_copy_bytes = num_tokens * sv_dim * sv_elem;
        char* sv_src = (char*)sv.data_ptr() + from_page * sv_page_bytes;
        char* sv_dst = (char*)sv.data_ptr() + to_page * sv_page_bytes;
        cudaMemcpyAsync(sv_dst, sv_src, sv_copy_bytes, cudaMemcpyDeviceToDevice, stream);
    }
}

struct EaseStepResult {
    int branch_winner; // 0: Branch A, 1: Branch B, 2: Fallback
    int num_accepted;
    std::vector<int64_t> committed_tokens;
};


std::tuple<int, int, std::vector<int64_t>> resolve_ease_b2_step_cuda(
    torch::Tensor v_logits,
    const std::vector<int64_t>& cand_a,
    const std::vector<int64_t>& cand_b,
    bool is_b2
) {
    const at::cuda::OptionalCUDAGuard device_guard(v_logits.device());
    int bsz = v_logits.size(0);
    int seq_len = v_logits.size(1);
    int vocab_size = v_logits.size(2);

    auto flat_logits = v_logits.view({bsz * seq_len, vocab_size});
    auto argmax_tokens = fast_argmax_cuda(flat_logits);
    
    auto cpu_tokens = argmax_tokens.to(torch::kCPU);
    const int64_t* tokens_ptr = (const int64_t*)cpu_tokens.data_ptr();

    int branch_winner = 0;
    int num_accepted = 1;
    std::vector<int64_t> committed_tokens;

    if (!is_b2 || bsz == 1) {
        int max_eval = std::min((int)cand_a.size(), seq_len);
        for (int k = 0; k < max_eval; ++k) {
            int64_t pred_k = tokens_ptr[k];
            if (pred_k == cand_a[k]) {
                committed_tokens.push_back(cand_a[k]);
            } else {
                committed_tokens.push_back(pred_k);
                break;
            }
        }
        if (committed_tokens.empty()) {
            committed_tokens.push_back(tokens_ptr[0]);
        } else if (committed_tokens.size() == cand_a.size() && (int)committed_tokens.size() < seq_len) {
            committed_tokens.push_back(tokens_ptr[committed_tokens.size()]);
        }
        num_accepted = (int)committed_tokens.size();
        branch_winner = (num_accepted > 1) ? 0 : 2;
    } else {
        int64_t pred_A0 = tokens_ptr[0];
        int64_t r1_tok = cand_a.empty() ? -1 : cand_a[0];
        int64_t r2_tok = cand_b.empty() ? -1 : cand_b[0];

        if (pred_A0 == r1_tok) {
            branch_winner = 0;
            committed_tokens.push_back(r1_tok);
            int max_eval_a = std::min((int)cand_a.size(), seq_len);
            for (int k = 1; k < max_eval_a; ++k) {
                int64_t pred_Ak = tokens_ptr[k];
                if (pred_Ak == cand_a[k]) {
                    committed_tokens.push_back(cand_a[k]);
                } else {
                    committed_tokens.push_back(pred_Ak);
                    break;
                }
            }
            if (committed_tokens.size() == cand_a.size() && (int)committed_tokens.size() < seq_len) {
                committed_tokens.push_back(tokens_ptr[committed_tokens.size()]);
            }
            num_accepted = (int)committed_tokens.size();
        } else if (pred_A0 == r2_tok) {
            branch_winner = 1;
            committed_tokens.push_back(r2_tok);
            int max_eval_b = std::min((int)cand_b.size(), seq_len);
            for (int k = 1; k < max_eval_b; ++k) {
                int64_t pred_Bk = tokens_ptr[seq_len + k];
                if (pred_Bk == cand_b[k]) {
                    committed_tokens.push_back(cand_b[k]);
                } else {
                    committed_tokens.push_back(pred_Bk);
                    break;
                }
            }
            if (committed_tokens.size() == cand_b.size() && (int)committed_tokens.size() < seq_len) {
                committed_tokens.push_back(tokens_ptr[seq_len + committed_tokens.size()]);
            }
            num_accepted = (int)committed_tokens.size();
        } else {
            branch_winner = 2;
            committed_tokens.push_back(pred_A0);
            num_accepted = 1;
        }
    }
    return {branch_winner, num_accepted, committed_tokens};
}


void ssm_conv_tree_cuda(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor conv_state,
    torch::Tensor replay_factors,
    torch::Tensor parent,
    torch::Tensor replay_path,
    torch::Tensor dst
) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    int channels = x.size(1);
    int nodes = x.size(0);
    int factor_slots = replay_factors.size(0);

    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    ssm_conv_tree_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)x.data_ptr(),
        (const float*)weights.data_ptr(),
        (const float*)conv_state.data_ptr(),
        (const float*)replay_factors.data_ptr(),
        (const int32_t*)parent.data_ptr(),
        replay_path.defined() ? (const int32_t*)replay_path.data_ptr() : nullptr,
        (float*)dst.data_ptr(),
        channels,
        nodes,
        factor_slots
    );
}

void gdn_factor_replay_cuda_op(
    torch::Tensor state,
    torch::Tensor gate,
    torch::Tensor delta,
    torch::Tensor key
) {
    const at::cuda::OptionalCUDAGuard device_guard(state.device());
    int num_heads = state.size(0);
    int v_dim = state.size(1);
    int k_dim = state.size(2);

    dim3 threads(16, 16);
    dim3 blocks(
        (k_dim + threads.x - 1) / threads.x,
        (v_dim + threads.y - 1) / threads.y,
        num_heads
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gdn_factor_replay_kernel<<<blocks, threads, 0, stream>>>(
        (float*)state.data_ptr(),
        (const float*)gate.data_ptr(),
        (const float*)delta.data_ptr(),
        (const float*)key.data_ptr(),
        num_heads,
        v_dim,
        k_dim
    );
}

void gdn_batched_48_layers_replay_cuda_op(
    torch::Tensor all_states,
    torch::Tensor all_gates,
    torch::Tensor all_deltas,
    torch::Tensor all_keys
) {
    const at::cuda::OptionalCUDAGuard device_guard(all_states.device());
    int num_layers = all_states.size(0);
    int num_heads = all_states.size(1);
    int v_dim = all_states.size(2);
    int k_dim = all_states.size(3);

    dim3 threads(16, 16);
    dim3 blocks(
        (k_dim + threads.x - 1) / threads.x,
        (v_dim + threads.y - 1) / threads.y,
        num_layers * num_heads
    );
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    gdn_batched_48_layers_replay_kernel<<<blocks, threads, 0, stream>>>(
        (float*)all_states.data_ptr(),
        (const float*)all_gates.data_ptr(),
        (const float*)all_deltas.data_ptr(),
        (const float*)all_keys.data_ptr(),
        num_layers,
        num_heads,
        v_dim,
        k_dim
    );
}

std::pair<torch::Tensor, int> evaluate_speculative_acceptance_cuda(
    torch::Tensor draft_tokens,
    torch::Tensor target_tokens
) {
    const at::cuda::OptionalCUDAGuard device_guard(draft_tokens.device());
    int K = draft_tokens.numel();
    auto out_tokens = torch::empty({K + 1}, torch::dtype(torch::kInt64).device(draft_tokens.device()));
    auto out_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(draft_tokens.device()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    evaluate_speculative_acceptance_kernel<<<1, 1, 0, stream>>>(
        (const int64_t*)draft_tokens.data_ptr(),
        (const int64_t*)target_tokens.data_ptr(),
        (int*)out_count.data_ptr(),
        (int64_t*)out_tokens.data_ptr(),
        K
    );

    int accepted_count = out_count.item<int>();
    return {out_tokens.slice(0, 0, accepted_count), accepted_count};
}

// ----------------------------------------------------------------------------
// 6. FAST TOP-2 PROBABILITIES AND TOKENS EXTRACTION (ZERO ALLOC / ZERO .item())
// ----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void fast_top2_probs_kernel(
    const scalar_t* __restrict__ logits,
    int vocab_size,
    float* __restrict__ out_p,
    int64_t* __restrict__ out_tok
) {
    __shared__ float s_max1[1024];
    __shared__ int64_t s_idx1[1024];
    __shared__ float s_max2[1024];
    __shared__ int64_t s_idx2[1024];
    __shared__ float s_sum[1024];

    int tid = threadIdx.x;
    float l_max1 = -1e20f;
    int64_t l_idx1 = -1;
    float l_max2 = -1e20f;
    int64_t l_idx2 = -1;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = static_cast<float>(logits[i]);
        if (val > l_max1) {
            l_max2 = l_max1;
            l_idx2 = l_idx1;
            l_max1 = val;
            l_idx1 = i;
        } else if (val > l_max2) {
            l_max2 = val;
            l_idx2 = i;
        }
    }

    s_max1[tid] = l_max1;
    s_idx1[tid] = l_idx1;
    s_max2[tid] = l_max2;
    s_idx2[tid] = l_idx2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v1_a = s_max1[tid];
            int64_t i1_a = s_idx1[tid];
            float v2_a = s_max2[tid];
            int64_t i2_a = s_idx2[tid];

            float v1_b = s_max1[tid + s];
            int64_t i1_b = s_idx1[tid + s];
            float v2_b = s_max2[tid + s];
            int64_t i2_b = s_idx2[tid + s];

            float best1, best2;
            int64_t bidx1, bidx2;
            if (v1_a >= v1_b) {
                best1 = v1_a; bidx1 = i1_a;
                if (v1_b >= v2_a) {
                    best2 = v1_b; bidx2 = i1_b;
                } else {
                    best2 = v2_a; bidx2 = i2_a;
                }
            } else {
                best1 = v1_b; bidx1 = i1_b;
                if (v1_a >= v2_b) {
                    best2 = v1_a; bidx2 = i1_a;
                } else {
                    best2 = v2_b; bidx2 = i2_b;
                }
            }

            s_max1[tid] = best1; s_idx1[tid] = bidx1;
            s_max2[tid] = best2; s_idx2[tid] = bidx2;
        }
        __syncthreads();
    }

    float g_max = s_max1[0];
    int64_t g_idx1 = s_idx1[0];
    float g_val2 = s_max2[0];
    int64_t g_idx2 = s_idx2[0];

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_sum += __expf(static_cast<float>(logits[i]) - g_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = s_sum[0];
        float inv_sum = 1.0f / fmaxf(total_sum, 1e-12f);
        out_p[0] = 1.0f * inv_sum;
        out_p[1] = __expf(g_val2 - g_max) * inv_sum;
        out_tok[0] = g_idx1;
        out_tok[1] = g_idx2;
    }
}

std::tuple<float, float, int64_t, int64_t> fast_top2_probs_cuda(torch::Tensor logits) {
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    int vocab_size = logits.size(-1);
    auto flat_logits = logits.contiguous().view({-1, vocab_size});
    const auto last_row = flat_logits.slice(0, flat_logits.size(0) - 1, flat_logits.size(0));

    static torch::Tensor dev_p = torch::zeros({2}, torch::dtype(torch::kFloat32).device(logits.device()));
    static torch::Tensor dev_tok = torch::zeros({2}, torch::dtype(torch::kInt64).device(logits.device()));
    static torch::Tensor host_p = torch::zeros({2}, torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true));
    static torch::Tensor host_tok = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (logits.scalar_type() == torch::kFloat32) {
        fast_top2_probs_kernel<float><<<1, 1024, 0, stream>>>(
            (const float*)last_row.data_ptr(),
            vocab_size,
            (float*)dev_p.data_ptr(),
            (int64_t*)dev_tok.data_ptr()
        );
    } else if (logits.scalar_type() == torch::kFloat16) {
        fast_top2_probs_kernel<c10::Half><<<1, 1024, 0, stream>>>(
            (const c10::Half*)last_row.data_ptr(),
            vocab_size,
            (float*)dev_p.data_ptr(),
            (int64_t*)dev_tok.data_ptr()
        );
    }

    cudaMemcpyAsync(host_p.data_ptr(), dev_p.data_ptr(), 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_tok.data_ptr(), dev_tok.data_ptr(), 2 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const float* p_ptr = (const float*)host_p.data_ptr();
    const int64_t* tok_ptr = (const int64_t*)host_tok.data_ptr();

    return std::make_tuple(p_ptr[0], p_ptr[1], tok_ptr[0], tok_ptr[1]);
}

// ----------------------------------------------------------------------------
// 7. FUSED MTP INPUT KERNEL (RMSNorm(emb) + RMSNorm(hidden) -> Concat in SRAM)
// ----------------------------------------------------------------------------
template <typename scalar_t, typename weight_t>
__global__ void fused_mtp_input_kernel(
    const scalar_t* __restrict__ hidden,          // [B, 5120]
    const scalar_t* __restrict__ emb,             // [B, 5120]
    const weight_t* __restrict__ weight_hidden,   // [5120] or nullptr
    const weight_t* __restrict__ weight_emb,      // [5120] or nullptr
    scalar_t* __restrict__ out_cat,               // [B, 10240] -> [norm_emb, norm_hidden]
    int dim,                                      // 5120
    float eps
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;

    const scalar_t* h_ptr = hidden + b * dim;
    const scalar_t* e_ptr = emb + b * dim;
    scalar_t* out_ptr = out_cat + b * (dim * 2);

    __shared__ float s_sum_h[1024];
    __shared__ float s_sum_e[1024];

    float local_h2 = 0.0f;
    float local_e2 = 0.0f;

    for (int i = tid; i < dim; i += blockDim.x) {
        float h_val = static_cast<float>(h_ptr[i]);
        float e_val = static_cast<float>(e_ptr[i]);
        local_h2 += h_val * h_val;
        local_e2 += e_val * e_val;
    }

    s_sum_h[tid] = local_h2;
    s_sum_e[tid] = local_e2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_h[tid] += s_sum_h[tid + s];
            s_sum_e[tid] += s_sum_e[tid + s];
        }
        __syncthreads();
    }

    __shared__ float inv_rms_h;
    __shared__ float inv_rms_e;

    if (tid == 0) {
        inv_rms_h = rsqrtf(s_sum_h[0] / static_cast<float>(dim) + eps);
        inv_rms_e = rsqrtf(s_sum_e[0] / static_cast<float>(dim) + eps);
    }
    __syncthreads();

    float r_h = inv_rms_h;
    float r_e = inv_rms_e;

    for (int i = tid; i < dim; i += blockDim.x) {
        float scale_e = (weight_emb != nullptr) ? (1.0f + static_cast<float>(weight_emb[i])) : 1.0f;
        float scale_h = (weight_hidden != nullptr) ? (1.0f + static_cast<float>(weight_hidden[i])) : 1.0f;

        float e_norm = static_cast<float>(e_ptr[i]) * r_e * scale_e;
        float h_norm = static_cast<float>(h_ptr[i]) * r_h * scale_h;

        out_ptr[i] = static_cast<scalar_t>(e_norm);
        out_ptr[dim + i] = static_cast<scalar_t>(h_norm);
    }
}

void fused_mtp_input_cuda(
    torch::Tensor hidden,
    torch::Tensor emb,
    torch::Tensor weight_hidden,
    torch::Tensor weight_emb,
    torch::Tensor out_cat,
    float eps
) {
    const at::cuda::OptionalCUDAGuard device_guard(hidden.device());
    int bsz = hidden.size(0);
    int dim = hidden.size(-1);

    dim3 blocks(bsz);
    dim3 threads(1024);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const void* wh_ptr = (weight_hidden.defined() && weight_hidden.numel() > 0) ? weight_hidden.data_ptr() : nullptr;
    const void* we_ptr = (weight_emb.defined() && weight_emb.numel() > 0) ? weight_emb.data_ptr() : nullptr;

    bool is_bfloat16_weights = weight_hidden.defined() && weight_hidden.scalar_type() == torch::kBFloat16;

    if (hidden.scalar_type() == torch::kFloat16) {
        if (is_bfloat16_weights) {
            fused_mtp_input_kernel<c10::Half, c10::BFloat16><<<blocks, threads, 0, stream>>>(
                (const c10::Half*)hidden.data_ptr(),
                (const c10::Half*)emb.data_ptr(),
                (const c10::BFloat16*)wh_ptr,
                (const c10::BFloat16*)we_ptr,
                (c10::Half*)out_cat.data_ptr(),
                dim,
                eps
            );
        } else {
            fused_mtp_input_kernel<c10::Half, c10::Half><<<blocks, threads, 0, stream>>>(
                (const c10::Half*)hidden.data_ptr(),
                (const c10::Half*)emb.data_ptr(),
                (const c10::Half*)wh_ptr,
                (const c10::Half*)we_ptr,
                (c10::Half*)out_cat.data_ptr(),
                dim,
                eps
            );
        }
    } else if (hidden.scalar_type() == torch::kFloat32) {
        fused_mtp_input_kernel<float, float><<<blocks, threads, 0, stream>>>(
            (const float*)hidden.data_ptr(),
            (const float*)emb.data_ptr(),
            (const float*)wh_ptr,
            (const float*)we_ptr,
            (float*)out_cat.data_ptr(),
            dim,
            eps
        );
    }
}





