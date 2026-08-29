#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

// ============================================================================
// D-CFR NATIVE CONTINUOUS C++/CUDA RUNNER FOR QWEN 3.8 27B
// ============================================================================

// 1. In-Stream Activation Kernel (SiLU + Mul in Shared Memory)
__global__ void silu_mul_fused_kernel(
    const half* __restrict__ gate, // [B, S, D]
    const half* __restrict__ up,   // [B, S, D]
    half* __restrict__ out,        // [B, S, D]
    const int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + __expf(-g));
    out[idx] = __float2half(silu_g * u);
}

// 2. In-Stream Residual Add + RMSNorm Kernel
__global__ void add_rmsnorm_fused_kernel(
    half* __restrict__ x,                // [B, S, D] (In/Out Residual)
    const half* __restrict__ residual,   // [B, S, D]
    const half* __restrict__ weight,     // [D]
    half* __restrict__ norm_out,         // [B, S, D]
    const int D,
    const float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    half* x_row = x + row * D;
    const half* r_row = residual + row * D;
    half* out_row = norm_out + row * D;

    // 1. Vectorized in-place residual addition
    for (int i = tid; i < D; i += blockDim.x) {
        float xv = __half2float(x_row[i]);
        float rv = __half2float(r_row[i]);
        x_row[i] = __float2half(xv + rv);
    }
    __syncthreads();

    // 2. Sum of squares reduction in shared memory
    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum_sq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }

    float mean_sq = s_sum[0] / (float)D;
    float rsqrt_val = rsqrtf(mean_sq + eps);

    // 3. Normalization and affine scaling
    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(val * rsqrt_val * w);
    }
}

// Dispatchers
void silu_mul_cuda(torch::Tensor gate, torch::Tensor up, torch::Tensor out) {
    const at::cuda::OptionalCUDAGuard device_guard(gate.device());
    int total = gate.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    silu_mul_fused_kernel<<<blocks, threads, 0, stream>>>(
        (const half*)gate.data_ptr(),
        (const half*)up.data_ptr(),
        (half*)out.data_ptr(),
        total
    );
}

void add_rmsnorm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor norm_out, float eps) {
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    int rows = x.size(0) * x.size(1);
    int D = x.size(2);
    int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    add_rmsnorm_fused_kernel<<<rows, threads, 0, stream>>>(
        (half*)x.data_ptr(),
        (const half*)residual.data_ptr(),
        (const half*)weight.data_ptr(),
        (half*)norm_out.data_ptr(),
        D,
        eps
    );
}
