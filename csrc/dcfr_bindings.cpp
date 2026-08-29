#include <torch/extension.h>

// Forward declarations
torch::Tensor fast_argmax_cuda(torch::Tensor logits);
void ease_snapshot_native(const std::vector<torch::Tensor>& rec_src, const std::vector<torch::Tensor>& conv_src, std::vector<torch::Tensor>& rec_dst, std::vector<torch::Tensor>& conv_dst);
void ease_restore_native(const std::vector<torch::Tensor>& rec_src, const std::vector<torch::Tensor>& conv_src, std::vector<torch::Tensor>& rec_dst, std::vector<torch::Tensor>& conv_dst);
void silu_mul_cuda(torch::Tensor gate, torch::Tensor up, torch::Tensor out);
void add_rmsnorm_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor norm_out, float eps);

void ssm_conv_tree_cuda(
    torch::Tensor x,
    torch::Tensor weights,
    torch::Tensor conv_state,
    torch::Tensor replay_factors,
    torch::Tensor parent,
    torch::Tensor replay_path,
    torch::Tensor dst
);

void gdn_factor_replay_cuda_op(
    torch::Tensor state,
    torch::Tensor gate,
    torch::Tensor delta,
    torch::Tensor key
);

void gdn_batched_48_layers_replay_cuda_op(
    torch::Tensor all_states,
    torch::Tensor all_gates,
    torch::Tensor all_deltas,
    torch::Tensor all_keys
);

std::pair<torch::Tensor, int> evaluate_speculative_acceptance_cuda(
    torch::Tensor draft_tokens,
    torch::Tensor target_tokens
);

std::tuple<int, int, std::vector<int64_t>> resolve_ease_b2_step_cuda(
    torch::Tensor v_logits,
    const std::vector<int64_t>& cand_a,
    const std::vector<int64_t>& cand_b,
    bool is_b2
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_argmax", &fast_argmax_cuda, "D-CFR Fast Vocab Argmax Reduction (CUDA)");
    m.def("ease_snapshot", &ease_snapshot_native, "EASE Ultra-Fast C++ Snapshot via Async DMA (CUDA)");
    m.def("ease_restore", &ease_restore_native, "EASE Ultra-Fast C++ Restore via Async DMA (CUDA)");
    m.def("resolve_ease_b2_step", &resolve_ease_b2_step_cuda, "EASE B=2 Fast In-Kernel Argmax & Branch Decision Resolver (CUDA)");
    m.def("silu_mul", &silu_mul_cuda, "D-CFR Fused SiLU + Mul In-Stream Kernel (CUDA)");
    m.def("add_rmsnorm", &add_rmsnorm_cuda, "D-CFR Fused Add + RMSNorm In-Stream Kernel (CUDA)");
    m.def("ssm_conv_tree", &ssm_conv_tree_cuda, "D-CFR SSM Convolution Tree Kernel (CUDA)");
    m.def("gdn_factor_replay", &gdn_factor_replay_cuda_op, "D-CFR Fused GDN Factor Replay Kernel (CUDA)");
    m.def("gdn_batched_48_layers_replay", &gdn_batched_48_layers_replay_cuda_op, "D-CFR Batched 48 Layers Fused Replay (CUDA)");
    m.def("evaluate_speculative_acceptance", &evaluate_speculative_acceptance_cuda, "D-CFR In-Kernel Speculative Tree Acceptance Resolver (CUDA)");
}



