#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


#include <cuda_fp8.h>   // for __nv_fp8_e4m3
#include <cuda_bf16.h>  // for __nv_bfloat16 if you want bf16
#include <cuda_fp16.h>  // for __half
#include <stdio.h>



#include <cstdint>
#include <limits>
#include <cmath>

#include <tuple>



std::tuple<torch::Tensor, torch::Tensor> build_quantization_meta_cuda(
    torch::Tensor X, //R*G, 128
    int64_t bit_width,
    bool symmetric
);

std::tuple<torch::Tensor, torch::Tensor> mse_quantization_grid_cuda(
    torch::Tensor X, //R*G, 128
    torch::Tensor scales,
    torch::Tensor qzeros,
    torch::Tensor candidates,
    float norm,
    int64_t bit_width,
    bool symmetric
);

torch::Tensor gptq_solver_cuda(
    torch::Tensor W,       // [C, R]
    torch::Tensor U,  // [C, C]
    torch::Tensor scales,  //{C, R}
    torch::Tensor qzeros, //{C, R}
    int64_t bits
);

torch::Tensor babai_solver_cuda(
    torch::Tensor weight,      // [C, R]
    torch::Tensor A,           // [C, C] upper-tri = chol(H)^T
    torch::Tensor scales, // [R, G, 1] or [R*G, 1]
    torch::Tensor qzeros, // [R, G, 1] or [R*G, 1]
    int64_t group_size,
    int64_t bits,
    int64_t block_size,
    torch::Tensor g_idx,
    int G
);

std::tuple<torch::Tensor, torch::Tensor> sparsegptq14_solver_cuda(
    torch::Tensor W,       // [C, R]
    torch::Tensor U,  // [C, C]
    torch::Tensor scales,  //{C, R}
    torch::Tensor qzeros, //{C, R}
    int64_t bits
);

torch::Tensor pack_sparsegptq14_to_u64x2_cuda(
    torch::Tensor qweight_rc,
    torch::Tensor M,
    torch::Tensor scales
);

torch::Tensor moe_proj_unstructured_sparse14_int4symq_gemm(
    torch::Tensor qW2S1u64, // [G2, R, 2] | G2=ceil(C/64) | ulonglong2 | Packing format defined above
    torch::Tensor X         // [N, C] | bfloat16
);

torch::Tensor usp14w4a16sym_sm80_fused_moe_w13_gemm(
    torch::Tensor W13,  //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=2I | C=H
    torch::Tensor X, //[N, C] permuted along N
    torch::Tensor offsets, // [E+1]
    torch::Tensor U //[#active experts <= E]
);

torch::Tensor usp14w4a16sym_sm80_fused_moe_w2_gemm(
    torch::Tensor W2,  //[E, G2, R] | G32=(C/32), G2 = G32/2 | R=H | C=I
    torch::Tensor X2, //[N, C] permuted along N
    torch::Tensor offsets, // [E+1]
    torch::Tensor U //[#active experts <= E]
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_group_meta_packed",
        &build_quantization_meta_cuda,
        "Build packed quantization metadata per group (scale, zero, flags)"
    );
    m.def(
        "mse_scale_groups_packed",
        &mse_quantization_grid_cuda,
        "Refine packed scales per group using MSE search over shrink factors"
    );

    m.def(
        "gptq_solver",
        &gptq_solver_cuda,
        "GPTQ Solver"
    );

    m.def(
        "sparsegptq14_solver",
        &sparsegptq14_solver_cuda,
        "GPTQ Solver"
    );

    m.def(
        "sparsegptq14_solver",
        &sparsegptq14_solver_cuda,
        "SPARSEGPTQ 1:4 Solver"
    );

    m.def(
        "pack_sparsegptq14_to_u64x2",
        &pack_sparsegptq14_to_u64x2_cuda,
        "SPARSEGPTQ 1:4 Packer for inference"
    );

    m.def(
        "sparsegptq14_gemm",
        &moe_proj_unstructured_sparse14_int4symq_gemm,
        "SPARSEGPTQ 1:4 GEMM for inference, tuned for deepseek v3.2"
    );

    m.def(
        "sparsegptq14_grouped_gemm_w13",
        &usp14w4a16sym_sm80_fused_moe_w13_gemm,
        "SPARSEGPTQ 1:4 GEMM for inference, tuned for deepseek v3.2"
    );

    m.def(
        "sparsegptq14_grouped_gemm_w2",
        &usp14w4a16sym_sm80_fused_moe_w2_gemm,
        "SPARSEGPTQ 1:4 GEMM for inference, tuned for deepseek v3.2"
    );

    m.def(
        "babai_solver",
        &babai_solver_cuda,
        "Butterfly Block Babai"
    );
}