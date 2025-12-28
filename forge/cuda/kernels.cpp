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
    torch::Tensor weight,       // [C, R]
    torch::Tensor hessian_inv,  // [C, C]
    torch::Tensor scales, 
    torch::Tensor qzeros,
    int64_t group_size,
    int64_t bits,
    int64_t block_size,
    torch::Tensor g_idx,
    int64_t G
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
        "babai_solver",
        &babai_solver_cuda,
        "Butterfly Block Babai"
    );
}