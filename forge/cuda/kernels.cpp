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



std::tuple<torch::Tensor, torch::Tensor> build_group_meta_packed_cuda(
    torch::Tensor x_groups,   // [G, group_size], any of {f32,f16,bf16,fp8_e4m3*}
    int64_t bit_width,
    bool symmetric
);

torch::Tensor mse_scale_groups_packed_cuda(
        torch::Tensor x_groups,    // [G, group_size], same dtype set as above
        torch::Tensor p,           // [P], float32 shrink factors
        torch::Tensor qmeta_bytes, // [G, sizeof(QMetaPacked)], uint8
        double maxq,
        double norm
);

torch::Tensor gptq_solver_cuda(
    torch::Tensor weight,
    torch::Tensor hessian_inv,
    torch::Tensor qmeta_bytes,
    int64_t group_size,
    int64_t bits,
    int64_t block_size   // if <= 0, infer
);

std::tuple<torch::Tensor, torch::Tensor> babai_solver_cuda(
    torch::Tensor weight,      // [C, R]
    torch::Tensor A,           // [C, C] upper-tri = chol(H)^T
    torch::Tensor qmeta_bytes, // [R, G, 4] or [R*G, 4]
    int64_t group_size,
    int64_t bits,
    int64_t block_size,
    torch::Tensor g_idx
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "build_group_meta_packed",
        &build_group_meta_packed_cuda,
        "Build packed quantization metadata per group (scale, zero, flags)"
    );
    m.def(
        "mse_scale_groups_packed",
        &mse_scale_groups_packed_cuda,
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