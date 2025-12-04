#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>


#include <cuda_fp8.h>   // for __nv_fp8_e4m3
#include <cuda_bf16.h>  // for __nv_bfloat16 if you want bf16
#include <cuda_fp16.h>  // for __half


#include <cstdint>
#include <limits>
#include <cmath>

#include <tuple>


namespace {


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

}


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
}