#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>


template <typename scalar_t>
__global__ void csr_dot_diag_cuda_kernel(
    const int64_t* __restrict__ t1_indices,
    const int64_t* __restrict__ t1_indptr,
    const scalar_t* __restrict__ t1_data,
    const scalar_t* __restrict__ t2,
    scalar_t* __restrict__ outp_data,
    const int64_t out_h,
    const int64_t out_w
)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t b = blockIdx.y;

    if (i < out_h)
    {
        const int64_t start = t1_indptr[b * out_h + i];
        const int64_t stop = t1_indptr[b * out_h + i + 1];

        for (int64_t data_idx = start; data_idx < stop; data_idx++)
        {
            int64_t row_idx = t1_indices[data_idx];
            outp_data[data_idx] = t1_data[data_idx] * t2[b * out_w + row_idx];
        }
    }
}


std::vector<at::Tensor> csr_dot_diag_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    auto outp_indices = at::clone(t1_indices);
    auto outp_indptr = at::clone(t1_indptr);
    auto outp_data = at::zeros_like(t1_data);

    const int threads = 1024;
    const dim3 blocks((out_h + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(t1_data.type(), "csr_dot_diag_cuda", ([&] {
    csr_dot_diag_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t1_indices.data<int64_t>(),
        t1_indptr.data<int64_t>(),
        t1_data.data<scalar_t>(),
        t2.data<scalar_t>(),
        outp_data.data<scalar_t>(),
        out_h,
        out_w);
    }));

    return {outp_indices, outp_indptr, outp_data};
}
