#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void bilinear_diag_csc_cuda_kernel(
    const int64_t* __restrict__ t1_indices,
    const int64_t* __restrict__ t1_indptr,
    const scalar_t* __restrict__ t1_data,
    const scalar_t* __restrict__ t2,
    const int64_t* __restrict__ t3_indices,
    const int64_t* __restrict__ t3_indptr,
    const scalar_t* __restrict__ t3_data,
    scalar_t* __restrict__ outp,
    const int64_t xlen,
    const int64_t feat_size
)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t b = blockIdx.y;

    if (i < xlen)
    {
        const int64_t ptr_idx = b * xlen + i;
        const int64_t t1_start = t1_indptr[ptr_idx];
        const int64_t t1_stop = t1_indptr[ptr_idx + 1];
        const int64_t t3_start = t3_indptr[ptr_idx];
        const int64_t t3_stop = t3_indptr[ptr_idx + 1];

        scalar_t _outp = 0;

        for (int64_t t1_idx = t1_start; t1_idx < t1_stop; t1_idx++)
        {
            for (int64_t t3_idx = t3_start; t3_idx < t3_stop; t3_idx++)
            {
                _outp += t2[b * feat_size * feat_size + t1_indices[t1_idx] * feat_size + t3_indices[t3_idx]]
                         * t1_data[t1_idx] * t3_data[t3_idx];
            }
        }
        outp[b * xlen + i] = _outp;
    }
}


at::Tensor bilinear_diag_csc_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    at::Tensor t3_indices,
    at::Tensor t3_indptr,
    at::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen
){
    auto outp = at::zeros({batch_size, xlen}, t2.type());
    auto feat_size = t2.size(1);

    const int threads = 1024;
    const dim3 blocks((xlen + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(t2.type(), "bilinear_diag_csc_cuda", ([&] {
    bilinear_diag_csc_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t1_indices.data<int64_t>(),
        t1_indptr.data<int64_t>(),
        t1_data.data<scalar_t>(),
        t2.data<scalar_t>(),
        t3_indices.data<int64_t>(),
        t3_indptr.data<int64_t>(),
        t3_data.data<scalar_t>(),
        outp.data<scalar_t>(),
        xlen,
        feat_size);
    }));

    return outp;
}
