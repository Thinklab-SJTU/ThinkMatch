#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void csr_dot_csc_cuda_kernel(
    const int64_t* __restrict__ t1_indices,
    const int64_t* __restrict__ t1_indptr,
    const scalar_t* __restrict__ t1_data,
    const int64_t* __restrict__ t2_indices,
    const int64_t* __restrict__ t2_indptr,
    const scalar_t* __restrict__ t2_data,
    scalar_t* __restrict__ out_dense,
    const int64_t out_h,
    const int64_t out_w
)
{
    const int64_t ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t b = blockIdx.y;

    if (ij < out_h * out_w)
    {
        const int64_t i = ij / out_w;
        const int64_t j = ij % out_w;

        const int64_t t1_start = t1_indptr[b * out_h + i];
        const int64_t t1_stop = t1_indptr[b * out_h + i + 1];

        const int64_t t2_start = t2_indptr[b * out_w + j];
        const int64_t t2_stop = t2_indptr[b * out_w + j + 1];

        scalar_t outp = 0;
        int64_t t1_ptr_idx = t1_start;
        int64_t t2_ptr_idx = t2_start;

        while (t1_ptr_idx < t1_stop && t2_ptr_idx < t2_stop)
        {
            int64_t t1_cur_indice = t1_indices[t1_ptr_idx];
            int64_t t2_cur_indice = t2_indices[t2_ptr_idx];
            if (t1_cur_indice == t2_cur_indice)
            {
                outp += t1_data[t1_ptr_idx] * t2_data[t2_ptr_idx];
                t1_ptr_idx++;
                t2_ptr_idx++;
            }
            else if (t1_cur_indice < t2_cur_indice)
                t1_ptr_idx++;
            else
                t2_ptr_idx++;
        }
        out_dense[b * out_w * out_h + i * out_w + j] = outp;
    }
}


at::Tensor csr_dot_csc_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2_indices,
    at::Tensor t2_indptr,
    at::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    auto out_dense = at::zeros({batch_size, out_h, out_w}, t1_data.type());

    const int threads = 1024;
    const dim3 blocks((out_h * out_w + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(t1_data.type(), "csr_dot_csc_cuda", ([&] {
    csr_dot_csc_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t1_indices.data<int64_t>(),
        t1_indptr.data<int64_t>(),
        t1_data.data<scalar_t>(),
        t2_indices.data<int64_t>(),
        t2_indptr.data<int64_t>(),
        t2_data.data<scalar_t>(),
        out_dense.data<scalar_t>(),
        out_h,
        out_w);
    }));
    return out_dense;
}
