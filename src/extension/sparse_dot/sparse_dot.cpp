#include <torch/torch.h>
#include <utility>

/* CUDA Declaration */

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
);


std::vector<at::Tensor> csr_dot_diag_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/* CSR dot CSC Implementation */

std::vector<at::Tensor> csr_dot_csc_cpu(
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
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2_indices);
    CHECK_CPU(t2_indptr);
    CHECK_CPU(t2_data);

    std::list<int64_t> out_indices_list[batch_size * out_h];
    std::list<float> out_data_list[batch_size * out_h];
    auto out_indptr = at::zeros({batch_size * out_h + 1}, t1_indptr.type());
    auto t1_indptr_acc = t1_indptr.accessor<int64_t, 1>();
    auto t2_indptr_acc = t2_indptr.accessor<int64_t, 1>();
    auto t1_indices_acc = t1_indices.accessor<int64_t, 1>();
    auto t2_indices_acc = t2_indices.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t t1_start = t1_indptr_acc[b * out_h + i];
            int64_t t1_stop = t1_indptr_acc[b * out_h + i + 1];
            int64_t row_nnz = 0;

            for (int64_t j = 0; j < out_w; j++)
            {
                int64_t t2_start = t2_indptr_acc[b * out_w + j];
                int64_t t2_stop = t2_indptr_acc[b * out_w + j + 1];

                float outp = 0;//at::zeros({}, t1_data.type());
                int64_t t1_ptr_idx = t1_start;
                int64_t t2_ptr_idx = t2_start;

                while (t1_ptr_idx < t1_stop && t2_ptr_idx < t2_stop)
                {
                    int64_t t1_cur_indice = t1_indices_acc[t1_ptr_idx];
                    int64_t t2_cur_indice = t2_indices_acc[t2_ptr_idx];
                    if (t1_cur_indice == t2_cur_indice)
                    {
                        auto tmp = t1_data[t1_ptr_idx] * t2_data[t2_ptr_idx];
                        auto tmp_acc = tmp.accessor<float, 1>();
                        outp += tmp_acc[0];
                        t1_ptr_idx++;
                        t2_ptr_idx++;
                    }
                    else if (t1_cur_indice < t2_cur_indice)
                        t1_ptr_idx++;
                    else
                        t2_ptr_idx++;
                }
                if (outp != 0)
                {
                    out_data_list[b * out_h + i].push_back(outp);
                    out_indices_list[b * out_h + i].push_back(j);
                    row_nnz++;
                }
            }
            out_indptr[b * out_h + i + 1] = out_indptr[b * out_h + i] + row_nnz;
        }
    }

    auto out_indptr_acc = out_indptr.accessor<int64_t, 1>();
    int64_t nnz = out_indptr_acc[-1];
    auto out_indices = at::zeros({nnz}, t1_indices.type());
    auto out_data = at::zeros({nnz}, t1_data.type());
    int64_t idx = 0;
    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            auto * tmp_indices_list = &out_indices_list[b * out_h + i];
            auto * tmp_data_list = &out_data_list[b * out_h + i];
            while (!tmp_indices_list->empty() && !tmp_data_list->empty())
            {
                out_indices[idx] = tmp_indices_list->front();
                tmp_indices_list->pop_front();
                out_data[idx] = tmp_data_list->front();
                tmp_data_list->pop_front();
                idx++;
            }
        }
    }

    return {out_indices, out_indptr, out_data};
}


at::Tensor csr_dot_csc_dense_cuda_wrapper(
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
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2_indices);
    CHECK_INPUT(t2_indptr);
    CHECK_INPUT(t2_data);
    return csr_dot_csc_cuda(t1_indices, t1_indptr, t1_data,
                            t2_indices, t2_indptr, t2_data,
                            batch_size, out_h, out_w);
}


std::vector<at::Tensor> csr_dot_csc(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2_indices,
    at::Tensor t2_indptr,
    at::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.type().is_cuda())
        throw std::runtime_error("Unexpected cuda tensor in sparse dot sparse -> sparse computation.");
    else
        return csr_dot_csc_cpu(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data, batch_size, out_h, out_w);
}

at::Tensor csr_dot_csc_dense_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2_indices,
    at::Tensor t2_indptr,
    at::Tensor t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    return csr_dot_csc_dense_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data,
                                          batch_size, out_h, out_w);
}


/* CSR dot diag implementation */

std::vector<at::Tensor> csr_dot_diag_cpu(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2);
    auto outp_indices = at::clone(t1_indices);
    auto outp_indptr = at::clone(t1_indptr);
    auto outp_data = at::zeros_like(t1_data);

    auto t1_indptr_acc = t1_indptr.accessor<int64_t, 1>();
    auto t1_indices_acc = t1_indices.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t start = t1_indptr_acc[b * out_h + i];
            int64_t stop = t1_indptr_acc[b * out_h + i + 1];
            for (int64_t data_idx = start; data_idx < stop; data_idx++)
            {
                int64_t row_idx = t1_indices_acc[data_idx];
                outp_data[data_idx] = t1_data[data_idx] * t2[b][row_idx];
            }
        }
    }
    return {outp_indices, outp_indptr, outp_data};
}


std::vector<at::Tensor> csr_dot_diag_cuda_wrapper(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2);
    return csr_dot_diag_cuda(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
}


std::vector<at::Tensor> csr_dot_diag(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.type().is_cuda())
        return csr_dot_diag_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
    else
        return csr_dot_diag_cpu(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);

}

/* PyBind Interface */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csr_dot_csc", &csr_dot_csc, "csr sparse matrix dot csc sparse matrix");
  m.def("csr_dot_csc_dense_cuda", &csr_dot_csc_dense_cuda,
        "cuda implementation of csr sparse matrix dot csc sparse matrix, result is dense");
  m.def("csr_dot_diag", &csr_dot_diag, "csr sparse matrix dot a diagonal of dense vector");
}
