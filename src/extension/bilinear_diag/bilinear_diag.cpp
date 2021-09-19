#include <torch/torch.h>
#include <utility>

#include <iostream>

/* CUDA Declaration */

at::Tensor bilinear_diag_csc_cuda(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    at::Tensor t3_indices,
    at::Tensor t3_indptr,
    at::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/* Dense Implementation */

at::Tensor bilinear_diag_dense(
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor t3
){
    auto sizes = t1.sizes();
    auto batch_size = sizes[0];
    auto xlen = sizes[1];
    auto outp = at::empty({batch_size, xlen}, t2.type());
    for(int64_t i = 0; i < xlen; i++)
    {
        auto _t1 = at::slice(t1, 1, i, i+1);
        auto tmp = at::bmm(_t1, t2);
        auto _t3 = at::slice(t3, 2, i, i+1);
        auto _outp = at::bmm(tmp, _t3).view(-1);
        for(int64_t j = 0; j < batch_size; j++)
            outp[j][i] = _outp[j];
    }
    return outp;
}


/* COO Sparse Implementation */

bool sort_smaller_than(int64_t main1, int64_t main2, int64_t minor1, int64_t minor2)
{
    if (main1 < main2)
        return true;
    else if (main1 > main2)
        return false;
    else if (minor1 < minor2)
        return true;
    else
        return false;
}


void sort_sparse_helper(at::Tensor main,
                        at::Tensor minor,
                        std::vector<at::Tensor> others,
                        int64_t begin,
                        int64_t end)
{
    if (begin >= end)
        return;

    auto head = begin;
    auto tail = end;
    auto reverse = true;
    auto main_access = main.accessor<int64_t, 1>();
    auto minor_access = minor.accessor<int64_t, 1>();
    while (head != tail)
    {
        if (sort_smaller_than(main_access[tail], main_access[head], minor_access[tail], minor_access[head]))
        {
            //swap
            std::swap(main_access[head], main_access[tail]);
            std::swap(minor_access[head], minor_access[tail]);
            for (auto iter = others.cbegin(); iter != others.cend(); iter++)
            {
                if (iter->dtype() == at::ScalarType::Float)
                {
                    auto others_access = iter->accessor<float ,1>();
                    std::swap(others_access[head], others_access[tail]);
                }
                else if (iter->dtype() == at::ScalarType::Double)
                {
                    auto others_access = iter->accessor<double, 1>();
                    std::swap(others_access[head], others_access[tail]);
                }
                else
                {
                    auto others_access = iter->accessor<int64_t, 1>();
                    std::swap(others_access[head], others_access[tail]);
                }
            }
            reverse = !reverse;
        }
        else
        {
            if (reverse)
                tail--;
            else
                head++;
        }
    }

    auto split = head;
    sort_sparse_helper(main, minor, others, begin, split - 1);
    sort_sparse_helper(main, minor, others, split + 1, end);
}


at::Tensor sort_sparse(at::Tensor ts, int64_t main_dim, int64_t minor_dim)
{
    assert(ts.is_sparse());
    auto max_dim = ts.dim();

    if (main_dim < 0)
        main_dim += max_dim;
    if (minor_dim < 0)
        minor_dim += max_dim;
    assert(0 <= main_dim && main_dim < max_dim);
    assert(0 <= minor_dim && minor_dim < max_dim);
    assert(main_dim != minor_dim);

    auto ind = ts._indices();
    auto data = ts._values();

    auto ind_sizes = ind.sizes();
    auto dim_len = ind_sizes[1];

    std::vector<at::Tensor> others;
    for (int64_t i = 0; i < max_dim; i++)
        if ((i != main_dim) && (i != minor_dim))
            others.push_back(ind[i]);

    others.push_back(data);

    sort_sparse_helper(ind[main_dim], ind[minor_dim], others, 0, dim_len - 1);

    return ts;
}


void split_sorted_coo(at::Tensor t, int64_t xlen_indices[], int64_t xlen_dim)
{
    auto indices = t._indices();
    auto t_nnz = t._nnz();
    auto indices_access = indices.accessor<int64_t, 2>();

    int64_t cur_batch = 0;
    int64_t cur_xlen = 0;
    int64_t xlen_offset = 0;
    for (int64_t i = 0; i < t_nnz;)
    {
        if (indices_access[xlen_dim][i] != cur_xlen)
        {
            xlen_indices[xlen_offset + ++cur_xlen] = i;
        }
        if (indices_access[0][i] != cur_batch)
        {
            xlen_offset = cur_xlen;
            cur_xlen = 0;
        }
        if ((indices_access[xlen_dim][i] == cur_xlen) && (indices_access[0][i] == cur_batch))
            i++;
    }
    xlen_indices[xlen_offset + ++cur_xlen] = t_nnz;
}


at::Tensor bilinear_diag_coo(
    at::Tensor t1,
    at::Tensor t2,
    at::Tensor t3
){
    auto t1_sizes = t1.sizes();
    auto batch_size = t1_sizes[0];
    auto xlen = t1_sizes[1];
    auto feat_size = t1_sizes[2];
    auto outp = at::zeros({batch_size, xlen}, t2.type());

    auto t1_indices = t1._indices();
    auto t1_values = t1._values();
    auto t3_indices = t3._indices();
    auto t3_values = t3._values();

    int64_t t1_xlen_indices[xlen * batch_size + 1] = {0};
    int64_t t3_xlen_indices[xlen * batch_size + 1] = {0};

    auto t1_idx_access = t1_indices.accessor<int64_t, 2>();
    auto t3_idx_access = t3_indices.accessor<int64_t, 2>();

    split_sorted_coo(t1, t1_xlen_indices, 1);
    split_sorted_coo(t3, t3_xlen_indices, 2);

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < xlen; i++)
        {
            auto t1_start = t1_xlen_indices[b * xlen + i];
            auto t1_stop = t1_xlen_indices[b * xlen + i + 1];
            auto t3_start = t3_xlen_indices[b * xlen + i];
            auto t3_stop = t3_xlen_indices[b * xlen + i + 1];

            for (auto t1_idx = t1_start; t1_idx < t1_stop; t1_idx++)
            {
                for (auto t3_idx = t3_start; t3_idx < t3_stop; t3_idx++)
                {
                    outp[b][i] += t2[b][t1_idx_access[2][t1_idx]][t3_idx_access[1][t3_idx]]
                                  * t1_values[t1_idx] * t3_values[t3_idx];
                }
            }

        }
    }
    return outp;
}


/* CSC Sparse Implementation */

at::Tensor bilinear_diag_csc_cpu(
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
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2);
    CHECK_CPU(t3_indices);
    CHECK_CPU(t3_indptr);
    CHECK_CPU(t3_data);

    auto outp = at::zeros({batch_size, xlen}, t2.type());
    auto t1_indptr_acc = t1_indptr.accessor<int64_t, 1>();
    auto t3_indptr_acc = t3_indptr.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < xlen; i++)
        {
            int64_t t1_start = t1_indptr_acc[b * xlen + i];
            int64_t t1_stop = t1_indptr_acc[b * xlen + i + 1];
            int64_t t3_start = t3_indptr_acc[b * xlen + i];
            int64_t t3_stop = t3_indptr_acc[b * xlen + i + 1];

            for (auto t1_idx = t1_start; t1_idx < t1_stop; t1_idx++)
            {
                for (auto t3_idx = t3_start; t3_idx < t3_stop; t3_idx++)
                {
                    outp[b][i] += t2[b][t1_indices[t1_idx]][t3_indices[t3_idx]]
                                  * t1_data[t1_idx] * t3_data[t3_idx];
                }
            }

        }
    }
    return outp;
}


at::Tensor bilinear_diag_csc_cuda_wrapper(
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
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2);
    CHECK_INPUT(t3_indices);
    CHECK_INPUT(t3_indptr);
    CHECK_INPUT(t3_data);
    return bilinear_diag_csc_cuda(t1_indices, t1_indptr, t1_data,
                                  t2,
                                  t3_indices, t3_indptr, t3_data,
                                  batch_size, xlen);
}


at::Tensor bilinear_diag_csc(
    at::Tensor t1_indices,
    at::Tensor t1_indptr,
    at::Tensor t1_data,
    at::Tensor t2,
    at::Tensor t3_indices,
    at::Tensor t3_indptr,
    at::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen
)
{
    if (t1_indices.type().is_cuda())
        return bilinear_diag_csc_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2, t3_indices, t3_indptr, t3_data, batch_size, xlen);

    else
        return bilinear_diag_csc_cpu(t1_indices, t1_indptr, t1_data, t2, t3_indices, t3_indptr, t3_data, batch_size, xlen);
}

/* PyBind Interface */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear_diag", &bilinear_diag_csc, "bilinear diagonal");
}
