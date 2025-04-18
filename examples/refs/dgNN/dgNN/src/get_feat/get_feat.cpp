#include <torch/types.h>
#include <torch/torch.h>
#include <vector>

torch::Tensor
get_feat_cuda(const int num_slots, torch::Tensor indices,
        torch::Tensor data_table, torch::Tensor memory);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_feat", &get_feat_cuda, "Get Data Batch");
}