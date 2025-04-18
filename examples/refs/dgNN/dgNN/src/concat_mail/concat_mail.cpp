#include <torch/types.h>
#include <torch/torch.h>
#include <vector>        

torch::Tensor
get_mailbox_all_data(const int mem_slots,
	const int ef_slots,
    torch::Tensor indices,
    torch::Tensor mailbox_table, 
    torch::Tensor data_table,
    torch::Tensor cache_table,
    torch::Tensor memory,

    torch::Tensor mailbox_efeat,
    torch::Tensor efeat_data_table,
    torch::Tensor efeat_memory
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_mailbox_all_data", &get_mailbox_all_data, "Get mailbox all data");
}