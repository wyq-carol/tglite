#include <torch/types.h>
#include <torch/torch.h>
#include <vector>        

torch::Tensor
get_mem_data(const int mem_slots,
    torch::Tensor indices,
    torch::Tensor data_table,
    torch::Tensor data_status,
    torch::Tensor memory
    );

torch::Tensor
get_feat_data(const int num_slots, torch::Tensor indices,
        torch::Tensor data_table, torch::Tensor memory);


torch::Tensor
get_mailbox_data(const int mem_slots,
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

void cache_dumper(
    torch::Tensor alloc,
    torch::Tensor dump_indices,
    torch::Tensor data_table,
    torch::Tensor data_ref,
    torch::Tensor data_status, 
    torch::Tensor cache_table,
    torch::Tensor cache_ref,
    torch::Tensor mail_box
);

std::vector<torch::Tensor> unique_with_count(torch::Tensor input);

torch::Tensor dump_launcher(
    int Nid,
    torch::Tensor old_mailbox,
    torch::Tensor uniq,
    torch::Tensor nbr,
    torch::Tensor cache_ref,
    torch::Tensor data_ref,
    torch::Tensor valid_indicies
);

std::tuple<torch::Tensor, torch::Tensor> check_data_valid(
    torch::Tensor indices,
    torch::Tensor data_status
);

void launch_allocate_space_kernel(
    torch::Tensor alloc,         // int32 [N, 2]
    torch::Tensor indices,       // int32 [N]
    torch::Tensor space_status,  // bool [num_blocks, num_slots]
    torch::Tensor space_table,   // int32 [num_blocks * num_slots]
    torch::Tensor data_table,    // int32
    int num_slots_per_block
);

void write_data_to_memory(
    torch::Tensor data_table,    // int32 [N, 2]
    torch::Tensor indices,       // int32 [N]
    torch::Tensor data,          // int32 [N]
    torch::Tensor memory,        // int32 [total_memory_size]
    int num_slots_per_block
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_mem_data", &get_mem_data, "Get mem data");
    m.def("get_feat_data", &get_feat_data, "Get Data Batch");
    m.def("get_mailbox_data", &get_mailbox_data, "Get mailbox data");
    m.def("cache_dumper", &cache_dumper, "Dump cache");
    m.def("unique_with_count", &unique_with_count, "Unique with count (CUDA)");
    m.def("dump_launcher", &dump_launcher, "Dump launcher");
    m.def("check_data_valid", &check_data_valid, "Check data valid");
    m.def("launch_allocate_space_kernel", &launch_allocate_space_kernel, "Launch allocate space kernel");
    m.def("write_data_to_memory", &write_data_to_memory, "Write data to memory");
}


