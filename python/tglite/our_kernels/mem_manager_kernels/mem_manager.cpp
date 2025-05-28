#include <torch/types.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>        

torch::Tensor
get_data(
    torch::Tensor indices,
    torch::Tensor data_table,
    torch::Tensor space_table,
    torch::Tensor memory
    );

torch::Tensor
concat_mailbox(
	torch::Tensor indices,
	torch::Tensor mailbox_table, 
    torch::Tensor mem_space_table,
	torch::Tensor mem_memory,

	torch::Tensor mailbox_efeat,
	torch::Tensor efeat_data_table,
    torch::Tensor efeat_space_table,
	torch::Tensor efeat_memory
	);

void cache_dumper(
    // torch::Tensor cache_mask,
    torch::Tensor dump_indices,
    torch::Tensor data_table,
    torch::Tensor data_ref,
    torch::Tensor data_status, 
    torch::Tensor cache_table,
    torch::Tensor cache_ref,
    torch::Tensor mail_box
);

// std::vector<torch::Tensor> unique_with_count(torch::Tensor input);

torch::Tensor dump_launcher(
    int Nid,
    torch::Tensor old_mailbox,
    torch::Tensor uniq,
    torch::Tensor nbr,
    torch::Tensor cache_ref,
    torch::Tensor data_ref,
    torch::Tensor valid_indicies,
    torch::Tensor unique,
    torch::Tensor counts
);

std::tuple<torch::Tensor, torch::Tensor> check_data_valid(
    torch::Tensor indices,
    torch::Tensor data_status
);

void launch_allocate_space_kernel(
    // torch::Tensor alloc,         // int32 [N, 2]
    torch::Tensor indices,       // int32 [N]
    torch::Tensor space_status,  // bool [num_blocks, num_slots]
    torch::Tensor data_status,   // bool [N]
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

torch::Tensor get_top_n_true_indices(torch::Tensor mask, int n);
void add_counts(torch::Tensor unique, torch::Tensor counts, torch::Tensor data_ref);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_data", &get_data, "Get data");
    m.def("concat_mailbox", &concat_mailbox, "Concat mailbox");
}


