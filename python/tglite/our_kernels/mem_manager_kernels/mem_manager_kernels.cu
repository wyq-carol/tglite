#include <cuda.h>
#include <vector>
#include <torch/types.h>
#include <stdint.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <vector>

#include <cub/cub.cuh>
#include <nvToolsExt.h>


using namespace std;
__global__ void get_mem_data_kernel(
    const int32_t mem_slots, const int32_t dim, 
    const int32_t *__restrict__ indices,       			// [K]
    const int32_t *__restrict__ data_table,   	   		// [N,2]
    const bool *__restrict__ data_status,  	  	  	// [N]
    const float *__restrict__ memory,     			// [P,dim]

    float *__restrict__ output)      // [K,dim]
{
	int32_t t = blockIdx.x;              
	int32_t row = indices[t];                  

    int32_t fid = threadIdx.x;

	bool mem_valid = data_status[row];

	int32_t memory_pos;

	if (mem_valid) {
		int32_t bid = data_table[row * 2 + 0];
		int32_t sid = data_table[row * 2 + 1];
		memory_pos = bid * mem_slots + sid;
	} else {
		memory_pos = -1;
	}

	size_t offset = static_cast<size_t>(memory_pos) * dim + fid;
	float v = memory_pos == -1? 0.0 : memory[offset];
	size_t out_col = fid;
	offset = static_cast<size_t>(t) * dim + out_col;

	output[ offset ] = v;
}
			

torch::Tensor
get_mem_data(const int32_t mem_slots,
    torch::Tensor indices,
    torch::Tensor data_table,
    torch::Tensor data_status,
    torch::Tensor memory
    ) {

    const int32_t K   = indices.size(0);
    const int32_t dim = memory.size(1);
    auto opts = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCUDA, memory.device().index());

    // output ：N × dim
    auto output = torch::empty({K, dim}, opts);

    // launch kernel：grid=(N,1,1)， block=(dim,1,1)
    get_mem_data_kernel<<<dim3(K, 1, 1), dim>>>(
        mem_slots,  dim,
        indices.data_ptr<int32_t>(),
        data_table.data_ptr<int32_t>(),
        data_status.data_ptr<bool>(),
        memory.data_ptr<float>(),
        output.data_ptr<float>()
    );

    return output;
	}


__global__ void get_feat_kernel(
    const int32_t num_slots,
    const int32_t feature_size,
    const int32_t *__restrict__ indices,
    const int32_t *__restrict__ data_table, 
    const float *__restrict__ memory,
    float * output)
{
  int32_t lid = blockIdx.x;
  int32_t tid = threadIdx.x;

  int32_t row = indices[lid];

  int32_t blk_id = data_table[row * 2];
  int32_t slot_id = data_table[row * 2 + 1];
  size_t pos = static_cast<size_t>(blk_id) * num_slots + slot_id;
  output[lid * feature_size + tid] = memory[pos * feature_size + tid];
}

torch::Tensor
get_feat_data(const int32_t num_slots, torch::Tensor indices,
              torch::Tensor data_table, torch::Tensor memory)
{
    const int32_t output_x = indices.size(0);
    const int32_t output_y = memory.size(1);
    auto devid = memory.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    torch::Tensor output = torch::empty({output_x, output_y}, options);
    get_feat_kernel<<<dim3(output_x, 1, 1), output_y>>>(
        num_slots,
        output_y,
        indices.data_ptr<int32_t>(), 
        data_table.data_ptr<int32_t>(), 
        memory.data_ptr<float>(),
        output.data_ptr<float>());
    return output;
}


using namespace std;
__global__ void get_mailbox_data_kernel(
    const int32_t K, const int32_t N, const int32_t mem_dim, const int32_t ef_dim, const int32_t mem_slots, const int32_t ef_slots, 
    const int32_t *__restrict__ indices,       // [K]
    const int32_t *__restrict__ mailbox_table,       // [N,2]
    const int32_t *__restrict__ data_table,       // [M1,2]
    const int32_t *__restrict__ cache_table,       // [M2,2]
    const float *__restrict__ memory,     // [P,dim]

	const int32_t *__restrict__ mailbox_efeat,       // [N]
    const int32_t *__restrict__ efeat_data_table,       // [N, 2]
    const float *__restrict__ efeat_memory,     // [P,dim]
    float *__restrict__ output)      // [K,2*dim]
{
	// 1) 哪个 X 索引：第 t = blockIdx.x
	int32_t t = blockIdx.x;              
	// 2) 取出要处理的行号 row = X[t]
	int32_t row = indices[t];                  

	// 3) 哪个任务：0 或 1, 0 为uniq , 1为nbr, 2 为nfeat
	int32_t task, fid;
	if (threadIdx.x < mem_dim) {
		task = 0;
		fid = threadIdx.x;
	} else if (threadIdx.x < mem_dim * 2) {
		task = 1;
		fid = threadIdx.x - mem_dim;
	} else {
		task = 2;
		fid = threadIdx.x - mem_dim * 2;
	}     

	if (task != 2) {
		// 5) 从 A[row,task] 拿到查表索引 aval
		int32_t mailbox_ch = mailbox_table[row * 2 + task];
		// 6) 根据 task 选表 B 或 C，取两个 int32_t 并相加
		int32_t bid, sid;
		int32_t memory_pos;  
		if (mailbox_ch == -1) {
			memory_pos = -1;
		} else if (mailbox_ch < N) {
			bid = data_table[mailbox_ch * 2 + 0];
			sid = data_table[mailbox_ch * 2 + 1];
			memory_pos = bid * mem_slots + sid;
		} else {
			mailbox_ch = mailbox_ch - N;
			bid = cache_table[mailbox_ch * 2 + 0];
			sid = cache_table[mailbox_ch * 2 + 1];
			memory_pos = bid * mem_slots + sid;
		}

		// 7) 从 D[sum_idx, fid] 读值
		size_t offset = static_cast<size_t>(memory_pos) * mem_dim + fid;
		float v = memory_pos == -1 ? 0.0 : memory[offset];

		// 8) 写到 output[t, task*dim + fid]
		size_t out_col = task * mem_dim + fid;
		offset = static_cast<size_t>(t) * (2 * mem_dim + ef_dim) + out_col;
		output[ offset ] = v;
	} else {
		int32_t efeat_ch = mailbox_efeat[row];
		int32_t bid, sid;
		int32_t memory_pos;
		if (efeat_ch == -1) {
			memory_pos = -1;
		} else {
			bid = efeat_data_table[efeat_ch * 2 + 0];
			sid = efeat_data_table[efeat_ch * 2 + 1];
			memory_pos = bid * ef_slots + sid;
		}
		size_t offset = static_cast<size_t>(memory_pos) * ef_dim + fid;
		float v = memory_pos == -1? 0.0 : efeat_memory[offset];
		// 8) 写到 output[t, task*dim + fid]
		size_t out_col = 2 * mem_dim + fid;
		offset = static_cast<size_t>(t) * (2 * mem_dim + ef_dim) + out_col;
		output[ offset ] = v;
}
}
				
torch::Tensor
get_mailbox_data(const int32_t mem_slots,
	const int32_t ef_slots,
	torch::Tensor indices,
	torch::Tensor mailbox_table, 
	torch::Tensor data_table,
	torch::Tensor cache_table,
	torch::Tensor memory,

	torch::Tensor mailbox_efeat,
	torch::Tensor efeat_data_table,
	torch::Tensor efeat_memory
	) {

	const int32_t K   = indices.size(0);
	const int32_t N   = mailbox_table.size(0);
	const int32_t mem_dim = memory.size(1);
	const int32_t ef_dim = efeat_memory.size(1);
	const int32_t dim = mem_dim * 2 + ef_dim;
	auto opts = torch::TensorOptions()
					.dtype(torch::kFloat32)
					.device(torch::kCUDA, memory.device().index());
	// output ：N × (2*dim)
	auto output = torch::empty({K, dim}, opts);

	// launch kernel：grid=(N,2,1)， block=(dim,1,1)
	get_mailbox_data_kernel<<<dim3(K, 1, 1), dim>>>(
		K,  N,  mem_dim, ef_dim , mem_slots, ef_slots,
		indices.data_ptr<int32_t>(), 
		mailbox_table.data_ptr<int32_t>(),
		data_table.data_ptr<int32_t>(), cache_table.data_ptr<int32_t>(),
		memory.data_ptr<float>(),
		mailbox_efeat.data_ptr<int32_t>(),
		efeat_data_table.data_ptr<int32_t>(),
		efeat_memory.data_ptr<float>(),
		output.data_ptr<float>());

	return output;
}

__global__ void extract_top_n_kernel(
    const int* mask,         // shape (N,)
    const int* prefix,       // shape (N,)
    int* output,             // shape (n,)
    int N, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (mask[i] == 1) {
        int pos = prefix[i];
        if (pos < n) {
            output[pos] = i;
        }
    }
}




torch::Tensor get_top_n_true_indices(torch::Tensor mask, int n) {
    int N = mask.size(0);
    auto options = mask.options();
    auto prefix = torch::empty({N}, options);

    // 1. 使用 CUB 进行 prefix sum，手动管理 workspace
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        mask.data_ptr<int>(),
        prefix.data_ptr<int>(), N);

    // 分配 workspace
    auto temp_tensor = torch::empty({static_cast<long>(temp_storage_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(mask.device()));
    d_temp_storage = temp_tensor.data_ptr();

    // 真正执行 prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        mask.data_ptr<int>(),
        prefix.data_ptr<int>(), N);

    // 2. 分配输出
    auto output = torch::empty({n}, options);

    // 3. kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    extract_top_n_kernel<<<blocks, threads>>>(
        mask.data_ptr<int>(),
        prefix.data_ptr<int>(),
        output.data_ptr<int>(),
        N, n
    );

    return output;
}



__device__ int32_t binary_search(const int32_t* A, int32_t size, int32_t target) {
    int32_t left = 0;
    int32_t right = size - 1;

    while (left <= right) {
        int32_t mid = (left + right) / 2;
        int32_t val = A[mid];
        if (val == target) return mid;
        if (val < target)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

__global__ void cache_swap_kernel(
	const int32_t len,
	const int32_t *__restrict__ alloc,                  // [len]
	const int32_t *__restrict__ dump_indices,           // [len]
	int32_t *__restrict__ data_table,
	int32_t *__restrict__ data_ref,
	bool *__restrict__ data_status,
	int32_t *__restrict__ cache_table,
	int32_t *__restrict__ cache_ref,
	int32_t *__restrict__ mailbox
) {
	int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int32_t dump_idx = dump_indices[i];
    int32_t alloc_idx = alloc[i];

    cache_table[alloc_idx * 2 + 0] = data_table[dump_idx * 2 + 0];
    cache_table[alloc_idx * 2 + 1] = data_table[dump_idx * 2 + 1];
    cache_ref[alloc_idx] = data_ref[dump_idx];
    data_status[dump_idx] = false;
    data_ref[dump_idx] = 0;

}

__global__ void swap_mailbox_kernel(
    int32_t* mailbox,  // (N, 2) flattened
    const int32_t* dump_indices,      // sorted (M,)
    const int32_t* alloc,      // (M,)
    const int32_t dump_size,
    const int32_t Nid
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * Nid) return;

    int32_t val = mailbox[idx];

    if (val < 0) return;

    int32_t found_idx = binary_search(dump_indices, dump_size, val);
    if (found_idx != -1) {
        mailbox[idx] = alloc[found_idx] + Nid;
    }
}

__global__ void eq_zero_mask_kernel(const int* input, int* mask, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    mask[i] = (input[i] == 0) ? 1 : 0;
}


void cache_dumper(
    // torch::Tensor cache_mask,
    torch::Tensor dump_indices,
    torch::Tensor data_table,
    torch::Tensor data_ref,
    torch::Tensor data_status, 
    torch::Tensor cache_table,
    torch::Tensor cache_ref,
    torch::Tensor mailbox
) {

    int32_t N = cache_ref.size(0);
    int32_t threads = 256;
    int32_t blocks = (N + threads - 1) / threads;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(cache_ref.device());
    auto cache_mask = torch::empty({N}, options);


    eq_zero_mask_kernel<<<blocks, threads>>>(
        cache_ref.data_ptr<int>(),
        cache_mask.data_ptr<int>(),
        N
    );

    auto alloc = get_top_n_true_indices(cache_mask, dump_indices.size(0));

    int32_t len = dump_indices.size(0);
    int32_t Nid = mailbox.size(0);
    blocks = (len + threads - 1) / threads;

    cache_swap_kernel<<<blocks, threads>>> (
        len,
        alloc.data_ptr<int32_t>(),
        dump_indices.data_ptr<int32_t>(),
        data_table.data_ptr<int32_t>(),
        data_ref.data_ptr<int32_t>(),
        data_status.data_ptr<bool>(),
        cache_table.data_ptr<int32_t>(),
        cache_ref.data_ptr<int32_t>(),
        mailbox.data_ptr<int32_t>()
    );


    blocks = (2 * len + threads - 1) / threads;
    swap_mailbox_kernel<<<blocks, threads>>>(
        mailbox.data_ptr<int32_t>(),
        dump_indices.data_ptr<int32_t>(),
        alloc.data_ptr<int32_t>(),
        len,
        Nid
    );
}

std::vector<torch::Tensor> unique_with_count(torch::Tensor input) {
    auto input_flat = input.contiguous().view(-1);
    auto input_size = input_flat.size(0);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());

    // sort the input
    torch::Tensor sorted = torch::empty_like(input_flat);
    sorted.copy_(input_flat);
    thrust::device_ptr<int> sorted_ptr(sorted.data_ptr<int>());
    thrust::sort(thrust::device, sorted_ptr, sorted_ptr + input_size);

    // allocate output buffers
    torch::Tensor uniq = torch::empty_like(sorted);
    torch::Tensor counts = torch::empty_like(sorted);
    thrust::device_ptr<int> uniq_ptr(uniq.data_ptr<int>());
    thrust::device_ptr<int> counts_ptr(counts.data_ptr<int>());

    // reduce by key
    auto new_end = thrust::reduce_by_key(
        thrust::device,
        sorted_ptr,
        sorted_ptr + input_size,
        thrust::make_constant_iterator(1),
        uniq_ptr,
        counts_ptr
    );

    int num_uniq = new_end.first - uniq_ptr;
    return {
        uniq.slice(0, 0, num_uniq),
        counts.slice(0, 0, num_uniq)
    };
}


__global__ void extract_rows_kernel(
    const int* __restrict__ row_indices,   // [U]
    const int* __restrict__ input,         // [Nid, D]
    int* __restrict__ output,              // [U, D]
    int U, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = U * D;
    if (tid >= total) return;

    int row = tid / D;  // 第几个输出行
    int col = tid % D;  // 第几个列

    int src_row = row_indices[row];  // 对应 old_mailbox 的第几行
    output[tid] = input[src_row * D + col];  // 取出 old_mailbox[src_row, col]
}

// std::pair<thrust::device_vector<int>, thrust::device_vector<int>> unique_with_count_from_vec(
//     // thrust::device_vector<int>& input_vec
//     thrust::device_ptr<int> dev_ptr, int N
// ) {
//     // int N = input_vec.size();

//     // sort
//     // thrust::sort(thrust::device, input_vec.begin(), input_vec.end());
//     thrust::sort(thrust::device, dev_ptr, dev_ptr + N);

//     // output vectors
//     thrust::device_vector<int> uniq_vec(N);
//     thrust::device_vector<int> count_vec(N);

//     auto new_end = thrust::reduce_by_key(
//         thrust::device,
//         dev_ptr,
//         dev_ptr + N,
//         thrust::make_constant_iterator(1),
//         uniq_vec.begin(),
//         count_vec.begin()
//     );

//     int num_uniq = new_end.first - uniq_vec.begin();
//     uniq_vec.resize(num_uniq);
//     count_vec.resize(num_uniq);
//     return {uniq_vec, count_vec};
// }

std::pair<torch::Tensor, torch::Tensor> 
unique_with_count(
    thrust::device_ptr<int> dev_ptr, 
    at::TensorOptions options,
    int N) {

    // int N = input_tensor.size(0);
    // auto options = input_tensor.options();

    // 1. 排序 inplace
    // auto input_ptr = thrust::device_pointer_cast(input_tensor.data_ptr<int>());
    thrust::sort(thrust::device, dev_ptr, dev_ptr + N);

    // 2. 创建最大可能容量的输出 tensor
    torch::Tensor uniq_tensor = torch::empty({N}, options);
    torch::Tensor count_tensor = torch::empty({N}, options);

    auto uniq_ptr = thrust::device_pointer_cast(uniq_tensor.data_ptr<int>());
    auto count_ptr = thrust::device_pointer_cast(count_tensor.data_ptr<int>());

    // 3. reduce_by_key
    auto new_end = thrust::reduce_by_key(
        thrust::device,
        dev_ptr, dev_ptr + N,
        thrust::make_constant_iterator(1),
        uniq_ptr,
        count_ptr
    );

    int num_unique = new_end.first - uniq_ptr;

    // 4. slice 出有效部分
    torch::Tensor uniq_result = uniq_tensor.slice(0, 0, num_unique);
    torch::Tensor count_result = count_tensor.slice(0, 0, num_unique);

    return {uniq_result, count_result};
}


__global__ void pre_ref_update_kernel(
    const int* __restrict__ mailbox,   // [U]
    const int* __restrict__ unique,         // [X]
    const int* __restrict__ count,    // [X]
    int* __restrict__ data_ref,         // [Nid, D]
    int* __restrict__ cache_ref,              // [U, D]
    int Nid, int U
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = U;
    if (tid >= total) return;

    int uniq_i = unique[tid];
    int count_i = count[tid];
    if (uniq_i < 0) {
        return;
    } else if (uniq_i < Nid) {
        int ref = data_ref[uniq_i];
        int new_ref = ref - count_i;
        data_ref[uniq_i] = new_ref;
    } else {
        int ref = cache_ref[uniq_i - Nid];
        int new_ref = ref - count_i;
        cache_ref[uniq_i - Nid] = new_ref;
    }

}


__global__ void make_mask_kernel(
    const int* __restrict__ valid_indices,
    const int* __restrict__ data_ref,
    int* __restrict__ mask,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int idx = valid_indices[tid];
    mask[tid] = (data_ref[idx] > 0) ? 1 : 0;
}


__global__ void fill_output_kernel(
    const int* __restrict__ valid_indices,
    const int* __restrict__ mask,
    const int* __restrict__ prefix,
    int* __restrict__ output,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    if (mask[tid]) {
        int out_idx = prefix[tid];
        output[out_idx] = valid_indices[tid];
    }
}


torch::Tensor dump_launcher(
    int Nid,
    torch::Tensor old_mailbox,
    torch::Tensor uniq,
    torch::Tensor nbr,
    torch::Tensor cache_ref,
    torch::Tensor data_ref,
    torch::Tensor valid_indicies
) {

    int num_rows = uniq.size(0);
    int D = old_mailbox.size(1);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(old_mailbox.device());

    nvtxRangePush("Kernel Launch");
    auto extracted_rows_torch = torch::empty({num_rows * 2}, options);
    int* raw_ptr = extracted_rows_torch.data_ptr<int>();
    // thrust::device_vector<int> extracted_rows();
    int64_t size = num_rows * 2;
    thrust::device_ptr<int> dev_ptr(raw_ptr);

    int threads = 256;
    int blocks  = (num_rows * 2 + threads - 1) / threads;
    extract_rows_kernel<<<blocks, threads>>>(
        uniq.data_ptr<int>(),
        old_mailbox.data_ptr<int>(),
        raw_ptr,
        num_rows, D
    );

    auto [uniq_vec, count_vec] = unique_with_count(dev_ptr, options, size);
    nvtxRangePop();

    blocks  = ((int)uniq_vec.size(0) + threads - 1) / threads;
    pre_ref_update_kernel<<<blocks, threads>>>(
    	old_mailbox.data_ptr<int>(),
        uniq_vec.data_ptr<int>(),
        count_vec.data_ptr<int>(),
        data_ref.data_ptr<int>(),
        cache_ref.data_ptr<int>(),
        Nid, (int)uniq_vec.size(0)
    );

    // preform valid_indices sort
    int N = valid_indicies.size(0);
    if (N == 0) {
        return torch::empty({0}, options);
    }
    

    thrust::device_ptr<int> ptr = thrust::device_pointer_cast(valid_indicies.data_ptr<int>());
    thrust::sort(thrust::device, ptr, ptr + valid_indicies.size(0));

    auto mask   = torch::empty({N}, options);
    auto prefix = torch::empty({N}, options);

    blocks = (N + threads - 1) / threads;

    make_mask_kernel<<<blocks, threads>>>(
        valid_indicies.data_ptr<int>(),
        data_ref.data_ptr<int>(),
        mask.data_ptr<int>(),
        N
    );

    // prefix sum
    thrust::device_ptr<int> mask_ptr(mask.data_ptr<int>());
    thrust::device_ptr<int> prefix_ptr(prefix.data_ptr<int>());
    thrust::exclusive_scan(thrust::device, mask_ptr, mask_ptr + N, prefix_ptr);

    // int total = mask_ptr[N - 1] + prefix_ptr[N - 1];  // 最后一个位置
    int last_mask = 0, last_prefix = 0;
    cudaMemcpy(&last_mask,   mask.data_ptr<int>()   + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_prefix, prefix.data_ptr<int>() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total = last_mask + last_prefix;
    if (total == 0) {
        auto out = torch::empty({0}, options);
        return out;
    }

    auto output = torch::empty({total}, options);

    fill_output_kernel<<<blocks, threads>>>(
        valid_indicies.data_ptr<int>(),
        mask.data_ptr<int>(),
        prefix.data_ptr<int>(),
        output.data_ptr<int>(),
        N
    );

    return output;

    // torch::Tensor out_uniq = torch::empty({(int)uniq_vec.size()}, options);

    // cudaMemcpy(out_uniq.data_ptr<int>(), thrust::raw_pointer_cast(uniq_vec.data()), out_uniq.numel() * sizeof(int), cudaMemcpyDeviceToDevice);



    // return out_uniq;
}

__global__ void check_valid_kernel(
    const int* indices,
    const bool* data_status,
    int* valid_mask,
    int* invalid_mask,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    bool status = data_status[indices[i]];
    valid_mask[i] = status ? 1 : 0;
    invalid_mask[i] = status ? 0 : 1;
}


std::tuple<torch::Tensor, torch::Tensor> check_data_valid(
    torch::Tensor indices,
    torch::Tensor data_status
) {
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(data_status.dtype() == torch::kBool, "data_status must be bool");
    TORCH_CHECK(indices.is_cuda() && data_status.is_cuda(), "must be CUDA tensors");

    int N = indices.size(0);
    auto options = indices.options();

    if (N == 0) {
        return {
            torch::empty({0}, options),
            torch::empty({0}, options)
        };
    }

    // Allocate masks and prefix
    auto valid_mask   = torch::empty({N}, torch::dtype(torch::kInt32).device(indices.device()));
    auto invalid_mask = torch::empty({N}, torch::dtype(torch::kInt32).device(indices.device()));
    auto valid_prefix   = torch::empty({N}, valid_mask.options());
    auto invalid_prefix = torch::empty({N}, valid_mask.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    check_valid_kernel<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        data_status.data_ptr<bool>(),
        valid_mask.data_ptr<int>(),
        invalid_mask.data_ptr<int>(),
        N
    );

    // Prefix sum
    thrust::device_ptr<int> vmask_ptr(valid_mask.data_ptr<int>());
    thrust::device_ptr<int> imask_ptr(invalid_mask.data_ptr<int>());
    thrust::device_ptr<int> vprefix_ptr(valid_prefix.data_ptr<int>());
    thrust::device_ptr<int> iprefix_ptr(invalid_prefix.data_ptr<int>());

    thrust::exclusive_scan(thrust::device, vmask_ptr, vmask_ptr + N, vprefix_ptr);
    thrust::exclusive_scan(thrust::device, imask_ptr, imask_ptr + N, iprefix_ptr);

    // Count valid and invalid
    int valid_total, invalid_total;
    int last_valid_mask, last_invalid_mask;
    cudaMemcpy(&valid_total, vprefix_ptr.get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&invalid_total, iprefix_ptr.get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_valid_mask, valid_mask.data_ptr<int>() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_invalid_mask, invalid_mask.data_ptr<int>() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    valid_total += last_valid_mask;
    invalid_total += last_invalid_mask;

    auto valid_out = torch::empty({valid_total}, options);
    auto invalid_out = torch::empty({invalid_total}, options);

    // Fill outputs
    fill_output_kernel<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        valid_mask.data_ptr<int>(),
        valid_prefix.data_ptr<int>(),
        valid_out.data_ptr<int>(),
        N
    );

    fill_output_kernel<<<blocks, threads>>>(
        indices.data_ptr<int>(),
        invalid_mask.data_ptr<int>(),
        invalid_prefix.data_ptr<int>(),
        invalid_out.data_ptr<int>(),
        N
    );

    return {invalid_out, valid_out};
}




__global__ void allocate_space_kernel(
    const int* alloc,         // shape (N, 2)
    const int* indices,       // shape (N,)
    int* space_status,       // shape (num_blocks * num_slots_per_block)
    bool* data_status,        // shape (N,)
    const int* space_table,   // shape (num_blocks * num_slots_per_block, 2)
    int* data_table,          // shape (num_blocks * num_slots_per_block, 2)
    int num_slots_per_block,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // int row_id = alloc[2 * i + 0];
    // int col_id  = alloc[2 * i + 1];
    // int flat_idx = row_id * num_slots_per_block + col_id;
    int flat_idx = alloc[i];

    space_status[flat_idx] = 0;
    data_status[indices[i]] = true;

    data_table[indices[i] * 2 + 0] = space_table[flat_idx * 2 + 0];
    data_table[indices[i] * 2 + 1] = space_table[flat_idx * 2 + 1];

}

void launch_allocate_space_kernel(
    // torch::Tensor alloc,         // int32 [N, 2]
    torch::Tensor indices,       // int32 [N]
    torch::Tensor space_status,  // int32 [num_blocks * num_slots]
    torch::Tensor data_status,   // bool [N]
    torch::Tensor space_table,   // int32 [num_blocks * num_slots]
    torch::Tensor data_table,    // int32
    int num_slots_per_block
) {
    int N = indices.size(0);

    // auto flat_mask = space_status.flatten(); // 展开为 (H * W,)

    auto alloced = get_top_n_true_indices(space_status, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;


    allocate_space_kernel<<<blocks, threads>>>(
        alloced.data_ptr<int>(),
        indices.data_ptr<int>(),
        space_status.data_ptr<int>(),
        data_status.data_ptr<bool>(),
        space_table.data_ptr<int>(),
        data_table.data_ptr<int>(),
        num_slots_per_block,
        N
    );
}


__global__ void write_data_to_memory_kernel(
    const int* data_table,    // shape: (total_table_size, 2)
    const int* indices,       // shape: (N,)
    const float* data,          // shape: (N,)
    float* memory,              // shape: (total_memory_size,)
    int num_slots_per_block,
    int feature_size,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int index = indices[i]; 

    int block = data_table[index * 2 + 0];
    int slot  = data_table[index * 2 + 1];
    int pos   = block * num_slots_per_block + slot;

    for (int j = 0; j < feature_size; ++j) {
        memory[pos * feature_size + j] = data[i * feature_size + j];
    }
}
void write_data_to_memory(
    torch::Tensor data_table,    // int32 [N, 2]
    torch::Tensor indices,       // int32 [M]
    torch::Tensor data,          // float [M]
    torch::Tensor memory,        // float [total_memory_size]
    int num_slots_per_block
){
    int M = indices.size(0);
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    write_data_to_memory_kernel<<<blocks, threads>>>(
        data_table.data_ptr<int>(),
        indices.data_ptr<int>(),
        data.data_ptr<float>(),
        memory.data_ptr<float>(),
        num_slots_per_block,
        data.size(1),
        M
    );
}

__global__ void add_counts_kernel(
    const int* unique,       // shape (N,)
    const int* counts,       // shape (N,)
    int* data_ref,           // shape (任意，但 unique[i] 是 data_ref 的合法索引)
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int idx = unique[i];
    data_ref[idx] += counts[i];  // 无需 atomic，因为 unique 中无重复
}

void add_counts(torch::Tensor unique, torch::Tensor counts, torch::Tensor data_ref) {
    int N = unique.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    add_counts_kernel<<<blocks, threads>>>(
        unique.data_ptr<int>(),
        counts.data_ptr<int>(),
        data_ref.data_ptr<int>(),
        N
    );
}
