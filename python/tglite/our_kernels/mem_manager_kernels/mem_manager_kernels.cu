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


void cache_dumper(
    torch::Tensor alloc,
    torch::Tensor dump_indices,
    torch::Tensor data_table,
    torch::Tensor data_ref,
    torch::Tensor data_status, 
    torch::Tensor cache_table,
    torch::Tensor cache_ref,
    torch::Tensor mailbox
) {
    int32_t len = dump_indices.size(0);
    int32_t Nid = mailbox.size(0);
    int32_t threads = 256;
    int32_t blocks = (len + threads - 1) / threads;

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

std::pair<thrust::device_vector<int>, thrust::device_vector<int>> unique_with_count_from_vec(
    thrust::device_vector<int>& input_vec
) {
    int N = input_vec.size();

    // sort
    thrust::sort(thrust::device, input_vec.begin(), input_vec.end());

    // output vectors
    thrust::device_vector<int> uniq_vec(N);
    thrust::device_vector<int> count_vec(N);

    auto new_end = thrust::reduce_by_key(
        thrust::device,
        input_vec.begin(),
        input_vec.end(),
        thrust::make_constant_iterator(1),
        uniq_vec.begin(),
        count_vec.begin()
    );

    int num_uniq = new_end.first - uniq_vec.begin();
    uniq_vec.resize(num_uniq);
    count_vec.resize(num_uniq);
    return {uniq_vec, count_vec};
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

    thrust::device_vector<int> extracted_rows(num_rows * 2);

    int threads = 256;
    int blocks  = (num_rows + threads - 1) / threads;
    extract_rows_kernel<<<blocks, threads>>>(
        uniq.data_ptr<int>(),
        old_mailbox.data_ptr<int>(),
        thrust::raw_pointer_cast(extracted_rows.data()),
        num_rows, D
    );
    cudaDeviceSynchronize();

    auto [uniq_vec, count_vec] = unique_with_count_from_vec(extracted_rows);

    blocks  = ((int)uniq_vec.size() + threads - 1) / threads;
    pre_ref_update_kernel<<<blocks, threads>>>(
    	old_mailbox.data_ptr<int>(),
        thrust::raw_pointer_cast(uniq_vec.data()),
        thrust::raw_pointer_cast(count_vec.data()),
        data_ref.data_ptr<int>(),
        cache_ref.data_ptr<int>(),
        Nid, (int)uniq_vec.size()
    );


    torch::Tensor out_uniq = torch::empty({(int)uniq_vec.size()}, options);

    cudaMemcpy(out_uniq.data_ptr<int>(), thrust::raw_pointer_cast(uniq_vec.data()), out_uniq.numel() * sizeof(int), cudaMemcpyDeviceToDevice);



    return out_uniq;
}