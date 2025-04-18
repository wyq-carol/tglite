#include <cuda.h>
#include <vector>
#include <torch/types.h>
#include <stdint.h>

using namespace std;
__global__ void fused_table_all_kernel(
    const int K, const int N, const int mem_dim, const int ef_dim, const int mem_slots, const int ef_slots, 
    const int *__restrict__ indices,       // [K]
    const int *__restrict__ mailbox_table,       // [N,2]
    const int *__restrict__ data_table,       // [M1,2]
    const int *__restrict__ cache_table,       // [M2,2]
    const float *__restrict__ memory,     // [P,dim]

	const int *__restrict__ mailbox_efeat,       // [N]
    const int *__restrict__ efeat_data_table,       // [N, 2]
    const float *__restrict__ efeat_memory,     // [P,dim]
    float *__restrict__ output)      // [K,2*dim]
{
	// 1) 哪个 X 索引：第 t = blockIdx.x
	int t = blockIdx.x;              
	// 2) 取出要处理的行号 row = X[t]
	int row = indices[t];                  

	// 3) 哪个任务：0 或 1, 0 为uniq , 1为nbr, 2 为nfeat
	int task, fid;
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
		int mailbox_ch = mailbox_table[row * 2 + task];
		// 6) 根据 task 选表 B 或 C，取两个 int 并相加
		int bid, sid;
		int memory_pos;  
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
		int efeat_ch = mailbox_efeat[row];
		int bid, sid;
		int memory_pos;
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
	) {

	const int K   = indices.size(0);
	const int N   = mailbox_table.size(0);
	const int mem_dim = memory.size(1);
	const int ef_dim = efeat_memory.size(1);
	const int dim = mem_dim * 2 + ef_dim;
	auto opts = torch::TensorOptions()
					.dtype(torch::kFloat32)
					.device(torch::kCUDA, memory.device().index());
	// output ：N × (2*dim)
	auto output = torch::empty({K, dim}, opts);

	// launch kernel：grid=(N,2,1)， block=(dim,1,1)
	fused_table_all_kernel<<<dim3(K, 1, 1), dim>>>(
		K,  N,  mem_dim, ef_dim , mem_slots, ef_slots,
		indices.data_ptr<int>(), 
		mailbox_table.data_ptr<int>(),
		data_table.data_ptr<int>(), cache_table.data_ptr<int>(),
		memory.data_ptr<float>(),
		mailbox_efeat.data_ptr<int>(),
		efeat_data_table.data_ptr<int>(),
		efeat_memory.data_ptr<float>(),
		output.data_ptr<float>());

	return output;
}
