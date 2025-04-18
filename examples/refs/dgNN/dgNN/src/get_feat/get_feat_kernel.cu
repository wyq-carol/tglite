#include <cuda.h>
#include <vector>
#include <torch/types.h>

using namespace std;

__global__ void get_feat_kernel(
    const int num_slots,
    const int feature_size,
    const int *__restrict__ indices,
    const int *__restrict__ data_table, 
    const float *__restrict__ memory,
    float * output)
{
  int lid = blockIdx.x;
  int tid = threadIdx.x;

  int row = indices[lid];

  int blk_id = data_table[row * 2];
  int slot_id = data_table[row * 2 + 1];
  size_t pos = static_cast<size_t>(blk_id) * num_slots + slot_id;
  output[lid * feature_size + tid] = memory[pos * feature_size + tid];
}

torch::Tensor
get_feat_cuda(const int num_slots, torch::Tensor indices,
              torch::Tensor data_table, torch::Tensor memory)
{
  const int output_x = indices.size(0);
  const int output_y = memory.size(1);
  auto devid = memory.device().index();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  torch::Tensor output = torch::empty({output_x, output_y}, options);
  get_feat_kernel<<<dim3(output_x, 1, 1), output_y>>>(
      num_slots,
      output_y,
      indices.data_ptr<int>(), 
      data_table.data_ptr<int>(), 
      memory.data_ptr<float>(),
      output.data_ptr<float>());
  return output;
}