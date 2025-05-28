#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>
#include <c10/util/Load.h>

namespace custom_unique {

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> unique_cpu_sorted(
    const torch::Tensor& self,
    const bool return_inverse) {
  const torch::Tensor& input = self.contiguous();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  int64_t numel = input.numel();
  torch::Tensor output = at::empty({numel}, input.options());
  torch::Tensor inverse_indices = at::empty({0}, self.options().dtype(torch::kLong));
  torch::Tensor counts = at::empty({0}, self.options().dtype(torch::kLong));

  const int64_t numel = input.numel();
  torch::Tensor output = at::empty({0}, self.options());
  torch::Tensor inverse_indices = at::empty({0}, self.options().dtype(torch::kLong));

  if (numel == 0) {
    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
    }
    return std::make_tuple(output, inverse_indices);
  }

  // index of first unique in each consecutive section
  // this is used to compute counts for parallelization purpose
  torch::Tensor unique_index = torch::empty({0}, self.options().dtype(torch::kLong));

  // original behavior with unique on scalar tensor
  // is to return a output size of ([1]), `flatten` here will do the job
  auto input_flattened = input.flatten();

  auto [input_sorted, indices] = input_flattened.sort();

  // Sort the input tensor
  auto [input_sorted, indices] = input.sort();
  auto input_sorted_data = input_sorted.data_ptr<scalar_t>();
  auto indices_data = indices.data_ptr<int64_t>();

  // Find unique elements
  std::vector<int64_t> unique_indices;
  unique_indices.push_back(0);
  for (const auto i : c10::irange(1, numel)) {
    if (input_sorted_data[i] != input_sorted_data[i - 1]) {
      unique_indices.push_back(i);
    }
  }
  unique_indices.push_back(numel);

  int64_t num_unique = unique_indices.size() - 1;
  output.resize_({num_unique});
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
  }

  auto output_data = output.data_ptr<scalar_t>();
  auto inverse_indices_data = return_inverse ? inverse_indices.data_ptr<int64_t>() : nullptr;

  // Parallelize the computation of inverse indices
  const int64_t grain_size = at::internal::GRAIN_SIZE;
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t unique_idx = 0;
      for (const auto j : c10::irange(unique_indices.size() - 1)) {
        if (unique_indices[j] <= i && i < unique_indices[j + 1]) {
          unique_idx = j;
          break;
        }
      }
      if (return_inverse) {
        inverse_indices_data[indices_data[i]] = unique_idx;
      }
    }
  });

  // Fill the output tensor with unique values
  for (const auto i : c10::irange(num_unique)) {
    output_data[i] = input_sorted_data[unique_indices[i]];
  }

  return std::make_tuple(output, inverse_indices);
}

std::tuple<torch::Tensor, torch::Tensor> unique(
    const torch::Tensor& self,
    const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
    return unique_cpu_sorted<scalar_t>(self, return_inverse);
  });
}

} // namespace custom_unique

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unique", &custom_unique::unique, "Custom unique operation on CPU",
        py::arg("self"), py::arg("return_inverse"));
}