#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/irange.h>
#include <c10/util/Load.h>

namespace custom_unique {

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_cpu_sorted(
    const torch::Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  const auto& input = self.contiguous();
  const int64_t numel = input.numel();
  torch::Tensor output = torch::empty({0}, self.options());
  torch::Tensor inverse_indices = torch::empty({0}, self.options().dtype(torch::kLong));
  torch::Tensor counts = torch::empty({0}, self.options().dtype(torch::kLong));

  if (numel == 0) {
    if (return_inverse) {
      inverse_indices.resize_(input.sizes());
    }
    return std::make_tuple(output, inverse_indices, counts);
  }

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
  if (return_counts) {
    counts.resize_({num_unique});
  }

  auto output_data = output.data_ptr<scalar_t>();
  auto inverse_indices_data = return_inverse ? inverse_indices.data_ptr<int64_t>() : nullptr;
  auto counts_data = return_counts ? counts.data_ptr<int64_t>() : nullptr;

  // Parallelize the computation of inverse indices and counts
  int num_threads = at::get_num_threads();
  std::vector<int64_t> unique_counts(num_threads, 0);
  std::vector<int64_t> offsets(num_threads, 0);

  const int64_t grain_size = at::internal::GRAIN_SIZE;
  at::parallel_for(0, num_unique, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for (const auto i : c10::irange(begin, end)) {
      output_data[i] = input_sorted_data[unique_indices[i]];
      if (return_counts) {
        unique_counts[tid] += unique_indices[i + 1] - unique_indices[i];
      }
    }
  });

  int64_t total_unique_count = 0;
  for (const auto t : c10::irange(num_threads)) {
    offsets[t] = total_unique_count;
    total_unique_count += unique_counts[t];
  }

  if (return_inverse) {
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            int64_t unique_idx = 0;
            for (const auto j : c10::irange(unique_indices.size() - 1)) {
                if (unique_indices[j] <= i && i < unique_indices[j + 1]) {
                    unique_idx = j;
                    break;
                }
            }
            inverse_indices_data[indices_data[i]] = unique_idx;
        }
    });
  }

  if (return_counts) {
    counts_data[0] = unique_indices[1] - unique_indices[0];
    for (const auto i : c10::irange(1, num_unique)) {
      counts_data[i] = unique_indices[i + 1] - unique_indices[i];
    }
  }

  return std::make_tuple(output, inverse_indices, counts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique(
    const torch::Tensor& self,
    const bool return_inverse,
    const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
    return unique_cpu_sorted<scalar_t>(self, return_inverse, return_counts);
  });
}

} // namespace custom_unique

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unique", &custom_unique::unique, "Custom unique operation on CPU",
        py::arg("self"), py::arg("return_inverse"), py::arg("return_counts"));
}