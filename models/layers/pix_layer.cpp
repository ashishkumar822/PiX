#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> pix_cuda_forward(
    const int zeta,
    const float tau,
    torch::Tensor input,
    torch::Tensor p);

std::vector<torch::Tensor> pixf_cuda_backward(
    const int zeta,
    float tau,
   torch::Tensor input,
    torch::Tensor p,
    torch::Tensor output_grad);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> pix_forward(
    const int zeta,
    float tau,
    torch::Tensor input,
     torch::Tensor p)
{
  CHECK_INPUT(input);
  CHECK_INPUT(p);
  return pix_cuda_forward(zeta, tau, input, p);
}

std::vector<torch::Tensor> pix_backward(
  const int zeta,
  float tau,
  torch::Tensor input,
  torch::Tensor p,
  torch::Tensor output_grad) {

  CHECK_INPUT(input);
  CHECK_INPUT(p);
  CHECK_INPUT(output_grad);

  return pix_cuda_backward(
    zeta, tau,
    input,
    p,
    output_grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
 {
  m.def("forward", &pix_forward, "PiX forward (CUDA)");
  m.def("backward", &pix_backward, "PiX backward (CUDA)");
}