#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/matmul.h>

//#include <cuda.h>
//#include <cuda_runtime.h>

#include <vector>
#include<iostream>

const int MAX_THREADS_PER_BLOCK = 512;

template <typename Dtype>
__global__ void PiX_Forward_cuda_kernel(const int n_threads,
                                   int out_channels, int in_channels,
                                   int height, int width,
                                   int zeta,
                                   float tau,
                                   const Dtype* __restrict__ bottom_data,
                                   const Dtype* __restrict__ prob_data,
                                   Dtype* __restrict__ top_data)
{
   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(thread_idx < n_threads)
   {
        unsigned int n = thread_idx / (out_channels*height*width);
        unsigned int c = (thread_idx / (height*width)) % out_channels;
        unsigned int h = (thread_idx / (width)) % height;
        unsigned int w = thread_idx % width;

        Dtype prob = prob_data[n*out_channels+c];

        int c_start = c * zeta;
        int c_end   = c_start + zeta;
        if(c_end > in_channels)
        c_end = in_channels;

        if(prob < tau)
        {
            Dtype max_val = -FLT_MAX;
//            int max_idx = c_start;

            for(int ch = c_start; ch < c_end; ch++)
            {
                unsigned long int bottom_index = ((n*in_channels+ch)*height+h)*width+w;

                Dtype bottom_val = bottom_data[bottom_index];
                if(max_val < bottom_val)
                {
                    max_val = bottom_val;
//                    max_idx = ch;
                }
            }

            top_data[thread_idx] = prob * max_val;
        }
        else
        {
            Dtype avg_val = 0;

            for(int ch = c_start; ch < c_end; ch++)
            {
                unsigned long int bottom_index = ((n*in_channels+ch)*height+h)*width+w;

                avg_val += bottom_data[bottom_index];
            }

            top_data[thread_idx] =  (prob) * avg_val / zeta;
        }
    }
}


template <typename Dtype>
__global__ void PiX_Backward_cuda_kernel(const int n_threads,
                                    int out_channels, int in_channels,
                                    int height, int width,
                                    int zeta,
                                    float tau,
                                    const Dtype* __restrict__ bottom_data,
                                    const Dtype* __restrict__ top_diff,
                                    const Dtype* __restrict__ prob_data,
                                    Dtype* __restrict__ prob_diff,
                                    Dtype* __restrict__ bottom_diff)
{
   int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(thread_idx < n_threads)
   {

        unsigned int n = thread_idx / (out_channels*height*width);
        unsigned int c = (thread_idx / (height*width)) % out_channels;
        unsigned int h = (thread_idx / (width)) % height;
        unsigned int w = thread_idx % width;

        Dtype prob = prob_data[n*out_channels+c];

        int c_start = c * zeta;
        int c_end   = c_start + zeta;
        if(c_end > in_channels)
        c_end = in_channels;


        if(prob < tau)
        {
            Dtype max_val = -FLT_MAX;
            int max_idx = c_start;

            for(int ch = c_start; ch < c_end; ch++)
            {
                unsigned long int bottom_index = ((n*in_channels+ch)*height+h)*width+w;

                Dtype bottom_val = bottom_data[bottom_index];
                if(max_val < bottom_val)
                {
                    max_val = bottom_val;
                    max_idx = ch;
                }
            }

            Dtype top_diff_val = top_diff[thread_idx];

            for(int ch = c_start; ch < c_end; ch++)
            {
                if(ch == max_idx)
                    bottom_diff[((n*in_channels+ch)*height+h)*width+w] = prob * top_diff_val;
                else
                    bottom_diff[((n*in_channels+ch)*height+h)*width+w] = 0;
            }

            prob_diff[((n*out_channels+c)*height+h)*width+w] = max_val * top_diff_val;
        }
        else
        {
            Dtype top_diff_val = top_diff[((n*out_channels+c)*height+h)*width+w];
            Dtype avg_val = 0;

            for(int ch = c_start; ch < c_end; ch++)
            {
                bottom_diff[((n*in_channels+ch)*height+h)*width+w] = (prob) * top_diff_val / zeta;
                avg_val +=  bottom_data[((n*in_channels+ch)*height+h)*width+w];
            }

            avg_val /= zeta;

            prob_diff[((n*out_channels+c)*height+h)*width+w] =   top_diff_val * avg_val;

        }
    }
}





std::vector<torch::Tensor> pix_cuda_forward(
    const int zeta,
    float tau,
    torch::Tensor input,
    torch::Tensor prob)
{

     int n = input.size(0);
     int in_c = input.size(1);
     int h = input.size(2);
     int w = input.size(3);

     int out_c = ceil((float)in_c / zeta);

 torch::Tensor output = torch::zeros({n, out_c, h , w}, input.options());

//https://caffe2.ai/doxygen-c/html/classat_1_1_tensor.html

//std::cout << output.dtype().name()  << "\n";
//for(int i=0;i<output.sizes().size();i++)
//std::cout << output.sizes()[i] << "\n";

//std::cout << input.itemsize()  << "\n";
//std::cout << input.element_size()  << "\n";

  const int n_threads = n * out_c * h * w;

  const dim3 blocks((n_threads - 1) / MAX_THREADS_PER_BLOCK + 1, 1, 1);

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(output.type(), "pix_cuda_forward", ([&] {
    PiX_Forward_cuda_kernel<scalar_t><<<blocks, MAX_THREADS_PER_BLOCK,0,stream>>>(
        n_threads,
        out_c, in_c,
        h, w,
        zeta,
        tau,
        (scalar_t*)input.data_ptr(),
        (scalar_t*)prob.data_ptr(),
        (scalar_t*)output.data_ptr());
  }));

//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
//AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  return {output};
}

std::vector<torch::Tensor> spcf_cuda_backward(
    const int zeta,
    float tau,
    torch::Tensor input,
    torch::Tensor prob,
    torch::Tensor output_grad)
{
      int n = input.size(0);
     int in_c = input.size(1);
     int h = input.size(2);
     int w = input.size(3);

     int out_c = ceil((float)in_c / zeta);

  torch::Tensor input_grad = torch::zeros_like(input);
  torch::Tensor prob_grad = torch::zeros_like(output_grad);

  const int n_threads = n * out_c * h * w;

  const dim3 blocks((n_threads - 1) / MAX_THREADS_PER_BLOCK + 1, 1, 1);

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(output_grad.type(), "pix_cuda_backward", ([&] {
    PiX_Backward_cuda_kernel<scalar_t><<<blocks, MAX_THREADS_PER_BLOCK,0,stream>>>(
        n_threads,
        out_c, in_c,
        h, w,
        zeta,
        tau,
        (scalar_t*)input.data_ptr(),
        (scalar_t*)output_grad.data_ptr(),
        (scalar_t*)prob.data_ptr(),
        (scalar_t*)prob_grad.data_ptr(),
        (scalar_t*)input_grad.data_ptr()
        );
  }));

  torch::Tensor ones = torch::ones({h*w, 1}, input.options());
  prob_grad = prob_grad.reshape({n*out_c, h*w});

  torch::Tensor fusion_prob_grad = at::matmul(prob_grad, ones); 
  fusion_prob_grad = fusion_prob_grad.reshape({n,out_c, 1,1});

//  cublasHandle_t cbublas_handle =  at::cuda::getCurrentCUDABlasHandle();

//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
//AT_CUDA_CHECK(cudaStreamSynchronize(stream));
//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

  return {input_grad, fusion_prob_grad};
}