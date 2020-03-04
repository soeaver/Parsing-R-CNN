#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>

__global__ void PoolPointsInterpForward(
        const int nthreads, const T* bottom_data,
        const T spatial_scale, const int channels,
        const int height, const int width,
        const T* bottom_rois, T* top_data
    ) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            int c = index % channels;
            int n = index / channels;
            //
            const T* offset_bottom_rois = bottom_rois + n * 3;

            int roi_batch_ind = n/196; // Should be original !!
            //
            T X_point = offset_bottom_rois[1] * spatial_scale;
            T Y_point = offset_bottom_rois[2] * spatial_scale;

            const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

            float val = bilinear_interpolate(offset_bottom_data, height, width, Y_point, X_point, index);
            top_data[index] = val;
        }
    }

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename T>
__global__ void PoolPointsInterpBackward(
        const int nthreads, const T* top_diff, const int num_rois, const T spatial_scale,
        const int channels, const int height, const int width,
        T* bottom_diff, const T* bottom_rois
    )
{
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int c = index  % channels;
    int n = index  / channels;

    const T* offset_bottom_rois = bottom_rois + n * 3;
    // int roi_batch_ind = offset_bottom_rois[0];
    int roi_batch_ind = n/196; // Should be original !!

    T X_point = offset_bottom_rois[1] * spatial_scale;
    T Y_point = offset_bottom_rois[2] * spatial_scale;

    T w1, w2, w3, w4;
    int x_low, x_high, y_low, y_high;

    bilinear_interpolate_gradient(height, width, Y_point, X_point,
        w1, w2, w3, w4,
        x_low, x_high, y_low, y_high,
        index);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    //
    int top_offset = (n * channels + c) ;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[0];
    //
    T g1 = top_diff_this_bin * w1 ;
    T g2 = top_diff_this_bin * w2 ;
    T g3 = top_diff_this_bin * w3 ;
    T g4 = top_diff_this_bin * w4 ;
    //
    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
    {
      atomicAdd(offset_bottom_diff + y_low * width + x_low,static_cast<T>(g1));
      atomicAdd(offset_bottom_diff + y_low * width + x_high,static_cast<T>(g2));
      atomicAdd(offset_bottom_diff + y_high * width + x_low,static_cast<T>(g3));
      atomicAdd(offset_bottom_diff + y_high * width + x_high,static_cast<T>(g4));
    } // if
  } // CUDA_1D_KERNEL_LOOP
} // PPIBackward


at::Tensor PoolPointsInterp_forward_cuda(const at::Tensor& input,
                                   const at::Tensor& rois,
                                   const float spatial_scale) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels}, input.options());
  auto output_size = num_rois * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "PoolPointsInterp_forward", [&] {
     PoolPointsInterpForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         width,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}


at::Tensor PoolPointsInterp_backward_cuda(
                                    const at::Tensor& grad,
                                    const at::Tensor& rois,
                                    const float spatial_scale,
                                    const int batch_size,
                                    const int channels,
                                    const int height,
                                    const int width) {

  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "PoolPointsInterp_backward", [&] {
     PoolPointsInterpBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         width,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
