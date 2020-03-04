#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor PoolPointsInterp_forward(const at::Tensor& input,
                                    const at::Tensor& rois,
                                    const float spatial_scale) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return PoolPointsInterp_forward_cuda(input, rois, spatial_scale);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
}

at::Tensor PoolPointsInterp_backward(const at::Tensor& grad,
                                     const at::Tensor& rois,
                                     const float spatial_scale,
                                     const int batch_size,
                                     const int channels,
                                     const int height,
                                     const int width) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return PoolPointsInterp_backward_cuda(grad, rois, spatial_scale, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

