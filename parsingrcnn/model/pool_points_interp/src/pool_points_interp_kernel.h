#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void PoolpointsinterpForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int height, const int width, const int channels, 
    const float* bottom_rois, float* top_data);

int PoolpointsinterpForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const float* bottom_rois,
    float* top_data, cudaStream_t stream);

__global__ void PoolpointsinterpBackward(const int nthreads, const float* top_diff,
    const float spatial_scale, const int height, const int width, const int channels, 
    float* bottom_diff, const float* bottom_rois);

int PoolpointsinterpBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, 
    const int num_rois, const int height, const int width, const int channels, const float* bottom_rois,
    float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

