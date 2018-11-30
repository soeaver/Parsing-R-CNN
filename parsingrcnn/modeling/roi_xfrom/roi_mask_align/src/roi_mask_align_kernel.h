#ifndef _ROI_MASK_ALIGN_KERNEL
#define _ROI_MASK_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIMaskAlignForward(const int nthreads, 
                                    const float* bottom_data,
                                    const float spatial_scale, 
                                    const int height, 
                                    const int width,
                                    const int channels, 
                                    const int aligned_height, 
                                    const int aligned_width, 
                                    const int sampling_ratio,
                                    const float spatial_shift,
                                    const int half_part,
                                    const float roi_scale,
                                    const float* bottom_rois,
                                    float* top_data);

int ROIMaskAlignForwardLaucher(const float* bottom_data, 
                               const float spatial_scale, 
                               const int num_rois, 
                               const int height,
                               const int width, 
                               const int channels, 
                               const int aligned_height,
                               const int aligned_width,  
                               const int sampling_ratio, 
                               const float spatial_shift,
                               const int half_part,
                               const float roi_scale,
                               const float* bottom_rois,
                               float* top_data, 
                               cudaStream_t stream);

__global__ void ROIMaskAlignBackward(const int nthreads, 
                                     const float* top_diff,
                                     const float spatial_scale, 
                                     const int height, 
                                     const int width,
                                     const int channels, 
                                     const int aligned_height, 
                                     const int aligned_width, 
                                     const int sampling_ratio,
                                     const float spatial_shift,
                                     const int half_part,
                                     const float roi_scale,
                                     float* bottom_diff,
                                     const float* bottom_rois);
 
int ROIMaskAlignBackwardLaucher(const float* top_diff,
                                const float spatial_scale,
                                const int batch_size,
                                const int num_rois,
                                const int height,
                                const int width,
                                const int channels,
                                const int aligned_height,
                                const int aligned_width,
                                const int sampling_ratio,
                                const float spatial_shift,
                                const int half_part,
                                const float roi_scale,
                                const float* bottom_rois,
                                float* bottom_diff,
                                cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

