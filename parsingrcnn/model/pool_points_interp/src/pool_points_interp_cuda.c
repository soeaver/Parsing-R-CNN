#include <THC/THC.h>
#include <math.h>
#include "pool_points_interp_kernel.h"

extern THCState *state;

int pool_points_interp_forward_cuda(float spatial_scale, THCudaTensor * features,
                                   THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 3)
    {
        return 0;
    }

    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    PoolpointsinterpForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, rois_flat,
        output_flat, stream);

    return 1;
}

int pool_points_interp_backward_cuda(float spatial_scale, THCudaTensor * top_grad, 
                                    THCudaTensor * rois, THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 3)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    PoolpointsinterpBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, rois_flat,
        bottom_grad_flat, stream);

    return 1;
}
