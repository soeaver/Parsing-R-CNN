int pool_points_interp_forward_cuda(float spatial_scale, 
    THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int pool_points_interp_backward_cuda(float spatial_scale,
    THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad);
