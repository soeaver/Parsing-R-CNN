#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "pool_points_interp_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

    /*** Forward ***/

    __device__ float bilinear_interpolate(const float* bottom_data, const int height, const int width,
                                          float y, float x, const int index /* index for debug only*/) {
            // deal with cases that inverse elements are out of feature map boundary
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                return 0;
            }

            if (y <= 0) {
                y = 0;
            }
            if (x <= 0) {
                x = 0;
            }
            
            int y_low = (int)y;
            int x_low = (int)x;
            int y_high;
            int x_high;
            
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (float)y_low;
            } else {
                y_high = y_low + 1;
            }
            
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (float)x_low;
            } else {
                x_high = x_low + 1;
            }
            
            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1. -ly, hx = 1. - lx;
            // do bilinear interpolation
            float v1 = bottom_data[y_low * width + x_low];
            float v2 = bottom_data[y_low * width + x_high];
            float v3 = bottom_data[y_high * width + x_low];
            float v4 = bottom_data[y_high * width + x_high];
            float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

            float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

            return val;
        }

    __global__ void PoolpointsinterpForward(
        const int nthreads, const float* bottom_data, const float spatial_scale, 
        const int height, const int width, const int channels, 
        const float* bottom_rois, float* top_data
    ) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            int c = index % channels;
            int n = index / channels;
            //
            const float* offset_bottom_rois = bottom_rois + n * 3;

            int roi_batch_ind = n/196; // Should be original !!
            //
            float X_point = offset_bottom_rois[1] * spatial_scale;
            float Y_point = offset_bottom_rois[2] * spatial_scale;

            const float* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

            float val = bilinear_interpolate(offset_bottom_data, height, width, Y_point, X_point, index);
            top_data[index] = val;
        }
    }

    int PoolpointsinterpForwardLaucher(
        const float* bottom_data, const float spatial_scale, const int num_rois, 
        const int height, const int width, const int channels,
        const float* bottom_rois, float* top_data, cudaStream_t stream
    ) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * channels;
        cudaError_t err;


        PoolpointsinterpForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, bottom_data, spatial_scale, height, width, channels,
          bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

    /*** Backward ***/
    inline __device__ float gpu_atomic_add(const float val, float* address);
    inline __device__ float gpu_atomic_add(const float val, float* address) {
        return atomicAdd(address, val);
    }

    __device__ void bilinear_interpolate_gradient(const int height, const int width, float y, float x,
                                                  float& w1, float& w2, float& w3, float& w4,
                                                  int& x_low, int& x_high, int& y_low, int& y_high,
                                                  const int index /* index for debug only*/) {
        // deal with cases that inverse elements are out of feature map boundary
        if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            w1 = w2 = w3 = w4 = 0.;
            x_low = x_high = y_low = y_high = -1;
            return;
        }

        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        y_low = (int)y;
        x_low = (int)x;

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (float)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
        } else {
            x_high = x_low + 1;
        }

        float ly = y - y_low;
        float lx = x - x_low;
        float hy = 1. - ly, hx = 1. - lx;

        w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        return;
    }

    __global__ void PoolpointsinterpBackward(
        const int nthreads, const float* top_diff, const float spatial_scale, 
        const int height, const int width, const int channels, 
        float* bottom_diff, const float* bottom_rois
    ) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            int c = index  % channels;
            int n = index  / channels;

            const float* offset_bottom_rois = bottom_rois + n * 3;
            // int roi_batch_ind = offset_bottom_rois[0];
            int roi_batch_ind = n/196; // Should be original !!

            float X_point = offset_bottom_rois[1] * spatial_scale;
            float Y_point = offset_bottom_rois[2] * spatial_scale;

            float w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            bilinear_interpolate_gradient(height, width, Y_point, X_point,
                w1, w2, w3, w4,
                x_low, x_high, y_low, y_high,
                index);

            float* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
            //
            int top_offset = (n * channels + c) ;
            const float* offset_top_diff = top_diff + top_offset;
            const float top_diff_this_bin = offset_top_diff[0];
            //
            float g1 = top_diff_this_bin * w1 ;
            float g2 = top_diff_this_bin * w2 ;
            float g3 = top_diff_this_bin * w3 ;
            float g4 = top_diff_this_bin * w4 ;
            //
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
            {
              gpu_atomic_add(g1, offset_bottom_diff + y_low * width + x_low);
              gpu_atomic_add(g2, offset_bottom_diff + y_low * width + x_high);
              gpu_atomic_add(g3, offset_bottom_diff + y_high * width + x_low);
              gpu_atomic_add(g4, offset_bottom_diff + y_high * width + x_high);
            } // if
        } // CUDA_1D_KERNEL_LOOP
    } // PPIBackward

    int PoolpointsinterpBackwardLaucher(
        const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, 
        const int height, const int width, const int channels, 
        const float* bottom_rois, float* bottom_diff, cudaStream_t stream
    ) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * channels;
        cudaError_t err;

        PoolpointsinterpBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, top_diff, spatial_scale, height, width, channels,
          bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


#ifdef __cplusplus
}
#endif
