#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

/*inline void CUDA_ERROR_CHECK(const cudaError_t &err){
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}*/


__device__ int mandel(float c_re, float c_im, int maxIteration)
{
        float z_re = c_re, z_im = c_im;
        int i;
        for (i = 0; i < maxIteration; ++i)
        {
                if (z_re * z_re + z_im * z_im > 4.f)
                break;

                float new_re = z_re * z_re - z_im * z_im;
                float new_im = 2.f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
        }

        return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int resX, int resY, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
	int now_x, now_y;
       	now_x = blockIdx.x * blockDim.x + threadIdx.x;
	now_y = blockIdx.y * blockDim.y + threadIdx.y;

	if(now_x >= resX || now_y >= resY) return;

	float x, y;
        x = lowerX + now_x * stepX;
	y = lowerY + now_y * stepY;
	int idx;
	idx = now_y*resX+now_x;
	d_res[idx] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	float stepX = (upperX - lowerX) / resX;
	float stepY = (upperY - lowerY) / resY;

	int blocksX = (int) ceil(resX/16.0);
	int blocksY = (int) ceil(resY/16.0);

	dim3 block(16, 16);
	dim3 grid(blocksX, blocksY);

	int *d_res, *h;
	int size;
	size = resX*resY*sizeof(int);
       	size_t pitch;
	
	cudaMallocPitch(&d_res, &pitch, resX*sizeof(int), resY);
	cudaHostAlloc(&h, size, cudaHostAllocMapped);
		
	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, d_res, resX, resY, maxIterations);
		
	cudaDeviceSynchronize();
	
	cudaMemcpy(h, d_res, size, cudaMemcpyDeviceToHost);
	memcpy(img, h, size);
	
	cudaFreeHost(h);
	cudaFree(d_res);
}
