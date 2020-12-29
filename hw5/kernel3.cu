#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define thread_num 16

/*inline void CUDA_ERROR_CHECK(const cudaError_t &err){
	if(err != cudaSuccess){
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}*/

__device__ int mandel(float x, float y, int maxIterations){
	float zx, zy, newzx, newzy, a, b;
	zx = x;
	zy = y;
	int i = 0;
	
	for(i = 0 ; i < maxIterations ; ++i){
		a = zx*zx;
		b = zy*zy;
		if(a + b > 4.0f) break;

		newzx = a - b;
		newzy = 2.f * zx * zy;
		zx = newzx + x;
		zy = newzy + y;
	}

	return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *d_res, int resX, int resY, int maxIterations, size_t x_pixels, size_t y_pixels){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
	int now_x, now_y, i, j, idx;
       	now_x = (blockIdx.x * blockDim.x + threadIdx.x)*x_pixels;
	now_y = (blockIdx.y * blockDim.y + threadIdx.y)*y_pixels;

	//now_x = (blockIdx.x * blockDim.x + threadIdx.x);
	//now_y = (blockIdx.y * blockDim.y + threadIdx.y);

	float x, y;
	/*for(i = now_y ; i < resY/thread_num ; i += blockDim.y*y_pixels){
		for(j = now_x ; j < resX/thread_num ; j += blockDim.x*x_pixels){
			x = lowerX + i * resY;
                        y = lowerY + j * resY;
			idx = j*resX+i;
			d_res[idx] = mandel(x, y, maxIterations);
		}
	}*/


	if(now_x >= resX || now_y >= resY) return;
	for(j = now_y ; j < now_y+y_pixels ; j++){
		if(j >= resY) return;
		for(i = now_x ; i < now_x+x_pixels ; i++){
			if(i >= resX) continue;
			x = lowerX + i * stepX;
			y = lowerY + j * stepY;
			idx = j*resX+i;
			d_res[idx] = mandel(x, y, maxIterations);
		}
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	float stepX = (upperX - lowerX) / resX;
	float stepY = (upperY - lowerY) / resY;
       	
	int *d_res, *h;
	int x_pixels = 2, y_pixels = 2;
	size_t pitch;
	
	cudaMallocPitch((void**)&d_res, &pitch, resX*sizeof(int), resY);
	
	int blocksX = (int) ceil(resX/(float)thread_num);
	int blocksY = (int) ceil(resY/(float)thread_num);
	
	dim3 block(thread_num/x_pixels, thread_num/y_pixels);
	dim3 grid(blocksX, blocksY);

	int size;
	size = resX*resY*sizeof(int);

	cudaHostAlloc((void**)&h, size, cudaHostAllocMapped);

	mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, d_res, resX, resY, maxIterations, x_pixels, y_pixels);

	cudaDeviceSynchronize();

	cudaMemcpy(h, d_res, size, cudaMemcpyDeviceToHost);
	memcpy(img, h, size);
	
	cudaFreeHost(h);
	cudaFree(d_res);
}
