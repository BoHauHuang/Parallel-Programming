#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int inputSize = imageHeight * imageWidth * sizeof(float);
    int hf = filterWidth/2;
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    //cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, inputSize, NULL, NULL);
    //cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, inputSize, NULL, NULL);
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, inputSize, inputImage, NULL);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize, filter, NULL);

    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, NULL);

    //clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, inputSize, inputImage, 0, NULL, NULL);
    //clEnqueueWriteBuffer(queue, filterBuffer, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(int), &imageHeight);
    //clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 5, sizeof(int), &hf);

    int inSize = (imageWidth*imageHeight)>>2;
    size_t globalThreads[2] = {inSize, 1};
    //size_t localThreads[] = {2, 2};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalThreads, NULL, 0, NULL, NULL);

    clFinish(queue);

    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, inputSize, outputImage, 0, NULL, NULL);

    /*clReleaseKernel(kernel);
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(filterBuffer);
    clReleaseCommandQueue(queue);*/
}
