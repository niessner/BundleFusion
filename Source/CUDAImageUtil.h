#pragma once
#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

#include <cuda_runtime.h>
#include "mLibCuda.h"

class CUDAImageUtil {
public:
	template<class T> static void copy(T* d_output, T* d_input, unsigned int width, unsigned int height);
	//template<class T> static void resample(T* d_output, unsigned int outputWidth, unsigned int outputHeight, T* d_input, unsigned int inputWidth, unsigned int inputHeight);
	static void resampleToIntensity(float* d_output, unsigned int outputWidth, unsigned int outputHeight, uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);
	
	static void resampleFloat(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const float* d_input, unsigned int inputWidth, unsigned int inputHeight);
	static void resampleUCHAR4(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);

	static void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, const float4x4& intrinsicsInv, unsigned int width, unsigned int height);
	static void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);

	static void jointBilateralFilterFloatMap(uchar4* d_output, uchar4* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
};

//TODO 
template void CUDAImageUtil::copy<float>(float*, float*, unsigned int, unsigned int);
template void CUDAImageUtil::copy<uchar4>(uchar4*, uchar4*, unsigned int, unsigned int);
//template void CUDAImageUtil::resample<float>(float*, unsigned int, unsigned int, float*, unsigned int, unsigned int);
//template void CUDAImageUtil::resample<uchar4>(uchar4*, unsigned int, unsigned int, uchar4*, unsigned int, unsigned int);

#endif //CUDA_IMAGE_UTIL_H