#pragma once
#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

#include <cuda_runtime.h>

class CUDAImageUtil {
public:
	template<class T> static void copy(T* d_output, T* d_input, unsigned int width, unsigned int height);
	template<class T> static void resample(T* d_output, unsigned int outputWidth, unsigned int outputHeight, T* d_input, unsigned int inputWidth, unsigned int inputHeight);
	static void resampleToIntensity(float* d_output, unsigned int outputWidth, unsigned int outputHeight, uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);
	
};

#endif //CUDA_IMAGE_UTIL_H