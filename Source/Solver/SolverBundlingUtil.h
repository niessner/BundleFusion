#pragma once

#ifndef _SOLVER_Stereo_UTIL_
#define _SOLVER_Stereo_UTIL_

#include "../SolverUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 512 // keep consistent with the CPU
#define WARP_SIZE 32

__inline__ __device__ float warpReduce(float val) {
	int offset = 32 >> 1;
	while (offset > 0) {
		val = val + __shfl_down(val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}

//extern __shared__ float bucket[];
//
//inline __device__ void scanPart1(unsigned int threadIdx, unsigned int blockIdx, unsigned int threadsPerBlock, float* d_output)
//{
//	__syncthreads();
//	blockReduce(bucket, threadIdx, threadsPerBlock);
//	if(threadIdx == 0) d_output[blockIdx] = bucket[0];
//}
//
//inline __device__ void scanPart2(unsigned int threadIdx, unsigned int threadsPerBlock, unsigned int blocksPerGrid, float* d_tmp)
//{
//	if(threadIdx < blocksPerGrid) bucket[threadIdx] = d_tmp[threadIdx];
//	else						  bucket[threadIdx] = 0.0f;
//	
//	__syncthreads();
//	blockReduce(bucket, threadIdx, threadsPerBlock);
//	__syncthreads();
//}

#endif
