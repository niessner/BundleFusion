#pragma once

#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#undef max
#undef min

#include <cutil_inline.h>
#include <cutil_math.h>

// Enable run time assertion checking in kernel code
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }
//#define cudaAssert(condition)

#if defined(__CUDA_ARCH__)
#define __CONDITIONAL_UNROLL__ #pragma unroll
#else
#define __CONDITIONAL_UNROLL__ 
#endif


#ifdef __CUDACC__
__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}
#endif

#endif
