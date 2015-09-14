
#include "core-base/common.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#define MLIB_CUDA_SAFE_CALL(b) { if(b != cudaSuccess) throw MLIB_EXCEPTION(cudaGetErrorString(b)); }
#define MLIB_CUDA_SAFE_FREE(b) { if(!b) { MLIB_CUDA_SAFE_CALL(cudaFree(b)); b = NULL; } }
#define MLIB_CUDA_CHECK_ERR(msg) {  cudaError_t err = cudaGetLastError();	if (err != cudaSuccess) { throw MLIB_EXCEPTION(cudaGetErrorString( err )); } }