
#include "SiftCameraParams.h"

__constant__ SiftCameraParams c_siftCameraParams;

extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params) {
	
	size_t size;
	cutilSafeCall(cudaGetSymbolSize(&size, c_siftCameraParams));
	cutilSafeCall(cudaMemcpyToSymbol(c_siftCameraParams, &params, size, 0, cudaMemcpyHostToDevice));
	
#ifdef DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}