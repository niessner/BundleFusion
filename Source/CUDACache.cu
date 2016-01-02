
#include "mLibCuda.h"
#include "GlobalDefines.h"
#include "CUDACacheUtil.h"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 4

__global__ void fuseCacheFrames_Kernel(const CUDACachedFrame* d_frames, const int* d_validImages, const float4x4 intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float* d_tmp)
{
	const unsigned int srcFrameIdx = blockIdx.x + 1; // image index to project from
	if (d_validImages[srcFrameIdx] == 0) return;

	const unsigned int idx = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int srcIdx = idx * gridDim.z + blockIdx.z;

	if (srcIdx < (width * height)) {
		const float4 srcCamPos = d_frames[srcFrameIdx].d_cameraposDownsampled[srcIdx];
		if (srcCamPos.x != MINF) {
			const float4 tgtCamPos = d_transforms[srcFrameIdx] * srcCamPos;
			const float3 proj = intrinsics * make_float3(tgtCamPos.x, tgtCamPos.y, tgtCamPos.z);
			const int2 tgtScreenPos = make_int2((int)roundf(proj.x / proj.z), (int)roundf(proj.y / proj.z));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)width && tgtScreenPos.y < (int)height) {
				const unsigned int tgtIdx = tgtScreenPos.y * width + tgtScreenPos.x;
				atomicAdd(&d_output[tgtIdx], tgtCamPos.z); //TODO make this efficient
				atomicAdd(&d_tmp[tgtIdx], 1.0f);
			}
		}
	}
}

__global__ void normalize_Kernel(unsigned int N, float* d_output, const float* d_tmp)
{
	const unsigned int tid = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int idx = tid * gridDim.x + blockIdx.x;
	if (idx < N) {
		if (d_output[idx] == 0) {
			d_output[idx] = MINF;
		}
		else {
			float norm = d_tmp[idx];
			if (norm > 1) d_output[idx] /= norm;
		}
	}
}

extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4x4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float* d_tmp) 
{
	cutilSafeCall(cudaMemset(d_output, 0, sizeof(float)*width*height));
	cutilSafeCall(cudaMemset(d_tmp, 0, sizeof(float)*width*height)); //TODO use running average instead?

	const int threadsPerBlock = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
	const int reductionGlobal = (width*height + threadsPerBlock - 1) / threadsPerBlock;
	if (threadsPerBlock * reductionGlobal != width*height) {
		printf("ERROR cache image size %d %d must be divisible by threadsPerBlock %d %d\n",
			width, height, THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	}
	dim3 grid(numFrames, 1, reductionGlobal); // each frame projects into first
	dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	fuseCacheFrames_Kernel << <grid, block>> >(d_frames, d_validImages, intrinsics, d_transforms, numFrames, width, height, d_output, d_tmp);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	normalize_Kernel << <reductionGlobal, block>> >(width*height, d_output, d_tmp);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

