
#include "mLibCuda.h"
#include "GlobalDefines.h"
#include "CUDACacheUtil.h"
#include "CUDACameraUtil.h"
#include "../Solver/ICPUtil.h" //for the bilinear...

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 4

#define MAX_DEPTH_LOCAL_FUSE 3.0f

#ifdef CUDACACHE_UCHAR_NORMALS
__global__ void fuseCacheFrames_Kernel(const CUDACachedFrame* d_frames, const int* d_validImages, const float4 intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, const uchar4* d_normals, const float* d_output, float2* d_tmp)
#else
__global__ void fuseCacheFrames_Kernel(const CUDACachedFrame* d_frames, const int* d_validImages, const float4 intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, const float4* d_normals, const float* d_output, float2* d_tmp)
#endif
{
	const unsigned int srcFrameIdx = blockIdx.x + 1; // image index to project from
	if (d_validImages[srcFrameIdx] == 0) return;

	const unsigned int idx = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int srcIdx = idx * gridDim.z + blockIdx.z;

	if (srcIdx < (width * height)) {
		const float4 srcCamPos = d_frames[srcFrameIdx].d_cameraposDownsampled[srcIdx];
		if (srcCamPos.z != MINF && srcCamPos.z < MAX_DEPTH_LOCAL_FUSE) {
#ifdef CUDACACHE_UCHAR_NORMALS
			const uchar4 srcNormalUCHAR4 = d_frames[srcFrameIdx].d_normalsDownsampledUCHAR4[srcIdx];
			if (*(int*)&srcNormalUCHAR4 != 0) {
				const float4x4 transform = d_transforms[srcFrameIdx];
				const float3 srcNormal = transform * make_float3(srcNormalUCHAR4.x, srcNormalUCHAR4.y, srcNormalUCHAR4.z) / 255.0f * 2.0f - 1.0f;
#else
			const float3 srcNormal = transform * make_float3(d_frames[srcFrameIdx].d_normalsDownsampled[srcIdx]);
			if (srcNormal.x != MINF) {
				const float4x4 transform = d_transforms[srcFrameIdx];
#endif
				const float3 tgtCamPos = transform * make_float3(srcCamPos.x, srcCamPos.y, srcCamPos.z);;
				const float2 tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtCamPos);
				const int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
				if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)width && tgtScreenPos.y < (int)height) {
					const unsigned int tgtIdx = tgtScreenPos.y * width + tgtScreenPos.x;
					float baseDepth = d_output[tgtIdx]; //TODO try using precomputed camera space positions too
					if (baseDepth != MINF) {
#ifdef CUDACACHE_UCHAR_NORMALS
						const uchar4 baseNormalUCHAR4 = d_normals[tgtIdx];
						if (*(int*)&baseNormalUCHAR4 != 0) {
							const float3 baseNormal = make_float3(baseNormalUCHAR4.x, baseNormalUCHAR4.y, baseNormalUCHAR4.z) / 255.0f * 2.0f - 1.0f;
#else
						const float3 baseNormal = make_float3(d_normals[tgtIdx]);
						//const float3 baseNormal = make_float3(bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, d_normals, width, height));
						if (baseNormal.x != MINF) {
#endif
							const float3 baseCamPos = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtScreenPos, baseDepth);
							if (length(baseCamPos - tgtCamPos) <= 0.1f && dot(baseNormal, srcNormal) >= 0.97f) {
								//float weight = max(0.0f, 0.5f*((1.0f - length(diff) / 0.1f) + (1.0f - camPosTgt.z / MAX_DEPTH_LOCAL_FUSE)));
								float weight = max(0.0f, (1.0f - tgtCamPos.z / MAX_DEPTH_LOCAL_FUSE));

								float* pf = (float*)d_tmp;
								atomicAdd(&pf[2 * tgtIdx], weight * tgtCamPos.z); //TODO make this efficient
								atomicAdd(&pf[2 * tgtIdx + 1], weight);
							}
						}
					}
				} //projects in image
			} //valid src normal
		} //valid src depth
	} //in image
}

__global__ void normalize_Kernel(unsigned int N, float* d_output, const float2* d_tmp)
{
	const unsigned int tid = threadIdx.y * THREADS_PER_BLOCK_X + threadIdx.x;
	const unsigned int idx = tid * gridDim.x + blockIdx.x;
	if (idx < N) {
		const float cur = d_output[idx];
		if (cur != MINF) {
			const float2 fuse = d_tmp[idx];
			if (fuse.y > 0) {
				float weight = max(0.0f, (1.0f - cur / 3.0f)); //NEEDS TO BE SYNC'D WITH ABOVE
				d_output[idx] = (cur*weight + fuse.x) / (weight + fuse.y);
			}
		}
	}
}

//TODO HERE ANGIE
#ifdef CUDACACHE_UCHAR_NORMALS
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const uchar4* d_normals) 
#elif defined(CUDACACHE_FLOAT_NORMALS)
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const float4* d_normals) 
#endif
{
	cutilSafeCall(cudaMemset(d_tmp, 0, sizeof(float2)*width*height)); //TODO use running average instead?

	const int threadsPerBlock = THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
	const int reductionGlobal = (width*height + threadsPerBlock - 1) / threadsPerBlock;

	dim3 grid(numFrames - 1, 1, reductionGlobal); // each frame projects into first
	dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	fuseCacheFrames_Kernel << <grid, block>> >(d_frames, d_validImages, intrinsics, d_transforms, numFrames, width, height, d_normals, d_output, d_tmp);
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

