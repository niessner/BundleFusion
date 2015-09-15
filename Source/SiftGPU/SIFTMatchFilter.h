#pragma once
#ifndef SIFT_MATCH_FILTER_H
#define SIFT_MATCH_FILTER_H

#include "SIFTImageManager.h"
#include "../CUDACache.h"

class SIFTMatchFilter
{
public:
	SIFTGPU_EXPORT SIFTMatchFilter() {}
	SIFTGPU_EXPORT ~SIFTMatchFilter() {}

	SIFTGPU_EXPORT static void filterKeyPointMatches(SIFTImageManager* siftManager);

	SIFTGPU_EXPORT static void filterBySurfaceArea(SIFTImageManager* siftManager, const std::vector<CUDACache::CUDACachedFrame>& cachedFrames);

	SIFTGPU_EXPORT static void filterByDenseVerify(SIFTImageManager* siftManager, const std::vector<CUDACache::CUDACachedFrame>& cachedFrames);

	SIFTGPU_EXPORT static void filterFrames(SIFTImageManager* siftManager);

	SIFTGPU_EXPORT static void printTimings(const std::string& filename);

	static void free() {
		SAFE_DELETE_ARRAY(s_input);
		SAFE_DELETE_ARRAY(s_inputNormals);
		SAFE_DELETE_ARRAY(s_inputColor);

		SAFE_DELETE_ARRAY(s_model);
		SAFE_DELETE_ARRAY(s_modelNormals);
		SAFE_DELETE_ARRAY(s_modelColor);

		SAFE_DELETE_ARRAY(s_correspondence);
		SAFE_DELETE_ARRAY(s_correspondenceNormals);
	}
private:

	static unsigned int filterImagePairKeyPointMatches(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform);
	static bool filterImagePairBySurfaceArea(const std::vector<SIFTKeyPoint>& keys, float* depth0, float* depth1, const std::vector<uint2>& keyPointIndices);
	// depth0 -> src, depth1 -> tgt
	static bool filterImagePairByDenseVerify(const float* depth0, const float* depth1, const uchar4* color0, const uchar4* color1, const float4x4& transform,
		unsigned int width, unsigned int height);

	static float2 computeSurfaceArea(const SIFTKeyPoint* keys, const uint2* keyPointIndices, float* depth0, float* depth1, unsigned int numMatches);

	static float2 computeProjectiveError(const float* depth0, const float* depth1, const uchar4* color0, const uchar4* color1, const float4x4& transform, unsigned int width, unsigned int height);

	// subsamples (change in size)
	static void computeCameraSpacePositions(const float* depth, unsigned int width, unsigned int height, float3* out);
	// on subsampled (no change in size)
	static void computeNormals(const float3* input, unsigned int width, unsigned int height, float4* out);
	template<typename T>
	static void reSample(const T* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, T* output);
	static void reSampleColor(const uchar4* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, float3* output);
	static float gaussD(float sigma, int x, int y) {
		return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
	}
	static void jointBilateralFilter(unsigned int width, unsigned int height, const uchar4* color, const float* depth, uchar4* out, float sigmaD, float sigmaR);

	//!!!TODO CAMERA INFO
	static float2 cameraToDepth(const float3& pos);
	static float cameraToDepthZ(const float3& pos);
	static inline float cameraToKinectProjZ(float z) {
		//!!!TODO PARAMS depthmin depthmax
		return (z - 0.1f) / (3.0f - 0.1f);
	}


	static inline void getBestCorrespondence1x1(
		const int2& screenPos, float3& pTarget, float4& nTarget, uchar4& cTarget,
		const float3* target, const float4* targetNormals, const uchar4* targetColors,
		unsigned int width)
	{
		pTarget = target[screenPos.y * width + screenPos.x];
		cTarget = targetColors[screenPos.y * width + screenPos.x];
		nTarget = targetNormals[screenPos.y * width + screenPos.x];
	}
	static void computeCorrespondences(unsigned int width, unsigned int height,
		const float* inputDepth, const float3* input, const float4* inputNormals, const uchar4* inputColor,
		const float* modelDepth, const float3* model, const float4* modelNormals, const uchar4* modelColor,
		float3* output, float4* outputNormals,
		const float4x4& transform,
		float distThres, float normalThres, float colorThresh,
		unsigned int level);
	static void computeProjCorrespondenceError(unsigned int width, unsigned int height,
		const float3* input, const float3* target, const float4* targetNormals,
		const float4x4& deltaTransform, float& sumResidual, float& sumWeight, unsigned int& numCorr);

	//tmp buffers for dense verify
	static float3*	 s_input;
	static float4*	 s_inputNormals;
	static uchar4*	 s_inputColor;

	static float3*	 s_model;
	static float4*	 s_modelNormals;
	static uchar4*	 s_modelColor;

	static float3*	 s_correspondence;
	static float4*	 s_correspondenceNormals;
};

template<typename T>
void SIFTMatchFilter::reSample(const T* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, T* output)
{
	const float scaleWidthFactor = (float)(width - 1) / (float)(newWidth - 1);
	const float scaleHeightFactor = (float)(height - 1) / (float)(newHeight - 1);

	for (unsigned int i = 0; i < newHeight; i++) {
		for (unsigned int j = 0; j < newWidth; j++) {
			const unsigned int x = (unsigned int)std::round((float)j * scaleWidthFactor);
			const unsigned int y = (unsigned int)std::round((float)i * scaleHeightFactor);
			assert(y >= 0 && y < height && x >= 0 && x < width);
			output[i * newWidth + j] = input[y * width + x];
		}
	}
}

#endif