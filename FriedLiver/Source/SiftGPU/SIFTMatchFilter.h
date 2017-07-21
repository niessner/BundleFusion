#pragma once
#ifndef SIFT_MATCH_FILTER_H
#define SIFT_MATCH_FILTER_H

#include "SIFTImageManager.h"
#include "../CUDACache.h"

class SIFTMatchFilter
{
public:
	SIFTMatchFilter() {}
	~SIFTMatchFilter() {}

	static void filterKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool printDebug);
	static void ransacKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool debugPrint);

	static void ransacKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool debugPrint);
	static void filterKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches);

	static void filterBySurfaceArea(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches);

	static void filterByDenseVerify(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames, const float4x4& depthIntrinsics, float depthMin, float depthMax);

	static void visualizeProjError(SIFTImageManager* siftManager, const vec2ui& imageIndices, const std::vector<CUDACachedFrame>& cachedFrames,
		const float4x4& depthIntrinsics, const float4x4& transformCurToPrv, float depthMin, float depthMax);

	static void filterFrames(SIFTImageManager* siftManager);

	static void init() {
		if (s_bInit) return;
		generateKCombinations(20, 4, s_combinations);
		//generateRandKCombinations(16, 4, 128, s_combinations);
		//generateKCombinations(20, 4, s_combinations, 128);
		s_bInit = true;
	}

	static void debugVis(const SensorData& sd, const vec2ui& imageIndices, const mat4f& transform, unsigned int subsampleFactor = 4);
private:
	static bool s_bInit;
	static std::vector<std::vector<unsigned int>> s_combinations;

	static void generateRandKCombinations(unsigned int n, unsigned int k, unsigned int numGen, std::vector<std::vector<unsigned int>>& combinations) {
		MLIB_ASSERT(k <= n);
		combinations.clear();
		std::vector<unsigned int> indices(n);
		for (unsigned int i = 0; i < n; i++) indices[i] = i;

		combinations.resize(numGen);
		for (unsigned int i = 0; i < numGen; i++) {
			std::random_shuffle(indices.begin(), indices.end());
			combinations[i].resize(k);
			for (unsigned int j = 0; j < k; j++) combinations[i][j] = indices[j];
		}
	}

	static void generateKCombinations(unsigned int n, unsigned int k, std::vector<std::vector<unsigned int>>& combinations, unsigned int maxNumGen = (unsigned int)-1) {
		MLIB_ASSERT(k <= n);
		combinations.clear();
		std::vector<unsigned int> current;
		generateKCombinationsInternal(0, k, n, current, combinations, maxNumGen);
	}
	static void generateKCombinationsInternal(unsigned int offset, unsigned int k, unsigned int n, std::vector<unsigned int>& current, std::vector<std::vector<unsigned int>>& combinations, unsigned int maxNumGen) {
		if (k == 0) {
			if (combinations.size() == maxNumGen) return;
			combinations.push_back(current);
			return;
		}
		for (unsigned int i = offset; i <= n - k; i++) {
			if (combinations.size() == maxNumGen) return;
			current.push_back(i);
			generateKCombinationsInternal(i + 1, k - 1, n, current, combinations, maxNumGen);
			current.pop_back();
		}
	}

	static unsigned int filterImagePairKeyPointMatchesRANSAC(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2,
		unsigned int k, const std::vector<std::vector<unsigned int>>& combinations, bool debugPrint);

	static unsigned int filterImagePairKeyPointMatches(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool printDebug);
	static bool filterImagePairBySurfaceArea(const std::vector<SIFTKeyPoint>& keys, float* depth0, float* depth1, const std::vector<uint2>& keyPointIndices, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches);
	// depth0 -> src, depth1 -> tgt
	static bool filterImagePairByDenseVerify(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const float* inputColor,
		const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const float* modelColor, const float4x4& transform,
		unsigned int width, unsigned int height, const float4x4& depthIntrinsics, float depthMin, float depthMax);

	static float2 computeSurfaceArea(const SIFTKeyPoint* keys, const uint2* keyPointIndices, float* depth0, float* depth1, unsigned int numMatches, const float4x4& siftIntrinsicsInv);

	static float2 computeProjectiveError(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const float* inputColor,
		const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const float* modelColor,
		const float4x4& transform, unsigned int width, unsigned int height, const float4x4& depthIntrinsics, float depthMin, float depthMax);
	//!!!TODO CAMERA INFO
	static float2 cameraToDepth(const float4x4& depthIntrinsics, const float4& pos);
	static float cameraToDepthZ(const float4x4& depthIntrinsics, const float4& pos);
	static inline float cameraToKinectProjZ(float z, float depthMin, float depthMax) {
		return (z - depthMin) / (depthMax - depthMin);
	}
	static vec2f cameraToDepth(const mat4f& depthIntrinsics, const vec3f& pos)
	{
		vec3f p = depthIntrinsics * pos;
		return vec2f(p.x / p.z, p.y / p.z);
	}
	static inline void getBestCorrespondence1x1(
		const vec2i& screenPos, vec3f& pTarget, vec3f& nTarget, vec3uc& cTarget,
		const PointImage& target, const PointImage& targetNormals, const ColorImageR8G8B8& targetColor)
	{
		pTarget = target(screenPos.x, screenPos.y);
		nTarget = targetNormals(screenPos.x, screenPos.y);
		cTarget = targetColor(screenPos.x, screenPos.y);
	}

	static inline void getBestCorrespondence1x1(
		const int2& screenPos, float4& pTarget, float4& nTarget, float& cTarget,
		const float4* target, const float4* targetNormals, const float* targetColors,
		unsigned int width)
	{
		pTarget = target[screenPos.y * width + screenPos.x];
		cTarget = targetColors[screenPos.y * width + screenPos.x];
		nTarget = targetNormals[screenPos.y * width + screenPos.x];
	}
	static void computeCorrespondences(unsigned int width, unsigned int height,
		const float* inputDepth, const float4* input, const float4* inputNormals, const float* inputColor,
		const float* modelDepth, const float4* model, const float4* modelNormals, const float* modelColor,
		const float4x4& transform, float distThres, float normalThres, float colorThresh,
		const float4x4& depthIntrinsics, float depthMin, float depthMax,
		float& sumResidual, float& sumWeight, unsigned int& numCorr);

	static void computeCorrespondencesDEBUG(unsigned int width, unsigned int height,
		const float* inputDepth, const float4* input, const float4* inputNormals, const float* inputColor,
		const float* modelDepth, const float4* model, const float4* modelNormals, const float* modelColor,
		const float4x4& transform, float distThres, float normalThres, float colorThresh,
		const float4x4& depthIntrinsics, float depthMin, float depthMax, float& sumResidual, float& sumWeight, unsigned int& numCorr);

private:
	//! debug
	static DepthImage32 s_debugCorr;
};


#endif