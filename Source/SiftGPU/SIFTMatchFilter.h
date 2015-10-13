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

	static void filterKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv)
	{
		const unsigned int numImages = siftManager->getNumImages();
		if (numImages <= 1) return;

		// current data
		std::vector<SIFTKeyPoint> keyPoints;
		siftManager->getSIFTKeyPointsDEBUG(keyPoints);
		std::vector<float4x4> transforms(curFrame);
		std::vector<float4x4> transformsInv(curFrame);

		for (unsigned int i = 0; i < curFrame; i++) { // previous frames

			std::vector<uint2> keyPointIndices;
			std::vector<float> matchDistances;
			siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

			float4x4 transform;
			unsigned int newNumMatches =
				filterImagePairKeyPointMatches(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv);
			//std::cout << "(" << curFrame << ", " << i << "): " << newNumMatches << std::endl; 

			if (newNumMatches > 0) {
				transforms[i] = transform;
				transformsInv[i] = transform.getInverse();
			}
			else {
				transforms[i].setValue(0.0f);
				transformsInv[i].setValue(0.0f);
			}

			// copy back
			cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
			if (newNumMatches > 0) {
				cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
				cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
			}
		}
		cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	}
	static void ransacKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, float maxResThresh2, bool debugPrint)
	{
		const unsigned int numImages = siftManager->getNumImages();
		if (numImages <= 1) return;

		if (!s_bInit) {
			std::cout << "warining: initialzing combinations" << std::endl;
			init();
		}

		// current data
		std::vector<SIFTKeyPoint> keyPoints;
		siftManager->getSIFTKeyPointsDEBUG(keyPoints);
		std::vector<float4x4> transforms(curFrame);
		std::vector<float4x4> transformsInv(curFrame);

		for (unsigned int i = 0; i < curFrame; i++) { // previous frames

			std::vector<uint2> keyPointIndices;
			std::vector<float> matchDistances;
			siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

			if (debugPrint) std::cout << "(" << i << ", " << curFrame << ")" << std::endl;
			float4x4 transform;
			unsigned int newNumMatches =
				filterImagePairKeyPointMatchesRANSAC(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv, maxResThresh2,
				(unsigned int)s_combinations[0].size(), s_combinations, debugPrint);

			if (newNumMatches > 0) {
				transforms[i] = transform;
				transformsInv[i] = transform.getInverse();
			}
			else {
				transforms[i].setValue(0.0f);
				transformsInv[i].setValue(0.0f);
			}

			// copy back
			cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
			if (newNumMatches > 0) {
				cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
				cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
			}
		}
		cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	}

	static void ransacKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, float maxResThresh2, bool debugPrint);
	static void filterKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv);

	static void filterBySurfaceArea(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames, const float4x4& siftIntrinsicsInv);

	static void filterByDenseVerify(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames);

	static void filterFrames(SIFTImageManager* siftManager);

	static void init() {
		if (s_bInit) return;
		generateKCombinations(20, 4, s_combinations);
		s_bInit = true;
	}
private:
	static bool s_bInit;
	static std::vector<std::vector<unsigned int>> s_combinations;

	static void generateKCombinations(unsigned int n, unsigned int k, std::vector<std::vector<unsigned int>>& combinations) {
		MLIB_ASSERT(k <= n);
		combinations.clear();
		std::vector<unsigned int> current;
		generateKCombinationsInternal(0, k, n, current, combinations);
	}
	static void generateKCombinationsInternal(unsigned int offset, unsigned int k, unsigned int n, std::vector<unsigned int>& current, std::vector<std::vector<unsigned int>>& combinations) {
		if (k == 0) {
			combinations.push_back(current);
			return;
		}
		for (unsigned int i = offset; i <= n - k; i++) {
			current.push_back(i);
			generateKCombinationsInternal(i + 1, k - 1, n, current, combinations);
			current.pop_back();
		}
	}

	static unsigned int filterImagePairKeyPointMatchesRANSAC(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform, const float4x4& siftIntrinsicsInv, float maxResThresh2,
		unsigned int k, const std::vector<std::vector<unsigned int>>& combinations, bool debugPrint);

	static unsigned int filterImagePairKeyPointMatches(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform, const float4x4& siftIntrinsicsInv);
	static bool filterImagePairBySurfaceArea(const std::vector<SIFTKeyPoint>& keys, float* depth0, float* depth1, const std::vector<uint2>& keyPointIndices, const float4x4& siftIntrinsicsInv);
	// depth0 -> src, depth1 -> tgt
	static bool filterImagePairByDenseVerify(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const uchar4* inputColor,
		const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const uchar4* modelColor, const float4x4& transform,
		unsigned int width, unsigned int height);

	static float2 computeSurfaceArea(const SIFTKeyPoint* keys, const uint2* keyPointIndices, float* depth0, float* depth1, unsigned int numMatches, const float4x4& siftIntrinsicsInv);

	static float2 computeProjectiveError(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const uchar4* inputColor,
		const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const uchar4* modelColor,
		const float4x4& transform, unsigned int width, unsigned int height);

	// subsamples (change in size)
	static void computeCameraSpacePositions(const float* depth, unsigned int width, unsigned int height, float4* out);
	// on subsampled (no change in size)
	static void computeNormals(const float4* input, unsigned int width, unsigned int height, float4* out);
	template<typename T>
	static void reSample(const T* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, T* output);
	static void reSampleColor(const uchar4* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, float4* output);
	static float gaussD(float sigma, int x, int y) {
		return exp(-((x*x + y*y) / (2.0f*sigma*sigma)));
	}
	static void jointBilateralFilter(unsigned int width, unsigned int height, const uchar4* color, const float* depth, uchar4* out, float sigmaD, float sigmaR);

	//!!!TODO CAMERA INFO
	static float2 cameraToDepth(const float4& pos);
	static float cameraToDepthZ(const float4& pos);
	static inline float cameraToKinectProjZ(float z) {
		//!!!TODO PARAMS depthmin depthmax
		return (z - 0.1f) / (5.0f - 0.1f);
	}


	static inline void getBestCorrespondence1x1(
		const int2& screenPos, float4& pTarget, float4& nTarget, uchar4& cTarget,
		const float4* target, const float4* targetNormals, const uchar4* targetColors,
		unsigned int width)
	{
		pTarget = target[screenPos.y * width + screenPos.x];
		cTarget = targetColors[screenPos.y * width + screenPos.x];
		nTarget = targetNormals[screenPos.y * width + screenPos.x];
	}
	static void computeCorrespondences(unsigned int width, unsigned int height,
		const float* inputDepth, const float4* input, const float4* inputNormals, const uchar4* inputColor,
		const float* modelDepth, const float4* model, const float4* modelNormals, const uchar4* modelColor,
		const float4x4& transform, float distThres, float normalThres, float colorThresh, unsigned int level,
		float& sumResidual, float& sumWeight, unsigned int& numCorr);
};

template<typename T>
void SIFTMatchFilter::reSample(const T* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, T* output)
{
	const float scaleWidthFactor = (float)width / (float)newWidth;
	const float scaleHeightFactor = (float)height / (float)newHeight;

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