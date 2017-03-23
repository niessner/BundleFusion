

#pragma once

#ifndef _IMAGE_MANAGER_H_
#define _IMAGE_MANAGER_H_

#include <windows.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

#include <vector>
#include <cassert>
#include <iostream>
#include <vector>

#include "GlobalDefines.h"
#include "cuda_SimpleMatrixUtil.h"
#include "../CUDACacheUtil.h"
#include "CUDATimer.h"

struct SIFTKeyPoint {
	float2 pos;
	float scale;
	float depth;
};

struct SIFTKeyPointDesc {
	unsigned char feature[128];
};

struct SIFTImageGPU {
	//int*					d_keyPointCounter;	//single counter value per image (into into global array)	//TODO we need this counter if we do multimatching
	SIFTKeyPoint*			d_keyPoints;		//array of key points (index into global array)
	SIFTKeyPointDesc*		d_keyPointDescs;	//array of key point descs (index into global array)
};

struct ImagePairMatch {
	int*		d_numMatches;		//single counter value per image
	float*		d_distances;		//array of distance (one per match)
	uint2*		d_keyPointIndices;	//array of index pair (one per match)	
};

//correspondence_idx -> image_Idx_i,j
struct EntryJ {
	unsigned int imgIdx_i;
	unsigned int imgIdx_j;
	float3 pos_i;
	float3 pos_j;

	__host__ __device__
	void setInvalid() {
		imgIdx_i = (unsigned int)-1;
		imgIdx_j = (unsigned int)-1;
	}
	__host__ __device__
	bool isValid() const {
		return imgIdx_i != (unsigned int)-1;
	}
};



class SIFTImageManager {
public:
	friend class SIFTMatchFilter;
	friend class TestMatching;

	SIFTImageManager(unsigned int maxImages = 500,
		unsigned int maxKeyPointsPerImage = 4096);

	~SIFTImageManager();


	SIFTImageGPU& getImageGPU(unsigned int imageIdx);

	const SIFTImageGPU& getImageGPU(unsigned int imageIdx) const;

	unsigned int getNumImages() const;

	unsigned int getNumKeyPointsPerImage(unsigned int imageIdx) const;
	unsigned int getMaxNumKeyPointsPerImage() const { return m_maxKeyPointsPerImage; }

	SIFTImageGPU& createSIFTImageGPU();

	void finalizeSIFTImageGPU(unsigned int numKeyPoints);

	// ------- image-image matching (API for the Sift matcher)
	ImagePairMatch& SIFTImageManager::getImagePairMatch(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset);

	ImagePairMatch& getImagePairMatchDEBUG(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset)
	{
		assert(prevImageIdx < getNumImages());
		assert(curImageIdx < getNumImages());
		keyPointOffset = make_uint2(m_numKeyPointsPerImagePrefixSum[prevImageIdx], m_numKeyPointsPerImagePrefixSum[curImageIdx]);
		return m_currImagePairMatches[prevImageIdx];
	}

	//void resetImagePairMatches(unsigned int numImageMatches = (unsigned int)-1) {

	//	if (numImageMatches == (unsigned int)-1) numImageMatches = m_maxNumImages;
	//	assert(numImageMatches < m_maxNumImages);

	//	CUDA_SAFE_CALL(cudaMemset(d_currNumMatchesPerImagePair, 0, sizeof(int)*numImageMatches));
	//	CUDA_SAFE_CALL(cudaMemset(d_currMatchDistances, 0, sizeof(float)*numImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	//	CUDA_SAFE_CALL(cudaMemset(d_currMatchKeyPointIndices, -1, sizeof(uint2)*numImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));

	//	CUDA_SAFE_CALL(cudaMemset(d_currNumFilteredMatchesPerImagePair, 0, sizeof(int)*numImageMatches));
	//	CUDA_SAFE_CALL(cudaMemset(d_currFilteredMatchDistances, 0, sizeof(float)*numImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	//	CUDA_SAFE_CALL(cudaMemset(d_currFilteredMatchKeyPointIndices, -1, sizeof(uint2)*numImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	//}
	void reset() {
		m_SIFTImagesGPU.clear();
		m_numKeyPointsPerImage.clear();
		m_numKeyPointsPerImagePrefixSum.clear();
		m_numKeyPoints = 0;
		m_globNumResiduals = 0;
		m_bFinalizedGPUImage = false;
		MLIB_CUDA_SAFE_CALL(cudaMemset(d_globNumResiduals, 0, sizeof(int)));

		m_validImages.clear();
		m_validImages.resize(m_maxNumImages, 0);
		m_validImages[0] = 1; // first is valid
	}

	//sorts the key point matches inside image pair matches
	void SortKeyPointMatchesCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames);

	void FilterKeyPointMatchesCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxKabschRes2);

	void FilterMatchesBySurfaceAreaCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& colorIntrinsicsInv, float areaThresh);

	void FilterMatchesByDenseVerifyCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, unsigned int imageWidth, unsigned int imageHeight,
		const float4x4& intrinsics, const CUDACachedFrame* d_cachedFrames,
		float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh, float sensorDepthMin, float sensorDepthMax);

	int VerifyTrajectoryCU(unsigned int numImages, float4x4* d_trajectory,
		unsigned int imageWidth, unsigned int imageHeight,
		const float4x4& intrinsics, const CUDACachedFrame* d_cachedFrames,
		float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh,
		float sensorDepthMin, float sensorDepthMax);

	void AddCurrToResidualsCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& colorIntrinsicsInv);

	void InvalidateImageToImageCU(const uint2& imageToImageIdx);

	void CheckForInvalidFramesSimpleCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars);
	void CheckForInvalidFramesCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars);

	//unsigned int FuseToGlobalKeyCU(SIFTImageGPU& globalImage, const float4x4* transforms, const float4x4& colorIntrinsics, const float4x4& colorIntrinsicsInv);

	unsigned int filterFrames(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames);

	//void computeSiftTransformCU(const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform, float4x4* d_siftTrajectory, unsigned int curFrameIndexAll, unsigned int curFrameIndex, float4x4* d_currIntegrateTrans);

	//only markers for up to num images have been set properly
	const std::vector<int>& getValidImages() const { return m_validImages; }
	void invalidateFrame(unsigned int frame) { m_validImages[frame] = 0; }

	void updateGPUValidImages() {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int)*getNumImages(), cudaMemcpyHostToDevice));
	}
	const int* getValidImagesGPU() const { return d_validImages; }

	//actually debug function...
	int* debugGetNumRawMatchesGPU() {
		return d_currNumMatchesPerImagePair;
	}
	int* debugGetNumFiltMatchesGPU() {
		return d_currNumFilteredMatchesPerImagePair;
	}

	unsigned int getTotalNumKeyPoints() const { return m_numKeyPoints; }
	void setNumImagesDEBUG(unsigned int numImages) {
		MLIB_ASSERT(numImages <= m_SIFTImagesGPU.size());
		if (numImages == m_SIFTImagesGPU.size()) return;
		m_SIFTImagesGPU.resize(numImages);
		m_numKeyPointsPerImage.resize(numImages);
		m_numKeyPointsPerImagePrefixSum.resize(numImages);
		m_numKeyPoints = m_numKeyPointsPerImagePrefixSum.back();
	}
	void setGlobalCorrespondencesDEBUG(const std::vector<EntryJ>& correspondences) {
		//warning: does not update d_globMatchesKeyPointIndices
		MLIB_ASSERT(correspondences.size() < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (m_maxNumImages*(m_maxNumImages - 1)) / 2); //less than max #residuals
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globMatches, correspondences.data(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyHostToDevice));
		m_globNumResiduals = (unsigned int)correspondences.size();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globNumResiduals, &m_globNumResiduals, sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
	void setValidImagesDEBUG(const std::vector<int>& valid) {
		m_validImages = valid;
	}
	void getNumRawMatchesDEBUG(std::vector<unsigned int>& numMatches) const {
		MLIB_ASSERT(getNumImages() > 1);
		if (getCurrentFrame() + 1 == getNumImages()) numMatches.resize(getNumImages() - 1);
		else										 numMatches.resize(getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost));
	}
	void getNumFiltMatchesDEBUG(std::vector<unsigned int>& numMatches) const {
		MLIB_ASSERT(getNumImages() > 1);
		if (getCurrentFrame() + 1 == getNumImages()) numMatches.resize(getNumImages() - 1);
		else										 numMatches.resize(getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost));
	}
	void getSIFTKeyPointsDEBUG(std::vector<SIFTKeyPoint>& keys) const {
		keys.resize(m_numKeyPoints);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(keys.data(), d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));
	}
	void getSIFTKeyPointDescsDEBUG(std::vector<SIFTKeyPointDesc>& descs) const {
		descs.resize(m_numKeyPoints);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(descs.data(), d_keyPointDescs, sizeof(SIFTKeyPointDesc) * descs.size(), cudaMemcpyDeviceToHost));
	}
	void getRawKeyPointIndicesAndMatchDistancesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		if (numMatches > MAX_MATCHES_PER_IMAGE_PAIR_RAW) numMatches = MAX_MATCHES_PER_IMAGE_PAIR_RAW;
		keyPointIndices.resize(numMatches);
		matchDistances.resize(numMatches);
		if (numMatches > 0) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_RAW, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistances.data(), d_currMatchDistances + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_RAW, sizeof(float) * numMatches, cudaMemcpyDeviceToHost));
		}
	}
	void getFiltKeyPointIndicesAndMatchDistancesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumFilteredMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		MLIB_ASSERT(numMatches <= MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		keyPointIndices.resize(numMatches);
		matchDistances.resize(numMatches);
		if (numMatches > 0) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistances.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(float) * numMatches, cudaMemcpyDeviceToHost));
		}
	}
	void getCurrMatchKeyPointIndicesDEBUG(std::vector<uint2>& keyPointIndices, std::vector<unsigned int>& numMatches, bool filtered) const
	{
		numMatches.resize(getNumImages());
		if (filtered)	{ MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost)); }
		else			{ MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost)); }		
		if (filtered)	{ keyPointIndices.resize(numMatches.size() * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED); MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices, sizeof(uint2) * keyPointIndices.size(), cudaMemcpyDeviceToHost)); }
		else			{ keyPointIndices.resize(numMatches.size() * MAX_MATCHES_PER_IMAGE_PAIR_RAW); MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currMatchKeyPointIndices, sizeof(uint2) * keyPointIndices.size(), cudaMemcpyDeviceToHost)); }
	}
	void getFiltKeyPointIndicesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumFilteredMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		keyPointIndices.resize(numMatches);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
	}
	const EntryJ* getGlobalCorrespondencesGPU() const { return d_globMatches; }
	EntryJ* getGlobalCorrespondencesGPU() { return d_globMatches; }
	unsigned int getNumGlobalCorrespondences() const { return m_globNumResiduals; }
	const float4x4* getFiltTransformsToWorldGPU() const { return d_currFilteredTransformsInv; }
	const int* getNumFiltMatchesGPU() const { return d_currNumFilteredMatchesPerImagePair; }

	void fuseToGlobal(SIFTImageManager* global, const float4x4& colorIntrinsics, const float4x4* d_transforms,
		const float4x4& colorIntrinsicsInv) const;
	void computeTracks(const std::vector<float4x4>& trajectory, const std::vector<EntryJ>& correspondences, const std::vector<uint2>& correspondenceKeyIndices,
		std::vector< std::vector<std::pair<uint2, float3>> >& tracks) const;

	//try to match previously invalidated images
	bool getTopRetryImage(unsigned int& idx) {
		if (m_imagesToRetry.empty()) return false;
		idx = m_imagesToRetry.front();
		m_imagesToRetry.pop_front();
		return true;
	}
	void addToRetryList(unsigned int idx) {
		m_imagesToRetry.push_front(idx);
	}
	unsigned int getCurrentFrame() const { return m_currentImage; }
	void setCurrentFrame(unsigned int idx) { m_currentImage = idx; }

	static void TestSVDDebugCU(const float3x3& m);

	void saveToFile(const std::string& s);

	void loadFromFile(const std::string& s);

	void evaluateTimings() {
		if (m_timer) m_timer->evaluate(true);
	}
	void setTimer(CUDATimer* timer) {
		m_timer = timer;
	}

	void lock() {
		m_mutex.lock();
	}

	void unlock() {
		m_mutex.unlock();
	}
private:
	std::mutex m_mutex;

	void alloc();
	void free();
	void initializeMatching();

	void fuseLocalKeyDepths(std::vector<SIFTKeyPoint>& globalKeys, const std::vector<float*>& depthFrames,
	//void fuseLocalKeyDepths(std::vector<SIFTKeyPoint>& globalKeys, const std::vector<CUDACachedFrame>& cachedFrames,
		unsigned int depthWidth, unsigned int depthHeight,
		const std::vector<float4x4>& transforms, const std::vector<float4x4>& transformsInv, 
		const float4x4& siftIntrinsicsInv, const float4x4& depthIntrinsics, const float4x4& depthIntrinsicsInv) const;

	// keypoints & descriptors
	std::vector<SIFTImageGPU>	m_SIFTImagesGPU;			// TODO if we ever do a global multi-match kernel, then we need this array on the GPU
	bool						m_bFinalizedGPUImage;

	unsigned int				m_numKeyPoints;						//current fill status of key point counts
	std::vector<unsigned int>	m_numKeyPointsPerImage;				//synchronized with key point counters;
	std::vector<unsigned int>	m_numKeyPointsPerImagePrefixSum;	//prefix sum of the array above

	SIFTKeyPoint*			d_keyPoints;		//array of all key points ever found	(linearly stored)
	SIFTKeyPointDesc*		d_keyPointDescs;	//array of all descriptors every found	(linearly stored)
	//int*					d_keyPointCounters;	//atomic counter once per image			(GPU array of int-valued counters)	// TODO incase we do a multi-match kernel we need this


	// matching
	std::vector<ImagePairMatch>	m_currImagePairMatches;		//image pair matches of the current frame

	int*			d_currNumMatchesPerImagePair;	// #key point matches
	float*			d_currMatchDistances;			// array of distances per key point pair
	uint2*			d_currMatchKeyPointIndices;		// array of indices to d_keyPoints
	
	int*			d_currNumFilteredMatchesPerImagePair;	// #key point matches
	float*			d_currFilteredMatchDistances;			// array of distances per key point pair
	uint2*			d_currFilteredMatchKeyPointIndices;		// array of indices to d_keyPoints
	float4x4*		d_currFilteredTransforms;				// array of transforms estimated in the first filter stage, prev to cur
	float4x4*		d_currFilteredTransformsInv;			// array of transforms estimated in the first filter stage, cur to prev

	std::vector<int> m_validImages;
	int*			 d_validImages; // for check invalid frames kernel only (from residual invalidation) //TODO some way to not have both?

	unsigned int	m_globNumResiduals;		//#residuals (host)
	int*			d_globNumResiduals;		//#residuals (device)
	EntryJ*			d_globMatches;			
	uint2*			d_globMatchesKeyPointIndices;
	int*			d_validOpt;

	unsigned int m_maxNumImages;			//max number of images maintained by the manager
	unsigned int m_maxKeyPointsPerImage;	//max number of SIFT key point that can be detected per image

	std::list<unsigned int> m_imagesToRetry;
	unsigned int			m_currentImage;

	CUDATimer *m_timer;
};


#endif

