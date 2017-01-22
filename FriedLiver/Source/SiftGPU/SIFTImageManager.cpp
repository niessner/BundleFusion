#include "stdafx.h"
#include "SIFTImageManager.h"
#include "../GlobalBundlingState.h"
#include "../GlobalAppState.h"
#include "mLibCuda.h"

SIFTImageManager::SIFTImageManager(unsigned int maxImages /*= 500*/, unsigned int maxKeyPointsPerImage /*= 4096*/)
{
	m_maxNumImages = maxImages;
	m_maxKeyPointsPerImage = maxKeyPointsPerImage;

	m_timer = NULL;
	m_currentImage = 0;
	alloc();
}

SIFTImageManager::~SIFTImageManager()
{
	free();
}

SIFTImageGPU& SIFTImageManager::getImageGPU(unsigned int imageIdx)
{
	assert(m_bFinalizedGPUImage);
	return m_SIFTImagesGPU[imageIdx];
}

const SIFTImageGPU& SIFTImageManager::getImageGPU(unsigned int imageIdx) const
{
	assert(m_bFinalizedGPUImage);
	return m_SIFTImagesGPU[imageIdx];
}

unsigned int SIFTImageManager::getNumImages() const
{
	return (unsigned int)m_SIFTImagesGPU.size();
}

unsigned int SIFTImageManager::getNumKeyPointsPerImage(unsigned int imageIdx) const
{
	return m_numKeyPointsPerImage[imageIdx];
}

SIFTImageGPU& SIFTImageManager::createSIFTImageGPU()
{
	assert(m_SIFTImagesGPU.size() == 0 || m_bFinalizedGPUImage);
	assert(m_SIFTImagesGPU.size() < m_maxNumImages);

	unsigned int imageIdx = (unsigned int)m_SIFTImagesGPU.size();
	m_SIFTImagesGPU.push_back(SIFTImageGPU());

	SIFTImageGPU& imageGPU = m_SIFTImagesGPU.back();

	//imageGPU.d_keyPointCounter = d_keyPointCounters + imageIdx;
	imageGPU.d_keyPoints = d_keyPoints + m_numKeyPoints;
	imageGPU.d_keyPointDescs = d_keyPointDescs + m_numKeyPoints;

	m_bFinalizedGPUImage = false;
	return imageGPU;
}

void SIFTImageManager::finalizeSIFTImageGPU(unsigned int numKeyPoints)
{
	assert(numKeyPoints <= m_maxKeyPointsPerImage);
	assert(!m_bFinalizedGPUImage);

	m_numKeyPointsPerImagePrefixSum.push_back(m_numKeyPoints);
	m_numKeyPoints += numKeyPoints;
	m_numKeyPointsPerImage.push_back(numKeyPoints);
	m_bFinalizedGPUImage = true;
	m_currentImage = (unsigned int)m_SIFTImagesGPU.size() - 1;

	assert(getNumImages() == m_numKeyPointsPerImage.size());
	assert(getNumImages() == m_numKeyPointsPerImagePrefixSum.size());
}

ImagePairMatch& SIFTImageManager::getImagePairMatch(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset)
{
	assert(prevImageIdx < getNumImages());
	assert(curImageIdx < getNumImages());
	keyPointOffset = make_uint2(m_numKeyPointsPerImagePrefixSum[prevImageIdx], m_numKeyPointsPerImagePrefixSum[curImageIdx]);
	return m_currImagePairMatches[prevImageIdx];
}






void SIFTImageManager::saveToFile(const std::string& s)
{
	std::ofstream out(s, std::ios::binary);
	if (!out.is_open()) {
		std::cout << "Error opening " << s << " for write" << std::endl;
		return;
	}

	std::vector<SIFTKeyPoint> keyPoints(m_maxNumImages*m_maxKeyPointsPerImage);
	std::vector<SIFTKeyPointDesc> keyPointDescs(m_maxNumImages*m_maxKeyPointsPerImage);
	std::vector<int> keyPointCounters(m_maxNumImages);

	MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPoints.data(), d_keyPoints, sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointDescs.data(), d_keyPointDescs, sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage, cudaMemcpyDeviceToHost));
	//CUDA_SAFE_CALL(cudaMemcpy(keyPointCounters.data(), d_keyPointCounters, sizeof(int)*m_maxNumImages, cudaMemcpyDeviceToHost));


	const unsigned maxImageMatches = m_maxNumImages;
	std::vector<int> currNumMatchesPerImagePair(maxImageMatches);
	std::vector<float> currMatchDistances(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
	std::vector<uint2> currMatchKeyPointIndices(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currNumMatchesPerImagePair.data(), d_currNumMatchesPerImagePair, sizeof(int)*maxImageMatches, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currMatchDistances.data(), d_currMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currMatchKeyPointIndices.data(), d_currMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToHost));


	std::vector<int> currNumFilteredMatchesPerImagePair(maxImageMatches);
	std::vector<float> currFilteredMatchDistances(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
	std::vector<uint2> currFilteredMatchKeyPointIndices(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
	std::vector < float4x4 > currFilteredTransforms(maxImageMatches);
	std::vector<float4x4> currFilteredTransformsInv(maxImageMatches);

	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currNumFilteredMatchesPerImagePair.data(), d_currNumFilteredMatchesPerImagePair, sizeof(int)*maxImageMatches, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currFilteredMatchDistances.data(), d_currFilteredMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currFilteredMatchKeyPointIndices.data(), d_currFilteredMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currFilteredTransforms.data(), d_currFilteredTransforms, sizeof(float4x4)*maxImageMatches, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currFilteredTransformsInv.data(), d_currFilteredTransformsInv, sizeof(float4x4)*maxImageMatches, cudaMemcpyDeviceToHost));

	std::vector<EntryJ> globMatches(m_globNumResiduals);
	std::vector<uint2> globMatchesKeyPointIndices(m_globNumResiduals);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(globMatches.data(), d_globMatches, sizeof(EntryJ)*m_globNumResiduals, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(globMatchesKeyPointIndices.data(), d_globMatchesKeyPointIndices, sizeof(uint2)*m_globNumResiduals, cudaMemcpyDeviceToHost));

	int validOpt;
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(&validOpt, d_validOpt, sizeof(int), cudaMemcpyDeviceToHost));

	const unsigned int numImages = getNumImages(); 
	out.write((char*)&m_maxNumImages, sizeof(unsigned int));
	out.write((char*)&m_maxKeyPointsPerImage, sizeof(unsigned int));
	out.write((char*)&numImages, sizeof(unsigned int));
	out.write((char*)&m_numKeyPoints, sizeof(unsigned int));
	out.write((char*)m_numKeyPointsPerImage.data(), sizeof(unsigned int)*m_numKeyPointsPerImage.size());
	out.write((char*)m_numKeyPointsPerImagePrefixSum.data(), sizeof(unsigned int)*m_numKeyPointsPerImagePrefixSum.size());

	out.write((char*)keyPoints.data(), sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage);
	out.write((char*)keyPointDescs.data(), sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage);
	out.write((char*)keyPointCounters.data(), sizeof(int)*m_maxNumImages);

	out.write((char*)currNumMatchesPerImagePair.data(), sizeof(int)*maxImageMatches);
	out.write((char*)currMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
	out.write((char*)currMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

	out.write((char*)currNumFilteredMatchesPerImagePair.data(), sizeof(int)*maxImageMatches);
	out.write((char*)currFilteredMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
	out.write((char*)currFilteredMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
	out.write((char*)currFilteredTransforms.data(), sizeof(float4x4)*maxImageMatches);
	out.write((char*)currFilteredTransformsInv.data(), sizeof(float4x4)*maxImageMatches);

	out.write((char*)m_validImages.data(), sizeof(int)*m_maxNumImages);

	out.write((char*)&m_globNumResiduals, sizeof(unsigned int));
	if (m_globNumResiduals) {
		out.write((char*)globMatches.data(), sizeof(EntryJ)*m_globNumResiduals);
		out.write((char*)globMatchesKeyPointIndices.data(), sizeof(uint2)*m_globNumResiduals);
	}
	out.write((char*)&validOpt, sizeof(unsigned int));

	out.close();
}

void SIFTImageManager::loadFromFile(const std::string& s)
{
	free();
	alloc();

	std::ifstream in(s, std::ios::binary);
	if (!in.is_open()) {
		std::cout << "Error opening " << s << " for read" << std::endl;
		return;
	}

	in.read((char*)&m_maxNumImages, sizeof(unsigned int));
	in.read((char*)&m_maxKeyPointsPerImage, sizeof(unsigned int));

	unsigned int numImages = 0;
	in.read((char*)&numImages, sizeof(unsigned int));
	m_SIFTImagesGPU.resize(numImages);
	m_numKeyPointsPerImage.resize(numImages);
	m_numKeyPointsPerImagePrefixSum.resize(numImages);

	in.read((char*)&m_numKeyPoints, sizeof(unsigned int));
	in.read((char*)m_numKeyPointsPerImage.data(), sizeof(unsigned int)*m_numKeyPointsPerImage.size());
	in.read((char*)m_numKeyPointsPerImagePrefixSum.data(), sizeof(unsigned int)*m_numKeyPointsPerImagePrefixSum.size());

	{
		std::vector<SIFTKeyPoint> keyPoints(m_maxNumImages*m_maxKeyPointsPerImage);
		std::vector<SIFTKeyPointDesc> keyPointDescs(m_maxNumImages*m_maxKeyPointsPerImage);
		std::vector<int> keyPointCounters(m_maxNumImages);

		in.read((char*)keyPoints.data(), sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage);
		in.read((char*)keyPointDescs.data(), sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage);
		in.read((char*)keyPointCounters.data(), sizeof(int)*m_maxNumImages);

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_keyPoints, keyPoints.data(), sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_keyPointDescs, keyPointDescs.data(), sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage, cudaMemcpyHostToDevice));
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_keyPointCounters, keyPointCounters.data(), sizeof(int)*m_maxNumImages, cudaMemcpyHostToDevice));
	}

	{
		const unsigned maxImageMatches = m_maxNumImages;
		std::vector<int> currNumMatchesPerImagePair(maxImageMatches);
		std::vector<float> currMatchDistances(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
		std::vector<uint2> currMatchKeyPointIndices(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

		in.read((char*)currNumMatchesPerImagePair.data(), sizeof(int)*maxImageMatches);
		in.read((char*)currMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
		in.read((char*)currMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currNumMatchesPerImagePair, currNumMatchesPerImagePair.data(), sizeof(int)*maxImageMatches, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currMatchDistances, currMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currMatchKeyPointIndices, currMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyHostToDevice));

		std::vector<int> currNumFilteredMatchesPerImagePair(maxImageMatches);
		std::vector<float> currFilteredMatchDistances(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		std::vector<uint2> currFilteredMatchKeyPointIndices(maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		std::vector<float4x4> currFilteredTransforms(maxImageMatches);
		std::vector<float4x4> currFilteredTransformsInv(maxImageMatches);

		in.read((char*)currNumFilteredMatchesPerImagePair.data(), sizeof(int)*maxImageMatches);
		in.read((char*)currFilteredMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		in.read((char*)currFilteredMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		in.read((char*)currFilteredTransforms.data(), sizeof(float4x4)*maxImageMatches);
		in.read((char*)currFilteredTransformsInv.data(), sizeof(float4x4)*maxImageMatches);

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currNumFilteredMatchesPerImagePair, currNumFilteredMatchesPerImagePair.data(), sizeof(int)*maxImageMatches, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currFilteredMatchDistances, currFilteredMatchDistances.data(), sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currFilteredMatchKeyPointIndices, currFilteredMatchKeyPointIndices.data(), sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currFilteredTransforms, currFilteredTransforms.data(), sizeof(float4x4)*maxImageMatches, cudaMemcpyHostToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currFilteredTransformsInv, currFilteredTransformsInv.data(), sizeof(float4x4)*maxImageMatches, cudaMemcpyHostToDevice));

		m_validImages.resize(m_maxNumImages, 0);
		in.read((char*)m_validImages.data(), sizeof(int) * m_maxNumImages);
	}

	{
		in.read((char*)&m_globNumResiduals, sizeof(unsigned int));
		if (m_globNumResiduals) {
			std::vector<EntryJ> globMatches(m_globNumResiduals);
			std::vector<uint2> globMatchesKeyPointIndices(m_globNumResiduals);
			in.read((char*)globMatches.data(), sizeof(EntryJ)*m_globNumResiduals);
			in.read((char*)globMatchesKeyPointIndices.data(), sizeof(uint2)*m_globNumResiduals);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globNumResiduals, &m_globNumResiduals, sizeof(unsigned int), cudaMemcpyHostToDevice))
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globMatches, globMatches.data(), sizeof(EntryJ)*m_globNumResiduals, cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globMatchesKeyPointIndices, globMatchesKeyPointIndices.data(), sizeof(uint2)*m_globNumResiduals, cudaMemcpyHostToDevice));
		}

		int validOpt;
		in.read((char*)&validOpt, sizeof(unsigned int));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_validOpt, &validOpt, sizeof(int), cudaMemcpyHostToDevice));
	}

	for (unsigned int i = 0; i < numImages; i++) {
		m_SIFTImagesGPU[i].d_keyPoints = d_keyPoints + m_numKeyPointsPerImagePrefixSum[i];
		m_SIFTImagesGPU[i].d_keyPointDescs = d_keyPointDescs + m_numKeyPointsPerImagePrefixSum[i];
		//m_SIFTImagesGPU[i].d_keyPointCounter = d_keyPointCounters + i;
	}

	assert(getNumImages() == m_numKeyPointsPerImage.size());
	assert(getNumImages() == m_numKeyPointsPerImagePrefixSum.size());

	in.close();
}

void SIFTImageManager::alloc()
{
	m_numKeyPoints = 0;

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_keyPoints, sizeof(SIFTKeyPoint)*m_maxNumImages*m_maxKeyPointsPerImage));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_keyPointDescs, sizeof(SIFTKeyPointDesc)*m_maxNumImages*m_maxKeyPointsPerImage));
	//MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_keyPointCounters, sizeof(int)*m_maxNumImages));
	//MLIB_CUDA_SAFE_CALL(cudaMemset(d_keyPointCounters, 0, sizeof(int)*m_maxNumImages));

	// matching
	m_currImagePairMatches.resize(m_maxNumImages);

	const unsigned maxImageMatches = m_maxNumImages;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currNumMatchesPerImagePair, sizeof(int)*maxImageMatches));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currNumFilteredMatchesPerImagePair, sizeof(int)*maxImageMatches));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currFilteredMatchDistances, sizeof(float)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currFilteredMatchKeyPointIndices, sizeof(uint2)*maxImageMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currFilteredTransforms, sizeof(float4x4)*maxImageMatches));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currFilteredTransformsInv, sizeof(float4x4)*maxImageMatches));

	m_validImages.resize(m_maxNumImages, 0);
	m_validImages[0] = 1; // first is valid
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_validImages, sizeof(int) *  m_maxNumImages));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int), cudaMemcpyHostToDevice)); // first is valid

	const unsigned int maxResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (m_maxNumImages*(m_maxNumImages - 1)) / 2;
	m_globNumResiduals = 0;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_globNumResiduals, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_globNumResiduals, 0, sizeof(int)));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_globMatches, sizeof(EntryJ)*maxResiduals));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_globMatchesKeyPointIndices, sizeof(uint2)*maxResiduals));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_validOpt, sizeof(int)));

	initializeMatching();
}

void SIFTImageManager::free()
{
	m_SIFTImagesGPU.clear();

	m_numKeyPoints = 0;
	m_numKeyPointsPerImage.clear();
	m_numKeyPointsPerImagePrefixSum.clear();

	MLIB_CUDA_SAFE_FREE(d_keyPoints);
	MLIB_CUDA_SAFE_FREE(d_keyPointDescs);
	//MLIB_CUDA_SAFE_FREE(cudaFree(d_keyPointCounters));

	m_currImagePairMatches.clear();

	MLIB_CUDA_SAFE_FREE(d_currNumMatchesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_currMatchDistances);
	MLIB_CUDA_SAFE_FREE(d_currMatchKeyPointIndices);

	MLIB_CUDA_SAFE_FREE(d_currNumFilteredMatchesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_currFilteredMatchDistances);
	MLIB_CUDA_SAFE_FREE(d_currFilteredMatchKeyPointIndices);
	MLIB_CUDA_SAFE_FREE(d_currFilteredTransforms);
	MLIB_CUDA_SAFE_FREE(d_currFilteredTransformsInv);

	m_validImages.clear();
	MLIB_CUDA_SAFE_FREE(d_validImages);

	m_globNumResiduals = 0;
	MLIB_CUDA_SAFE_FREE(d_globNumResiduals);
	MLIB_CUDA_SAFE_FREE(d_globMatches);
	MLIB_CUDA_SAFE_FREE(d_globMatchesKeyPointIndices);

	MLIB_CUDA_SAFE_FREE(d_validOpt);

	//MLIB_CUDA_SAFE_FREE(d_fuseGlobalKeyCount);
	//MLIB_CUDA_SAFE_FREE(d_fuseGlobalKeyMarker);

	m_imagesToRetry.clear();

	SAFE_DELETE(m_timer);
}

void SIFTImageManager::initializeMatching()
{
	for (unsigned int r = 0; r < m_maxNumImages; r++) {
		ImagePairMatch& imagePairMatch = m_currImagePairMatches[r];
		imagePairMatch.d_numMatches = d_currNumMatchesPerImagePair + r;
		imagePairMatch.d_distances = d_currMatchDistances + r * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
		imagePairMatch.d_keyPointIndices = d_currMatchKeyPointIndices + r * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	}
}

void findTrack(const std::vector< std::vector<std::pair<uint2, float3>> >& corrPerKey, std::vector<bool>& marker,
	std::vector<std::pair<uint2, float3>>& track, unsigned int curKey)
{
	for (unsigned int i = 0; i < corrPerKey[curKey].size(); i++) {
		const auto& c = corrPerKey[curKey][i];
		if (!marker[c.first.y]) { //not assigned to a track already
			track.push_back(c);
			marker[c.first.y] = true;
			findTrack(corrPerKey, marker, track, c.first.y);
		}
	}
}

#define MAX_TRACK_CORR_ERROR 0.03f
void SIFTImageManager::computeTracks(const std::vector<float4x4>& trajectory, const std::vector<EntryJ>& correspondences, const std::vector<uint2>& correspondenceKeyIndices,
	std::vector< std::vector<std::pair<uint2, float3>> >& tracks) const {
	tracks.clear();
	const unsigned int numImages = getNumImages();
	std::vector< std::vector<std::pair<uint2, float3>> > corrPerKey(m_numKeyPoints); //(image,keyIndex)
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid()) {
			const uint2& keyIndices = correspondenceKeyIndices[i];
			float err = length(trajectory[corr.imgIdx_i] * corr.pos_i - trajectory[corr.imgIdx_j] * corr.pos_j);
			if (err < MAX_TRACK_CORR_ERROR) {
				corrPerKey[keyIndices.x].push_back(std::make_pair(make_uint2(corr.imgIdx_j, keyIndices.y), corr.pos_j));
				corrPerKey[keyIndices.y].push_back(std::make_pair(make_uint2(corr.imgIdx_i, keyIndices.x), corr.pos_i));
			}
			else {
				corrPerKey[keyIndices.x].push_back(std::make_pair(make_uint2(corr.imgIdx_j, keyIndices.y), make_float3(-std::numeric_limits<float>::infinity())));
				corrPerKey[keyIndices.y].push_back(std::make_pair(make_uint2(corr.imgIdx_i, keyIndices.x), make_float3(-std::numeric_limits<float>::infinity())));
			}
		}
	}

	std::vector<bool> marker(m_numKeyPoints, false);
	for (unsigned int i = 0; i < numImages; i++) {
		for (unsigned int k = 0; k < m_numKeyPointsPerImage[i]; k++) {
			//std::vector<std::pair<uint2, float3>> track; track.reserve(numImages);
			//findTrack(corrPerKey, marker, track, m_numKeyPointsPerImagePrefixSum[i] + k);
			//if (!track.empty()) tracks.push_back(track);
			if (tracks.empty() || !tracks.back().empty()) tracks.push_back(std::vector<std::pair<uint2, float3>>()); //TODO how is this different?
			findTrack(corrPerKey, marker, tracks.back(), m_numKeyPointsPerImagePrefixSum[i] + k);
		}
	}
}

void SIFTImageManager::fuseToGlobal(SIFTImageManager* global, const float4x4& colorIntrinsics, const float4x4* d_transforms,
	const float4x4& colorIntrinsicsInv) const
{
	MLIB_ASSERT(m_globNumResiduals > 0);
	std::vector<EntryJ> correspondences(m_globNumResiduals);
	cutilSafeCall(cudaMemcpy(correspondences.data(), d_globMatches, sizeof(EntryJ) * m_globNumResiduals, cudaMemcpyDeviceToHost));
	std::vector<uint2> correspondenceKeyIndices(m_globNumResiduals);
	cutilSafeCall(cudaMemcpy(correspondenceKeyIndices.data(), d_globMatchesKeyPointIndices, sizeof(uint2) * m_globNumResiduals, cudaMemcpyDeviceToHost));
	std::vector<float4x4> transforms(getNumImages());
	cutilSafeCall(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*transforms.size(), cudaMemcpyDeviceToHost));

	std::vector<SIFTKeyPoint> allKeys;
	getSIFTKeyPointsDEBUG(allKeys);
	std::vector<SIFTKeyPointDesc> allDesc;
	getSIFTKeyPointDescsDEBUG(allDesc);

	std::vector<SIFTKeyPoint> curKeys;	std::vector<SIFTKeyPointDesc> curDesc;

	std::vector< std::vector<std::pair<uint2, float3>> > tracks;
	computeTracks(transforms, correspondences, correspondenceKeyIndices, tracks);
	for (unsigned int t = 0; t < tracks.size(); t++) {
		if (tracks[t].empty()) continue;
		const auto& repKey = tracks[t].front(); //arbitrarily pick a key for the descriptor
		float3 pos = make_float3(0.0f); unsigned int num = 0;
		for (unsigned int i = 0; i < tracks[t].size(); i++) {
			if (tracks[t][i].second.x != -std::numeric_limits<float>::infinity()) {
				pos += transforms[tracks[t][i].first.x] * tracks[t][i].second;
				num++;
			}
		}
		if (num > 0) {
			pos /= (float)num; //average track locations
			//project to first frame
			pos = colorIntrinsics * pos;
			float2 loc = make_float2(pos.x / pos.z, pos.y / pos.z);
			SIFTKeyPoint key;
			key.pos = loc;
			key.scale = allKeys[repKey.first.y].scale;
			key.depth = pos.z;
			curKeys.push_back(key);
			// desc
			curDesc.push_back(allDesc[repKey.first.y]);
		}
	} // track

	unsigned int numKeys = std::min((unsigned int)curKeys.size(), m_maxKeyPointsPerImage);
	if (curKeys.size() > m_maxKeyPointsPerImage) {
		std::sort(curKeys.begin(), curKeys.end(), [](const SIFTKeyPoint& left, const SIFTKeyPoint& right) {
			return left.depth < right.depth;
		});
	}
	SIFTImageGPU& cur = global->createSIFTImageGPU();
	cutilSafeCall(cudaMemcpy(cur.d_keyPoints, curKeys.data(), sizeof(SIFTKeyPoint) * numKeys, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(cur.d_keyPointDescs, curDesc.data(), sizeof(SIFTKeyPointDesc) * numKeys, cudaMemcpyHostToDevice));
	global->finalizeSIFTImageGPU(numKeys);
	//SIFTImageGPU& cur = global->createSIFTImageGPU();	//FOR DEBUGGING ONLY (just use first frame's keys)
	//unsigned int numKeys = m_numKeyPointsPerImage[0];
	//if (numKeys > 0) {
	//	cutilSafeCall(cudaMemcpy(cur.d_keyPoints, d_keyPoints, sizeof(SIFTKeyPoint) * numKeys, cudaMemcpyDeviceToDevice));
	//	cutilSafeCall(cudaMemcpy(cur.d_keyPointDescs, d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeys, cudaMemcpyDeviceToDevice));
	//}
	//global->finalizeSIFTImageGPU(numKeys);
}
//void SIFTImageManager::fuseToGlobal(SIFTImageManager* global, const float4x4& colorIntrinsics, const float4x4* d_transforms,
//	const float4x4& colorIntrinsicsInv) const
//{
//	//const unsigned int overlapImageIndex = getNumImages() - 1; // overlap frame
//
//	std::vector<EntryJ> correspondences(m_globNumResiduals);
//	cutilSafeCall(cudaMemcpy(correspondences.data(), d_globMatches, sizeof(EntryJ) * m_globNumResiduals, cudaMemcpyDeviceToHost));
//	std::vector<uint2> correspondenceKeyIndices(m_globNumResiduals);
//	cutilSafeCall(cudaMemcpy(correspondenceKeyIndices.data(), d_globMatchesKeyPointIndices, sizeof(uint2) * m_globNumResiduals, cudaMemcpyDeviceToHost));
//	std::vector<float4x4> transforms(getNumImages());
//	cutilSafeCall(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*transforms.size(), cudaMemcpyDeviceToHost));
//
//	std::vector<SIFTKeyPoint> allKeys;
//	getSIFTKeyPointsDEBUG(allKeys);
//	std::vector<SIFTKeyPointDesc> allDesc;
//	getSIFTKeyPointDescsDEBUG(allDesc);
//	std::vector<bool> keyMarker(allKeys.size(), false);
//
//	const unsigned int widthSIFT = GlobalBundlingState::get().s_widthSIFT;
//	const unsigned int heightSIFT = GlobalBundlingState::get().s_heightSIFT;
//	const unsigned int padding = 3;
//	const unsigned int pixelDistThresh = 3;
//	std::vector<bool> imageMarker(padding * widthSIFT * padding * heightSIFT, false);
//
//	std::vector<SIFTKeyPoint> curKeys;
//	std::vector<SIFTKeyPointDesc> curDesc;
//
//	for (unsigned int i = 0; i < m_globNumResiduals; i++) {
//		const EntryJ& corr = correspondences[i];
//		if (corr.isValid()) {
//			const uint2& keyIndices = correspondenceKeyIndices[i];
//			uint2 k0 = make_uint2(corr.imgIdx_i, keyIndices.x);
//			uint2 k1 = make_uint2(corr.imgIdx_j, keyIndices.y);
//
//			if (!keyMarker[k0.y] && curKeys.size() < m_maxKeyPointsPerImage) {
//				float3 pos = ((transforms[k0.x] * corr.pos_i) + (transforms[k1.x] * corr.pos_j)) / 2.0f; // average locations in world space
//				//float3 pos = transforms[k0.x] * corr.pos_i; // just pick one
//				// project to first frame
//				pos = colorIntrinsics * pos;
//				float2 loc = make_float2(pos.x / pos.z, pos.y / pos.z);
//
//				int2 pixLocDiscretized = make_int2((int)round((loc.x + widthSIFT) / (float)pixelDistThresh), (int)round((loc.y + heightSIFT) / (float)pixelDistThresh));
//				int linIdx = pixLocDiscretized.y * (widthSIFT * padding) + pixLocDiscretized.x;
//				if (pixLocDiscretized.x >= 0 && pixLocDiscretized.y >= 0 && linIdx >= 0 && linIdx < imageMarker.size() && !imageMarker[linIdx]) {
//
//					SIFTKeyPoint key;
//					key.pos = loc;
//					key.scale = allKeys[k0.y].scale;
//					key.depth = pos.z;
//					curKeys.push_back(key);
//					// desc
//					curDesc.push_back(allDesc[k0.y]);
//					keyMarker[k0.y] = true;
//
//					imageMarker[linIdx] = true;
//				}
//			} // not already found
//		}// valid corr
//	} // correspondences/residual
//
//	//std::vector<float4x4> transformsInv(transforms.size());
//	//for (unsigned int i = 0; i < transforms.size(); i++) {
//	//	transformsInv[i] = transforms[i].getInverse();
//	//}
//	//fuseLocalKeyDepths(curKeys, depthFrames, depthWidth, depthHeight,
//	//	transforms, transformsInv, colorIntrinsicsInv, depthIntrinsics, depthIntrinsicsInv);
//
//	unsigned int numKeys = (unsigned int)curKeys.size();
//	SIFTImageGPU& cur = global->createSIFTImageGPU();
//	cutilSafeCall(cudaMemcpy(cur.d_keyPoints, curKeys.data(), sizeof(SIFTKeyPoint) * numKeys, cudaMemcpyHostToDevice));
//	cutilSafeCall(cudaMemcpy(cur.d_keyPointDescs, curDesc.data(), sizeof(SIFTKeyPointDesc) * numKeys, cudaMemcpyHostToDevice));
//	global->finalizeSIFTImageGPU(numKeys);
//}

unsigned int SIFTImageManager::filterFrames(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames)
{
	if (numFrames == 0) return (unsigned int)-1;

	int connected = 0;
	unsigned int lastMatchedFrame = (unsigned int)-1;

	std::vector<unsigned int> currNumFilteredMatchesPerImagePair(numFrames - startFrame);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(currNumFilteredMatchesPerImagePair.data(), d_currNumFilteredMatchesPerImagePair + startFrame, sizeof(unsigned int) * currNumFilteredMatchesPerImagePair.size(), cudaMemcpyDeviceToHost));

	for (int i = (int)numFrames-1; i >= (int)startFrame; i--) { // previous frames (reverse direction) 
		if (m_validImages[i] != 0 && currNumFilteredMatchesPerImagePair[i - startFrame] > 0 && i != curFrame) {
			connected = 1;
			lastMatchedFrame = (unsigned int)i;
			break;
		}
	}

	//if (GlobalBundlingState::get().s_verbose && !connected)
	//	std::cout << "frame " << curFrame << " not connected to previous!" << std::endl;

	m_validImages[curFrame] = connected;
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_validImages + curFrame, &m_validImages[curFrame], sizeof(int), cudaMemcpyHostToDevice));
	return lastMatchedFrame;
}

//void SIFTImageManager::fuseLocalKeyDepths(std::vector<SIFTKeyPoint>& globalKeys, const std::vector<CUDACachedFrame>& cachedFrames,
void SIFTImageManager::fuseLocalKeyDepths(std::vector<SIFTKeyPoint>& globalKeys, const std::vector<float*>& depthFrames,
	unsigned int depthWidth, unsigned int depthHeight,
	const std::vector<float4x4>& transforms, const std::vector<float4x4>& transformsInv,
	const float4x4& siftIntrinsicsInv, const float4x4& depthIntrinsics, const float4x4& depthIntrinsicsInv) const
{
	const float depthDiffThresh = 0.05f;

	// copy depth frames to CPU
	std::vector<DepthImage32> depthImages(getNumImages());
	for (unsigned int i = 0; i < depthImages.size(); i++) {
		depthImages[i].allocate(depthWidth, depthHeight);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthImages[i].getData(), cachedFrames[i].d_depthDownsampled, sizeof(float) * depthImages[i].getNumPixels(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthImages[i].getData(), depthFrames[i], sizeof(float) * depthImages[i].getNumPixels(), cudaMemcpyDeviceToHost));
	}

	for (unsigned int i = 0; i < globalKeys.size(); i++) {
		SIFTKeyPoint& key = globalKeys[i];
		// camera space
		float3 keyCamPos = siftIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));

		float sumDepth = key.depth; // first frame
		unsigned int numDepth = 1;

		// project to all frames, then to first frame
		for (unsigned int j = 1; j < depthImages.size(); j++) {
			float3 curCamPos = transformsInv[j] * keyCamPos;
			float3 p = depthIntrinsics * curCamPos; // first frame to cam pos of frame j
			float2 loc = make_float2(p.x / p.z, p.y / p.z);
			int2 iloc = make_int2((int)round(loc.x), (int)round(loc.y));
			if (iloc.x >= 0 && iloc.y >= 0 && iloc.x < (int)depthWidth && iloc.y < (int)depthHeight) {
				float d = depthImages[j](iloc.x, iloc.y);
				if (d != -std::numeric_limits<float>::infinity() && fabs(d - curCamPos.z) < depthDiffThresh) { // project to first frame
					float3 firstCamPos = transforms[j] * (depthIntrinsicsInv * (d * make_float3(loc.x, loc.y, 1.0f)));
					sumDepth += firstCamPos.z;
					numDepth++;
				}
			}
		}
		//float ff = sumDepth / numDepth;
		//if (fabs(key.depth - ff) > 0.03f) {
		//	std::cout << "warning: original depth " << key.depth << ", new fused depth " << ff << std::endl;
		//	getchar();
		//}

		// compute average
		key.depth = sumDepth / numDepth;
	}
}


