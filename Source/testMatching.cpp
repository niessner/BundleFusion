
#include "stdafx.h"
#include "SIFTImageManager.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/SiftMatchFilter.h"
#include "SiftGPU/MatrixConversion.h"
#include "ImageHelper.h"

#include "testMatching.h"

TestMatching::TestMatching()
{
	unsigned maxNumImages = GlobalBundlingState::get().s_maxNumImages;
	m_siftManager = new SIFTImageManager(GlobalBundlingState::get().s_submapSize, maxNumImages, GlobalBundlingState::get().s_maxNumKeysPerImage);

	unsigned int maxNumMatches = (maxNumImages - 1) * maxNumImages / 2;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_numMatchesPerImagePair, sizeof(int)*maxNumMatches));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_matchDistancesPerImagePair, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_matchKeyIndicesPerImagePair, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_filtNumMatchesPerImagePair, sizeof(int)*maxNumMatches));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_filtMatchDistancesPerImagePair, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_filtMatchKeyIndicesPerImagePair, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));

	//!!!DEBUGGING
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_numMatchesPerImagePair, -1, sizeof(int)*maxNumMatches));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_matchDistancesPerImagePair, -1, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_matchKeyIndicesPerImagePair, -1, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_RAW));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_filtNumMatchesPerImagePair, -1, sizeof(int)*maxNumMatches));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_filtMatchDistancesPerImagePair, -1, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_filtMatchKeyIndicesPerImagePair, -1, sizeof(int)*maxNumMatches*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED));
	//!!!DEBUGGING

	m_bHasMatches = false;
	m_siftIntrinsics.setZero();
	m_siftIntrinsicsInv.setZero();
}

TestMatching::~TestMatching()
{
	SAFE_DELETE(m_siftManager);

	MLIB_CUDA_SAFE_FREE(d_numMatchesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_matchDistancesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_matchKeyIndicesPerImagePair);

	MLIB_CUDA_SAFE_FREE(d_filtNumMatchesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_filtMatchDistancesPerImagePair);
	MLIB_CUDA_SAFE_FREE(d_filtMatchKeyIndicesPerImagePair);
}

void TestMatching::load(const std::string& filename, const std::string siftFile)
{
	if (!util::fileExists(siftFile)) throw MLIB_EXCEPTION(siftFile + " does not exist");
	m_siftManager->loadFromFile(siftFile);

	if (filename.empty() || !util::fileExists(filename)) {
		std::cout << "warning: " << filename << " does not exist, need to re-compute" << std::endl;
		return;
	}
	BinaryDataStreamFile s(filename, false);
	unsigned int numImages;
	s >> numImages;
	s >> m_siftIntrinsics;
	s >> m_siftIntrinsicsInv;
	s >> m_origMatches;
	for (unsigned int i = 1; i < numImages; i++) { // no matches for image 0
		{
			const bool filtered = false;
			std::vector<unsigned int> numMatchesPerImagePair;
			std::vector<float> matchDistancesPerImagePair;
			std::vector<vec2ui> matchKeyIndicesPerImagePair;
			s >> numMatchesPerImagePair;
			s >> matchDistancesPerImagePair;
			s >> matchKeyIndicesPerImagePair;
			MLIB_ASSERT(numMatchesPerImagePair.size() == i && matchDistancesPerImagePair.size() == i*MAX_MATCHES_PER_IMAGE_PAIR_RAW && matchKeyIndicesPerImagePair.size() == i*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getNumMatchesCUDA(i, filtered), numMatchesPerImagePair.data(), sizeof(int)*numMatchesPerImagePair.size(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchDistsCUDA(i, filtered), matchDistancesPerImagePair.data(), sizeof(float)*matchDistancesPerImagePair.size(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchKeyIndicesCUDA(i, filtered), matchKeyIndicesPerImagePair.data(), sizeof(uint2)*matchKeyIndicesPerImagePair.size(), cudaMemcpyHostToDevice));
		}
		{
			const bool filtered = true;
			std::vector<unsigned int> numMatchesPerImagePair;
			std::vector<float> matchDistancesPerImagePair;
			std::vector<vec2ui> matchKeyIndicesPerImagePair;
			s >> numMatchesPerImagePair;
			s >> matchDistancesPerImagePair;
			s >> matchKeyIndicesPerImagePair;
			MLIB_ASSERT(numMatchesPerImagePair.size() == i && matchDistancesPerImagePair.size() == i*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED && matchKeyIndicesPerImagePair.size() == i*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getNumMatchesCUDA(i, filtered), numMatchesPerImagePair.data(), sizeof(int)*numMatchesPerImagePair.size(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchDistsCUDA(i, filtered), matchDistancesPerImagePair.data(), sizeof(float)*matchDistancesPerImagePair.size(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchKeyIndicesCUDA(i, filtered), matchKeyIndicesPerImagePair.data(), sizeof(uint2)*matchKeyIndicesPerImagePair.size(), cudaMemcpyHostToDevice));
		}
	}
	s.closeStream();

	m_bHasMatches = true;
}

void TestMatching::save(const std::string& filename) const
{
	BinaryDataStreamFile s(filename, true);
	unsigned int numImages = m_siftManager->getNumImages();
	s << numImages;
	s << m_siftIntrinsics;
	s << m_siftIntrinsicsInv;
	s << m_origMatches;
	for (unsigned int i = 1; i < numImages; i++) { // no matches for image 0
		{
			const bool filtered = false;
			std::vector<unsigned int> numMatchesPerImagePair(i);
			std::vector<float> matchDistancesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
			std::vector<vec2ui> matchKeyIndicesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatchesPerImagePair.data(), getNumMatchesCUDA(i, filtered), sizeof(int)*numMatchesPerImagePair.size(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistancesPerImagePair.data(), getMatchDistsCUDA(i, filtered), sizeof(float)*matchDistancesPerImagePair.size(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndicesPerImagePair.data(), getMatchKeyIndicesCUDA(i, filtered), sizeof(uint2)*matchKeyIndicesPerImagePair.size(), cudaMemcpyDeviceToHost));

			s << numMatchesPerImagePair;
			s << matchDistancesPerImagePair;
			s << matchKeyIndicesPerImagePair;
		}
		{
			const bool filtered = true;
			std::vector<unsigned int> numMatchesPerImagePair(i);
			std::vector<float> matchDistancesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
			std::vector<vec2ui> matchKeyIndicesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatchesPerImagePair.data(), getNumMatchesCUDA(i, filtered), sizeof(int)*numMatchesPerImagePair.size(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistancesPerImagePair.data(), getMatchDistsCUDA(i, filtered), sizeof(float)*matchDistancesPerImagePair.size(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndicesPerImagePair.data(), getMatchKeyIndicesCUDA(i, filtered), sizeof(uint2)*matchKeyIndicesPerImagePair.size(), cudaMemcpyDeviceToHost));

			s << numMatchesPerImagePair;
			s << matchDistancesPerImagePair;
			s << matchKeyIndicesPerImagePair;
		}
	}
	s.closeStream();
}

void TestMatching::test()
{
	if (!m_bHasMatches) {
		matchAll();
		save("debug/matchAll.bin");
	}
	//!!!DEBUGGING
	//std::vector<vec2ui> matches = { vec2ui(0, 2) };
	//printMatches("debug/", matches, false);
	//!!!DEBUGGING

	unsigned int numImages = m_siftManager->getNumImages();

	std::vector<vec2ui> imagePairMatchesFiltered;
	filter(imagePairMatchesFiltered);

	//!!!DEBUGGING
	//saveMatchToPLY("debug/", vec2ui(118, 135), true);
	//!!!DEBUGGING

	// compare to reference
	std::vector<vec2ui> falsePositives, falseNegatives;
	checkFiltered(imagePairMatchesFiltered, falsePositives, falseNegatives);

	std::cout << "#false positives = " << falsePositives.size() << std::endl;
	std::cout << "#false negatives = " << falseNegatives.size() << std::endl;

	// visualize
	printMatches("debug/falsePositives/", falsePositives, true);
	printMatches("debug/falseNegatives/", falseNegatives, false);
}

void TestMatching::matchAll()
{
	SiftMatchGPU* siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	siftMatcher->InitSiftMatch();
	const float ratioMax = GlobalBundlingState::get().s_siftMatchRatioMaxGlobal;
	const float matchThresh = GlobalBundlingState::get().s_siftMatchThresh;
	unsigned int numImages = m_siftManager->getNumImages();
	const bool filtered = false;
	const unsigned int minNumMatches = GlobalBundlingState::get().s_minNumMatchesGlobal;

	m_origMatches.clear();
	std::cout << "matching all... ";
	Timer t;

	const std::vector<int>& valid = m_siftManager->getValidImages();
	for (unsigned int cur = 1; cur < numImages; cur++) {
		if (valid[cur] == 0) continue;

		SIFTImageGPU& curImage = m_siftManager->getImageGPU(cur);
		int num2 = (int)m_siftManager->getNumKeyPointsPerImage(cur);

		// match to all previous
		for (unsigned int prev = 0; prev < cur; prev++) {
			if (valid[prev] == 0) {
				unsigned int num = 0;
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_siftManager->d_currNumMatchesPerImagePair + prev, &num, sizeof(int), cudaMemcpyHostToDevice));
				continue;
			}
			SIFTImageGPU& prevImage = m_siftManager->getImageGPU(prev);
			int num1 = (int)m_siftManager->getNumKeyPointsPerImage(prev);

			uint2 keyPointOffset = make_uint2(0, 0);
			ImagePairMatch& imagePairMatch = m_siftManager->getImagePairMatchDEBUG(prev, cur, keyPointOffset);

			siftMatcher->SetDescriptors(0, num1, (unsigned char*)prevImage.d_keyPointDescs);
			siftMatcher->SetDescriptors(1, num2, (unsigned char*)curImage.d_keyPointDescs);
			siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset, matchThresh, ratioMax);
		}  // prev frames

		// save
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getNumMatchesCUDA(cur, filtered), m_siftManager->d_currNumMatchesPerImagePair, sizeof(int)*cur, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchDistsCUDA(cur, filtered), m_siftManager->d_currMatchDistances, sizeof(float)*cur*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchKeyIndicesCUDA(cur, filtered), m_siftManager->d_currMatchKeyPointIndices, sizeof(uint2)*cur*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToDevice));

		std::vector<int> numMatches(cur);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), getNumMatchesCUDA(cur, filtered), sizeof(int)*cur, cudaMemcpyDeviceToHost));
		for (unsigned int i = 0; i < numMatches.size(); i++) {
			if (numMatches[i] >(int)minNumMatches) {
				m_origMatches.push_back(vec2ui(i, cur));
			}
		}
	} // cur frames
	t.stop();
	std::cout << "done! (" << t.getElapsedTimeMS() << " ms)" << std::endl;

	SAFE_DELETE(siftMatcher);
}

void TestMatching::filter(std::vector<vec2ui>& imagePairMatchesFiltered)
{
	const float maxResThresh2 = GlobalBundlingState::get().s_maxKabschResidual2;

	imagePairMatchesFiltered.clear();
	std::cout << "filtering... ";
	Timer t;

	unsigned int numImages = m_siftManager->getNumImages();
	const std::vector<int>& valid = m_siftManager->getValidImages();
	for (unsigned int cur = 1; cur < numImages; cur++) {
		if (valid[cur] == 0) continue;

		// copy respective matches
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_siftManager->d_currNumMatchesPerImagePair, getNumMatchesCUDA(cur, false), sizeof(int)*cur, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_siftManager->d_currMatchDistances, getMatchDistsCUDA(cur, false), sizeof(float)*cur*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_siftManager->d_currMatchKeyPointIndices, getMatchKeyIndicesCUDA(cur, false), sizeof(uint2)*cur*MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToDevice));

		//!!!DEBUGGING
		{
			std::vector<int> numMatches(cur);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), m_siftManager->d_currNumMatchesPerImagePair, sizeof(int)*cur, cudaMemcpyDeviceToHost));
			for (unsigned int ii = 0; ii < cur; ii++) {
				if (numMatches[ii] < 0 || numMatches[ii] > MAX_MATCHES_PER_IMAGE_PAIR_RAW) {
					std::cout << "ERROR NUM MATCHES FOR " << ii << ", " << cur << std::endl;
					int a = 5;
				}
			}
		}
		//!!!DEBUGGING

		bool debugPrint = false;//(cur == 8);
		//SIFTMatchFilter::filterKeyPointMatchesDEBUG(cur, m_siftManager, MatrixConversion::toCUDA(m_siftIntrinsicsInv), maxResThresh2, debugPrint);
		SIFTMatchFilter::ransacKeyPointMatchesDEBUG(cur, m_siftManager, MatrixConversion::toCUDA(m_siftIntrinsicsInv), maxResThresh2, debugPrint);
		if (debugPrint){
			std::cout << "debug print waiting..." << std::endl;
			getchar();
		}

		std::vector<unsigned int> numMatches(cur);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), m_siftManager->d_currNumFilteredMatchesPerImagePair, sizeof(int)*cur, cudaMemcpyDeviceToHost));
		for (unsigned int i = 0; i < cur; i++) {
			if (numMatches[i] > 0) imagePairMatchesFiltered.push_back(vec2ui(i, cur));
		}
		// to have for debug printing
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getNumMatchesCUDA(cur, true), m_siftManager->d_currNumFilteredMatchesPerImagePair, sizeof(int)*cur, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchDistsCUDA(cur, true), m_siftManager->d_currFilteredMatchDistances, sizeof(float)*cur*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(getMatchKeyIndicesCUDA(cur, true), m_siftManager->d_currFilteredMatchKeyPointIndices, sizeof(uint2)*cur*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToDevice));
	} // cur frames

	t.stop();
	std::cout << "done! (" << t.getElapsedTimeMS() << " ms)" << std::endl;
}

void TestMatching::loadIntrinsics(const std::string& filename)
{
	CalibratedSensorData cs;
	BinaryDataStreamFile s(filename, false);
	s >> cs;
	s.closeStream();

	m_siftIntrinsics = cs.m_CalibrationColor.m_Intrinsic;
	m_siftIntrinsicsInv = cs.m_CalibrationColor.m_IntrinsicInverse;

	unsigned int siftWidth = GlobalBundlingState::get().s_widthSIFT;
	unsigned int siftHeight = GlobalBundlingState::get().s_heightSIFT;
	if (cs.m_ColorImageWidth != siftWidth && cs.m_ColorImageHeight != siftHeight) {
		// adapt intrinsics
		const float scaleWidth = (float)siftWidth / (float)cs.m_ColorImageWidth;
		const float scaleHeight = (float)siftHeight / (float)cs.m_ColorImageHeight;

		m_siftIntrinsics._m00 *= scaleWidth;  m_siftIntrinsics._m02 *= scaleWidth;
		m_siftIntrinsics._m11 *= scaleHeight; m_siftIntrinsics._m12 *= scaleHeight;

		m_siftIntrinsicsInv._m00 /= scaleWidth; m_siftIntrinsicsInv._m11 /= scaleHeight;
	}
}

void TestMatching::checkFiltered(const std::vector<vec2ui>& imagePairMatchesFiltered, std::vector<vec2ui>& falsePositives, std::vector<vec2ui>& falseNegatives) const
{
	const float maxResidualThres2 = 0.05 * 0.05;

	// get data
	std::vector<SIFTKeyPoint> keys;
	keys.resize(m_siftManager->m_numKeyPoints);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(keys.data(), m_siftManager->d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	unsigned int numImages = m_siftManager->getNumImages();
	unsigned int maxNumMatches = (numImages - 1)*numImages / 2;
	std::vector<int> numMatchesRaw(maxNumMatches); std::vector<float> matchDistsRaw; std::vector<uint2> matchKeyIndicesRaw;
	std::vector<int> numMatchesFilt(maxNumMatches); std::vector<float> matchDistsFilt; std::vector<uint2> matchKeyIndicesFilt;
	unsigned int offsetRaw = MAX_MATCHES_PER_IMAGE_PAIR_RAW, offsetFilt = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED;
	// filtered
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatchesFilt.data(), d_filtNumMatchesPerImagePair, sizeof(int)*numMatchesFilt.size(), cudaMemcpyDeviceToHost));
	matchDistsFilt.resize(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxNumMatches);
	matchKeyIndicesFilt.resize(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxNumMatches);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistsFilt.data(), d_filtMatchDistancesPerImagePair, sizeof(float)*matchDistsFilt.size(), cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndicesFilt.data(), d_filtMatchKeyIndicesPerImagePair, sizeof(uint2)*matchKeyIndicesFilt.size(), cudaMemcpyDeviceToHost));
	// raw
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatchesRaw.data(), d_numMatchesPerImagePair, sizeof(int)*numMatchesRaw.size(), cudaMemcpyDeviceToHost));
	matchDistsRaw.resize(MAX_MATCHES_PER_IMAGE_PAIR_RAW * maxNumMatches);
	matchKeyIndicesRaw.resize(MAX_MATCHES_PER_IMAGE_PAIR_RAW * maxNumMatches);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistsRaw.data(), d_matchDistancesPerImagePair, sizeof(float)*matchDistsRaw.size(), cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndicesRaw.data(), d_matchKeyIndicesPerImagePair, sizeof(uint2)*matchKeyIndicesRaw.size(), cudaMemcpyDeviceToHost));

	// compare to reference trajectory
	std::unordered_set<vec2ui> filteredSet;
	for (unsigned int i = 0; i < imagePairMatchesFiltered.size(); i++) filteredSet.insert(imagePairMatchesFiltered[i]);

	for (unsigned int i = 0; i < m_origMatches.size(); i++) {
		const vec2ui& imageIndices = m_origMatches[i];
		MLIB_ASSERT(m_referenceTrajectory[imageIndices.x][0] != -std::numeric_limits<float>::infinity() &&
			m_referenceTrajectory[imageIndices.y][0] != -std::numeric_limits<float>::infinity());
		const mat4f transform = m_referenceTrajectory[imageIndices.y].getInverse() * m_referenceTrajectory[imageIndices.x]; // src to tgt

		unsigned int idx = (imageIndices.y - 1) * imageIndices.y / 2 + imageIndices.x;
		std::vector<vec3f> srcPts(MAX_MATCHES_PER_IMAGE_PAIR_RAW), tgtPts(MAX_MATCHES_PER_IMAGE_PAIR_RAW);

		MLIB_ASSERT(numMatchesRaw[idx] > 0);
		if (filteredSet.find(imageIndices) != filteredSet.end()) { // classified good
			MLIB_ASSERT(numMatchesFilt[idx] > 0);
			getSrcAndTgtPts(keys.data(), matchKeyIndicesFilt.data() + idx*offsetFilt, numMatchesFilt[idx],
				(float3*)srcPts.data(), (float3*)tgtPts.data(), MatrixConversion::toCUDA(m_siftIntrinsicsInv));
			float maxRes = 0.0f;
			for (int i = 0; i < numMatchesFilt[idx]; i++) {
				vec3f d = transform * srcPts[i] - tgtPts[i];
				float res = d | d;
				if (res > maxRes) maxRes = res;
			}
			if (maxRes > maxResidualThres2) { // bad
				//!!!DEBUGGING
				float m = std::sqrt(maxRes);
				//!!!DEBUGGING
				falsePositives.push_back(imageIndices);
			}
		}
		else { // filtered out
			getSrcAndTgtPts(keys.data(), matchKeyIndicesRaw.data() + idx*offsetRaw, numMatchesRaw[idx],
				(float3*)srcPts.data(), (float3*)tgtPts.data(), MatrixConversion::toCUDA(m_siftIntrinsicsInv));
			float maxRes = 0.0f; const unsigned int minNumMatches = GlobalBundlingState::get().s_minNumMatchesGlobal;
			for (unsigned int i = 0; i < minNumMatches; i++) {//numMatchesRaw[idx]; i++) {
				vec3f d = transform * srcPts[i] - tgtPts[i];
				float res = d | d;
				if (res > maxRes) maxRes = res;
			}
			if (maxRes <= maxResidualThres2) { // good
				//!!!DEBUGGING
				float m = std::sqrt(maxRes);
				//!!!DEBUGGING
				falseNegatives.push_back(imageIndices);
			}
		}
	}
}

void TestMatching::printMatches(const std::string& outDir, const std::vector<vec2ui>& imagePairMatches, bool filtered) const
{
	if (!util::directoryExists(outDir)) util::makeDirectory(outDir);

	// get data
	std::vector<SIFTKeyPoint> keys;
	keys.resize(m_siftManager->m_numKeyPoints);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(keys.data(), m_siftManager->d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	const float distMax = GlobalBundlingState::get().s_siftMatchThresh;

	unsigned int numImages = m_siftManager->getNumImages();
	unsigned int maxNumMatches = (numImages - 1)*numImages / 2;
	std::vector<int> numMatches(maxNumMatches);
	std::vector<float> matchDists;
	std::vector<uint2> matchKeyIndices;
	unsigned int OFFSET;
	if (filtered) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_filtNumMatchesPerImagePair, sizeof(int)*numMatches.size(), cudaMemcpyDeviceToHost));
		matchDists.resize(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxNumMatches);
		matchKeyIndices.resize(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxNumMatches);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDists.data(), d_filtMatchDistancesPerImagePair, sizeof(float)*matchDists.size(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndices.data(), d_filtMatchKeyIndicesPerImagePair, sizeof(uint2)*matchKeyIndices.size(), cudaMemcpyDeviceToHost));
		OFFSET = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED;
	}
	else {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_numMatchesPerImagePair, sizeof(int)*numMatches.size(), cudaMemcpyDeviceToHost));
		matchDists.resize(MAX_MATCHES_PER_IMAGE_PAIR_RAW * maxNumMatches);
		matchKeyIndices.resize(MAX_MATCHES_PER_IMAGE_PAIR_RAW * maxNumMatches);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDists.data(), d_matchDistancesPerImagePair, sizeof(float)*matchDists.size(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndices.data(), d_matchKeyIndicesPerImagePair, sizeof(uint2)*matchKeyIndices.size(), cudaMemcpyDeviceToHost));
		OFFSET = MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	}
	for (unsigned int i = 0; i < imagePairMatches.size(); i++) {
		const vec2ui& imageIndices = imagePairMatches[i];
		const ColorImageR8G8B8A8& image1 = m_debugColorImages[imageIndices.x];
		const ColorImageR8G8B8A8& image2 = m_debugColorImages[imageIndices.y];

		unsigned int idx = (imageIndices.y - 1) * imageIndices.y / 2 + imageIndices.x;
		if (numMatches[idx] == 0) return;

		const std::string filename = outDir + std::to_string(imageIndices.x) + "-" + std::to_string(imageIndices.y) + ".png";

		ColorImageR32G32B32 matchImage(image1.getWidth() * 2, image1.getHeight());
		ColorImageR32G32B32 im1(image1);
		ColorImageR32G32B32 im2(image2);
		matchImage.copyIntoImage(im1, 0, 0);
		matchImage.copyIntoImage(im2, image1.getWidth(), 0);

		const float scaleWidth = (float)m_debugColorImages[0].getWidth() / (float)GlobalBundlingState::get().s_widthSIFT;
		const float scaleHeight = (float)m_debugColorImages[0].getHeight() / (float)GlobalBundlingState::get().s_heightSIFT;

		float maxMatchDistance = 0.0f;
		RGBColor lowColor = ml::RGBColor::Blue;
		RGBColor highColor = ml::RGBColor::Red;
		//unsigned int numMatchesToPrint = filtered ? numMatches[idx] : std::min(GlobalBundlingState::get().s_minNumMatchesGlobal, (unsigned int)numMatches[idx]);
		unsigned int numMatchesToPrint = numMatches[idx];
		for (unsigned int i = 0; i < numMatchesToPrint; i++) {
			const SIFTKeyPoint& key1 = keys[matchKeyIndices[idx*OFFSET + i].x];
			const SIFTKeyPoint& key2 = keys[matchKeyIndices[idx*OFFSET + i].y];
			if (matchDists[idx*OFFSET + i] > maxMatchDistance) maxMatchDistance = matchDists[idx*OFFSET + i];

			vec2f pf0(key1.pos.x * scaleWidth, key1.pos.y * scaleHeight);
			vec2f pf1(key2.pos.x * scaleWidth, key2.pos.y * scaleHeight);

			RGBColor c = RGBColor::interpolate(lowColor, highColor, matchDists[idx*OFFSET + i] / distMax);
			vec3f color(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
			vec2i p0 = ml::math::round(ml::vec2f(pf0.x, pf0.y));
			vec2i p1 = ml::math::round(ml::vec2f(pf1.x + image1.getWidth(), pf1.y));
			ImageHelper::drawCircle(matchImage, p0, ml::math::round(key1.scale), color);
			ImageHelper::drawCircle(matchImage, p1, ml::math::round(key2.scale), color);
			ImageHelper::drawLine(matchImage, p0, p1, color);
		}
		//std::cout << "(" << imageIndices << "): max match distance = " << maxMatchDistance << std::endl;
		FreeImageWrapper::saveImage(filename, matchImage);
	}
}

void TestMatching::loadFromSensor(const std::string& sensorFile, const std::string& trajectoryFile, unsigned int skip)
{
	std::cout << "loading color images from sensor... ";
	CalibratedSensorData cs; std::vector<mat4f> refTrajectory;
	{
		BinaryDataStreamFile s(sensorFile, false);
		s >> cs;
	}
	{
		BinaryDataStreamFile s(trajectoryFile, false);
		s >> refTrajectory;
	}
	m_debugColorImages.resize((cs.m_ColorNumFrames - 1) / skip + 1);
	m_debugDepthImages.resize(m_debugColorImages.size());
	m_referenceTrajectory.resize(m_debugColorImages.size());
	MLIB_ASSERT((cs.m_ColorNumFrames - 1) / skip == m_debugColorImages.size() - 1);
	for (unsigned int i = 0; i < cs.m_ColorNumFrames; i += skip) {
		MLIB_ASSERT(i / skip < m_debugColorImages.size());
		m_debugColorImages[i / skip] = ColorImageR8G8B8A8(cs.m_ColorImageWidth, cs.m_ColorImageHeight, cs.m_ColorImages[i]);
		m_debugDepthImages[i / skip] = DepthImage32(cs.m_DepthImageWidth, cs.m_DepthImageHeight, cs.m_DepthImages[i]);
		m_referenceTrajectory[i / skip] = refTrajectory[i];
	}
	m_depthIntrinsicsInv = cs.m_CalibrationDepth.m_IntrinsicInverse;
	std::cout << "done! (" << m_debugColorImages.size() << " of " << cs.m_ColorNumFrames << ")" << std::endl;
}

void TestMatching::saveImages(const std::string& filename) const
{
	if (m_debugColorImages.empty() || m_debugDepthImages.empty() || m_referenceTrajectory.empty()) return;
	std::cout << "saving color/depth images and trajectory... ";

	BinaryDataStreamFile s(filename, true);
	s << m_debugColorImages.size();
	for (unsigned int i = 0; i < m_debugColorImages.size(); i++) {
		s << m_debugColorImages[i];
		s << m_debugDepthImages[i];
	}
	s << m_referenceTrajectory;
	s << m_depthIntrinsicsInv;
	s.closeStream();
	std::cout << "done!" << std::endl;
}

void TestMatching::loadImages(const std::string& filename)
{
	std::cout << "loading images... ";

	BinaryDataStreamFile s(filename, false);
	size_t numImages;
	s >> numImages;
	m_debugColorImages.resize(numImages);
	m_debugDepthImages.resize(numImages);
	for (unsigned int i = 0; i < m_debugColorImages.size(); i++) {
		s >> m_debugColorImages[i];
		s >> m_debugDepthImages[i];
	}
	s >> m_referenceTrajectory;
	s >> m_depthIntrinsicsInv;
	s.closeStream();
	std::cout << "done!" << std::endl;
}

void TestMatching::saveToPointCloud(const std::string& filename, const std::vector<unsigned int>& frameIndices /*= std::vector<unsigned int>()*/) const
{
	std::cout << "computing point cloud..." << std::endl;
	std::vector<unsigned int> pointCloudFrameIndices;
	if (frameIndices.empty()) {
		for (unsigned int i = 0; i < m_debugColorImages.size(); i++) pointCloudFrameIndices.push_back(i);
	}
	else {
		pointCloudFrameIndices = frameIndices;
	}

	PointCloudf pc;
	unsigned int width = m_debugDepthImages[0].getWidth();
	float scaleWidth = (float)m_debugColorImages[0].getWidth() / (float)m_debugDepthImages[0].getWidth();
	float scaleHeight = (float)m_debugColorImages[0].getHeight() / (float)m_debugDepthImages[0].getHeight();
	for (unsigned int k = 0; k < pointCloudFrameIndices.size(); k++) {
		recordPointCloud(pc, pointCloudFrameIndices[k]);
	} // frames
	std::cout << "saving to file... ";
	PointCloudIOf::saveToFile(filename, pc);
	std::cout << "done!" << std::endl;
}

void TestMatching::recordPointCloud(PointCloudf& pc, unsigned int frame) const
{
	if (m_referenceTrajectory[frame][0] == -std::numeric_limits<float>::infinity()) {
		std::cout << "warning: invalid frame " << frame << std::endl;
		return;
	}
	unsigned int depthWidth = m_debugDepthImages[0].getWidth();
	float scaleWidth = (float)m_debugColorImages[0].getWidth() / (float)m_debugDepthImages[0].getWidth();
	float scaleHeight = (float)m_debugColorImages[0].getHeight() / (float)m_debugDepthImages[0].getHeight();

	for (unsigned int p = 0; p < m_debugDepthImages[frame].getNumPixels(); p++) {
		unsigned int x = p%depthWidth; unsigned int y = p/depthWidth;
		float depth = m_debugDepthImages[frame](x, y);
		if (depth != -std::numeric_limits<float>::infinity()) {
			vec3f camPos = m_depthIntrinsicsInv * (depth * vec3f((float)x, (float)y, 1.0f));
			pc.m_points.push_back(m_referenceTrajectory[frame] * camPos);
			unsigned int cx = (unsigned int)math::round(x * scaleWidth);
			unsigned int cy = (unsigned int)math::round(y * scaleHeight);
			vec4uc c = m_debugColorImages[frame](cx, cy);
			pc.m_colors.push_back(vec4f(c) / 255.0f);

			vec3f wpos = m_referenceTrajectory[frame] * camPos;
			if (isnan(wpos.x) || isnan(wpos.y) || isnan(wpos.z))
				int a = 5;
		} // valid depth
	} // depth pixels
}

void TestMatching::saveMatchToPLY(const std::string& dir, const vec2ui& imageIndices, bool filtered) const
{
	std::cout << "saving match " << imageIndices << "... ";
	// frames
	PointCloudf pc0, pc1;
	recordPointCloud(pc0, imageIndices.x);
	recordPointCloud(pc1, imageIndices.y);
	PointCloudIOf::saveToFile(dir + std::to_string(imageIndices.x) + ".ply", pc0);
	PointCloudIOf::saveToFile(dir + std::to_string(imageIndices.y) + ".ply", pc1);

	// keys
	vec4f red(1.0f, 0.0f, 0.0f, 1.0f);
	vec4f green(0.0f, 1.0f, 0.0f, 1.0f);
	MeshDataf keys0, keys1;
	recordKeysMeshData(keys0, keys1, imageIndices, filtered, red, green);
	MeshIOf::saveToFile(dir + std::to_string(imageIndices.x) + "-keys.ply", keys0);
	MeshIOf::saveToFile(dir + std::to_string(imageIndices.y) + "-keys.ply", keys1);
	std::cout << "done!" << std::endl;
}

void TestMatching::recordKeysMeshData(MeshDataf& keys0, MeshDataf& keys1, const vec2ui& imageIndices, bool filtered, const vec4f& color0, const vec4f& color1) const
{
	MLIB_ASSERT(m_referenceTrajectory[imageIndices.x][0] != -std::numeric_limits<float>::infinity() &&
		m_referenceTrajectory[imageIndices.y][0] != -std::numeric_limits<float>::infinity());
	const unsigned int matchIdx = ((imageIndices.y - 1) * imageIndices.y) / 2 + imageIndices.x;

	std::vector<SIFTKeyPoint> keys;
	m_siftManager->getSIFTKeyPointsDEBUG(keys);

	const unsigned int OFFSET = filtered ? MAX_MATCHES_PER_IMAGE_PAIR_FILTERED : MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	unsigned int numMatches;
	std::vector<float> matchDists;
	std::vector<uint2> matchKeyIndices;
	if (filtered) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_filtNumMatchesPerImagePair + matchIdx, sizeof(int), cudaMemcpyDeviceToHost));
		MLIB_ASSERT(numMatches < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
	}
	else {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_numMatchesPerImagePair + matchIdx, sizeof(int), cudaMemcpyDeviceToHost));
		MLIB_ASSERT(numMatches < MAX_MATCHES_PER_IMAGE_PAIR_RAW);
	}
	if (numMatches == 0) {
		std::cout << "error: no matches for images " << imageIndices << std::endl;
		return;
	}
	matchDists.resize(numMatches); matchKeyIndices.resize(numMatches);
	if (filtered) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDists.data(), d_filtMatchDistancesPerImagePair + matchIdx*OFFSET, sizeof(float)*numMatches, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndices.data(), d_filtMatchKeyIndicesPerImagePair + matchIdx*OFFSET, sizeof(uint2)*numMatches, cudaMemcpyDeviceToHost));
	}
	else {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDists.data(), d_matchDistancesPerImagePair + matchIdx*OFFSET, sizeof(float)*numMatches, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchKeyIndices.data(), d_matchKeyIndicesPerImagePair + matchIdx*OFFSET, sizeof(uint2)*numMatches, cudaMemcpyDeviceToHost));
	}

	const float radius = 0.02f;
	std::vector<vec3f> srcPts(numMatches), tgtPts(numMatches);
	getSrcAndTgtPts(keys.data(), matchKeyIndices.data(), numMatches, (float3*)srcPts.data(), (float3*)tgtPts.data(), MatrixConversion::toCUDA(m_siftIntrinsicsInv));
	for (unsigned int i = 0; i < numMatches; i++) {
		keys0.merge(Shapesf::sphere(radius, m_referenceTrajectory[imageIndices.x] * srcPts[i], 10, 10, color0).getMeshData());
		keys1.merge(Shapesf::sphere(radius, m_referenceTrajectory[imageIndices.y] * tgtPts[i], 10, 10, color1).getMeshData());
	}
}



