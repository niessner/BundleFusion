
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
			std::vector<float> matchDistancesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_RAW);
			std::vector<vec2ui> matchKeyIndicesPerImagePair(i*MAX_MATCHES_PER_IMAGE_PAIR_RAW);

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
	unsigned int numImages = m_siftManager->getNumImages();

	std::vector<vec2ui> imagePairMatchesFiltered;
	filter(imagePairMatchesFiltered);
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

		//if (cur == 4) { //3,4
		//	std::vector<int> numMatches(cur);
		//	std::vector<uint2> keyIndices(MAX_MATCHES_PER_IMAGE_PAIR_RAW);
		//	int a = 5;
		//}

		std::vector<int> numMatches(cur);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), getNumMatchesCUDA(cur, filtered), sizeof(int)*cur, cudaMemcpyDeviceToHost));
		for (unsigned int i = 0; i < numMatches.size(); i++) {
			if (numMatches[i] >(int)minNumMatches) {
				m_origMatches.push_back(vec2ui(i, cur));
			}
		}

		//!!!DEBUGGING 
		//!!!DEBUGGING

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
				if (numMatches[ii] < 0 || numMatches[ii] > MAX_MATCHES_PER_IMAGE_PAIR_RAW)
					int a = 5;
			}
		}
		//!!!DEBUGGING

		SIFTMatchFilter::filterKeyPointMatchesDEBUG(cur, m_siftManager, MatrixConversion::toCUDA(m_siftIntrinsicsInv));
		//SIFTMatchFilter::ransacKeyPointMatchesDEBUG(cur, m_siftManager, MatrixConversion::toCUDA(m_siftIntrinsicsInv), maxResThresh2, false);

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
	//std::unordered_set<vec2ui> unsureMatches, closeMatches, badMatches;
	//initDebugVerifyMatches(unsureMatches, closeMatches, badMatches);
	//
	//falsePositives.clear();
	//falseNegatives.clear();
	//std::unordered_set<vec2ui> filteredSet;
	//for (unsigned int i = 0; i < imagePairMatchesFiltered.size(); i++) {
	//	if (badMatches.find(imagePairMatchesFiltered[i]) != badMatches.end()) { // bad but found good
	//		falsePositives.push_back(imagePairMatchesFiltered[i]);
	//	}
	//	filteredSet.insert(imagePairMatchesFiltered[i]);
	//}

	//for (unsigned int i = 0; i < m_origMatches.size(); i++) {
	//	if (filteredSet.find(m_origMatches[i]) == filteredSet.end()) { // filtered out
	//		if (badMatches.find(m_origMatches[i]) == badMatches.end()) { // not a bad match
	//			falseNegatives.push_back(m_origMatches[i]);
	//		}
	//	}
	//}

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

void TestMatching::initDebugVerifyMatches(std::unordered_set<vec2ui>& unsureMatches, std::unordered_set<vec2ui>& closeMatches, std::unordered_set<vec2ui>& badMatches)
{
	unsureMatches.clear();
	closeMatches.clear();
	badMatches.clear();

	//unsure
	unsureMatches.insert(vec2ui(29, 227));
	unsureMatches.insert(vec2ui(139, 179));
	unsureMatches.insert(vec2ui(140, 179));
	unsureMatches.insert(vec2ui(140, 181));
	unsureMatches.insert(vec2ui(181, 197));

	//close
	closeMatches.insert(vec2ui(6, 221));
	closeMatches.insert(vec2ui(75, 85));

	//completely off
	badMatches.insert(vec2ui(3, 57));
	badMatches.insert(vec2ui(4, 63));
	badMatches.insert(vec2ui(5, 63));
	badMatches.insert(vec2ui(5, 130));
	badMatches.insert(vec2ui(5, 208));
	badMatches.insert(vec2ui(8, 19));
	badMatches.insert(vec2ui(12, 68));
	badMatches.insert(vec2ui(16, 35));
	badMatches.insert(vec2ui(16, 43));
	badMatches.insert(vec2ui(17, 73));
	badMatches.insert(vec2ui(17, 89));
	badMatches.insert(vec2ui(17, 183));
	badMatches.insert(vec2ui(18, 68));
	badMatches.insert(vec2ui(18, 73));
	badMatches.insert(vec2ui(22, 29));
	badMatches.insert(vec2ui(24, 114));
	badMatches.insert(vec2ui(25, 33));
	badMatches.insert(vec2ui(25, 139));
	badMatches.insert(vec2ui(25, 140));
	badMatches.insert(vec2ui(25, 179));
	badMatches.insert(vec2ui(26, 184));
	badMatches.insert(vec2ui(27, 141));
	badMatches.insert(vec2ui(27, 185));
	badMatches.insert(vec2ui(28, 45));
	badMatches.insert(vec2ui(28, 64));
	badMatches.insert(vec2ui(28, 140));
	badMatches.insert(vec2ui(28, 141));
	badMatches.insert(vec2ui(28, 183));
	badMatches.insert(vec2ui(28, 184));
	badMatches.insert(vec2ui(28, 185));
	badMatches.insert(vec2ui(29, 140));
	badMatches.insert(vec2ui(29, 141));
	badMatches.insert(vec2ui(29, 153));
	badMatches.insert(vec2ui(29, 179));
	badMatches.insert(vec2ui(29, 181));
	badMatches.insert(vec2ui(29, 182));
	badMatches.insert(vec2ui(29, 185));
	badMatches.insert(vec2ui(30, 105));
	badMatches.insert(vec2ui(32, 180));
	badMatches.insert(vec2ui(32, 183));
	badMatches.insert(vec2ui(32, 184));
	badMatches.insert(vec2ui(32, 185));
	badMatches.insert(vec2ui(33, 141));
	badMatches.insert(vec2ui(33, 142));
	badMatches.insert(vec2ui(33, 180));
	badMatches.insert(vec2ui(33, 181));
	badMatches.insert(vec2ui(33, 182));
	badMatches.insert(vec2ui(33, 183));
	badMatches.insert(vec2ui(33, 184));
	badMatches.insert(vec2ui(34, 162));
	badMatches.insert(vec2ui(35, 182));
	badMatches.insert(vec2ui(35, 184));
	badMatches.insert(vec2ui(35, 185));
	badMatches.insert(vec2ui(36, 185));
	badMatches.insert(vec2ui(43, 181));
	badMatches.insert(vec2ui(43, 228));
	badMatches.insert(vec2ui(44, 185));
	badMatches.insert(vec2ui(44, 228));
	badMatches.insert(vec2ui(56, 92));
	badMatches.insert(vec2ui(61, 168));
	badMatches.insert(vec2ui(62, 70));
	badMatches.insert(vec2ui(63, 70));
	badMatches.insert(vec2ui(64, 85));
	badMatches.insert(vec2ui(64, 97));
	badMatches.insert(vec2ui(69, 77));
	badMatches.insert(vec2ui(70, 102));
	badMatches.insert(vec2ui(71, 130));
	badMatches.insert(vec2ui(74, 142));
	badMatches.insert(vec2ui(78, 184));
	badMatches.insert(vec2ui(83, 119));
	badMatches.insert(vec2ui(89, 103));
	badMatches.insert(vec2ui(92, 162));
	badMatches.insert(vec2ui(92, 168));
	badMatches.insert(vec2ui(92, 172));
	badMatches.insert(vec2ui(92, 175));
	badMatches.insert(vec2ui(92, 176));
	badMatches.insert(vec2ui(92, 177));
	badMatches.insert(vec2ui(93, 162));
	badMatches.insert(vec2ui(93, 177));
	badMatches.insert(vec2ui(94, 163));
	badMatches.insert(vec2ui(94, 166));
	badMatches.insert(vec2ui(94, 169));
	badMatches.insert(vec2ui(94, 171));
	badMatches.insert(vec2ui(96, 163));
	badMatches.insert(vec2ui(96, 166));
	badMatches.insert(vec2ui(96, 171));
	badMatches.insert(vec2ui(96, 173));
	badMatches.insert(vec2ui(96, 174));
	badMatches.insert(vec2ui(97, 162));
	badMatches.insert(vec2ui(97, 168));
	badMatches.insert(vec2ui(97, 169));
	badMatches.insert(vec2ui(97, 172));
	badMatches.insert(vec2ui(97, 173));
	badMatches.insert(vec2ui(97, 174));
	badMatches.insert(vec2ui(97, 176));
	badMatches.insert(vec2ui(98, 128));
	badMatches.insert(vec2ui(98, 162));
	badMatches.insert(vec2ui(98, 171));
	badMatches.insert(vec2ui(98, 173));
	badMatches.insert(vec2ui(99, 162));
	badMatches.insert(vec2ui(99, 168));
	badMatches.insert(vec2ui(99, 176));
	badMatches.insert(vec2ui(100, 122));
	badMatches.insert(vec2ui(100, 162));
	badMatches.insert(vec2ui(100, 168));
	badMatches.insert(vec2ui(100, 176));
	badMatches.insert(vec2ui(101, 120));
	badMatches.insert(vec2ui(101, 172));
	badMatches.insert(vec2ui(102, 118));
	badMatches.insert(vec2ui(102, 176));
	badMatches.insert(vec2ui(110, 142));
	badMatches.insert(vec2ui(114, 186));
	badMatches.insert(vec2ui(117, 149));
	badMatches.insert(vec2ui(121, 215));
	badMatches.insert(vec2ui(123, 148));
	badMatches.insert(vec2ui(124, 207));
	badMatches.insert(vec2ui(124, 221));
	badMatches.insert(vec2ui(134, 208));
	badMatches.insert(vec2ui(139, 174));
	badMatches.insert(vec2ui(140, 228));
	badMatches.insert(vec2ui(142, 228));
	badMatches.insert(vec2ui(142, 229));
	badMatches.insert(vec2ui(153, 184));
	badMatches.insert(vec2ui(174, 183));
	badMatches.insert(vec2ui(179, 229));
	badMatches.insert(vec2ui(183, 229));

	// new ones
	badMatches.insert(vec2ui(7, 208));
	badMatches.insert(vec2ui(10, 19));
	badMatches.insert(vec2ui(15, 93));
	badMatches.insert(vec2ui(16, 34));
	badMatches.insert(vec2ui(17, 153));
	badMatches.insert(vec2ui(19, 227));
	badMatches.insert(vec2ui(22, 32));
	badMatches.insert(vec2ui(22, 36));
	badMatches.insert(vec2ui(24, 33));
	badMatches.insert(vec2ui(24, 180));
	badMatches.insert(vec2ui(24, 219));
	badMatches.insert(vec2ui(25, 71));
	badMatches.insert(vec2ui(25, 115));
	badMatches.insert(vec2ui(25, 142));
	badMatches.insert(vec2ui(25, 153));
	badMatches.insert(vec2ui(25, 183));
	badMatches.insert(vec2ui(25, 186));
	badMatches.insert(vec2ui(25, 226));
	badMatches.insert(vec2ui(27, 184));
	badMatches.insert(vec2ui(27, 186));
	badMatches.insert(vec2ui(28, 182));
	badMatches.insert(vec2ui(29, 139));
	badMatches.insert(vec2ui(29, 152));
	badMatches.insert(vec2ui(29, 180));
	badMatches.insert(vec2ui(29, 183));
	badMatches.insert(vec2ui(32, 140));
	badMatches.insert(vec2ui(32, 181));
	badMatches.insert(vec2ui(33, 139));
	badMatches.insert(vec2ui(33, 140));
	badMatches.insert(vec2ui(33, 185));
	badMatches.insert(vec2ui(34, 140));
	badMatches.insert(vec2ui(34, 141));
	badMatches.insert(vec2ui(34, 142));
	badMatches.insert(vec2ui(34, 159));
	badMatches.insert(vec2ui(35, 140));
	badMatches.insert(vec2ui(35, 142));
	badMatches.insert(vec2ui(35, 162));
	badMatches.insert(vec2ui(35, 183));
	badMatches.insert(vec2ui(36, 139));
	badMatches.insert(vec2ui(36, 140));
	badMatches.insert(vec2ui(36, 182));
	badMatches.insert(vec2ui(36, 184));
	badMatches.insert(vec2ui(39, 165));
	badMatches.insert(vec2ui(43, 179));
	badMatches.insert(vec2ui(43, 180));
	badMatches.insert(vec2ui(43, 184));
	badMatches.insert(vec2ui(44, 182));
	badMatches.insert(vec2ui(44, 183));
	badMatches.insert(vec2ui(44, 186));
	badMatches.insert(vec2ui(44, 198));
	badMatches.insert(vec2ui(55, 207));
	badMatches.insert(vec2ui(57, 67));
	badMatches.insert(vec2ui(59, 71));
	badMatches.insert(vec2ui(60, 71));
	badMatches.insert(vec2ui(63, 182));
	badMatches.insert(vec2ui(64, 98));
	badMatches.insert(vec2ui(64, 99));
	badMatches.insert(vec2ui(65, 101));
	badMatches.insert(vec2ui(66, 168));
	badMatches.insert(vec2ui(66, 170));
	badMatches.insert(vec2ui(67, 177));
	badMatches.insert(vec2ui(70, 175));
	badMatches.insert(vec2ui(72, 199));
	badMatches.insert(vec2ui(76, 110));
	badMatches.insert(vec2ui(78, 216));
	//badMatches.insert(vec2ui(82, 111)); // close but not quite
	badMatches.insert(vec2ui(85, 112));
	badMatches.insert(vec2ui(85, 123));
	badMatches.insert(vec2ui(85, 124));
	badMatches.insert(vec2ui(85, 126));
	badMatches.insert(vec2ui(86, 126));
	badMatches.insert(vec2ui(87, 220));
	badMatches.insert(vec2ui(90, 103));
	badMatches.insert(vec2ui(93, 124));
	badMatches.insert(vec2ui(93, 168));
	badMatches.insert(vec2ui(93, 172));
	badMatches.insert(vec2ui(94, 165));
	badMatches.insert(vec2ui(95, 103));
	badMatches.insert(vec2ui(96, 165));
	badMatches.insert(vec2ui(96, 168));
	badMatches.insert(vec2ui(96, 169));
	badMatches.insert(vec2ui(96, 176));
	badMatches.insert(vec2ui(98, 122));
	badMatches.insert(vec2ui(98, 174));
	badMatches.insert(vec2ui(99, 172));
	badMatches.insert(vec2ui(101, 121));
	badMatches.insert(vec2ui(105, 127));
	badMatches.insert(vec2ui(110, 115));
	badMatches.insert(vec2ui(114, 140));
	badMatches.insert(vec2ui(116, 204));
	badMatches.insert(vec2ui(117, 128)); // close but not quite
	badMatches.insert(vec2ui(124, 132));
	badMatches.insert(vec2ui(124, 177));
	badMatches.insert(vec2ui(127, 185));
	badMatches.insert(vec2ui(132, 146));
	badMatches.insert(vec2ui(139, 181));
	badMatches.insert(vec2ui(140, 186));
	badMatches.insert(vec2ui(150, 204));
	badMatches.insert(vec2ui(151, 182));
	badMatches.insert(vec2ui(152, 182));
	badMatches.insert(vec2ui(153, 199));
	badMatches.insert(vec2ui(159, 174));
	badMatches.insert(vec2ui(159, 221));
	badMatches.insert(vec2ui(179, 228));
	badMatches.insert(vec2ui(180, 213));
	badMatches.insert(vec2ui(180, 228));
	badMatches.insert(vec2ui(181, 229));
	badMatches.insert(vec2ui(182, 228));
	badMatches.insert(vec2ui(183, 228));
	badMatches.insert(vec2ui(185, 229));
	badMatches.insert(vec2ui(197, 229));
	badMatches.insert(vec2ui(213, 222));
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
	//TODO HERE
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

		float maxMatchDistance = 0.0f;
		RGBColor lowColor = ml::RGBColor::Blue;
		RGBColor highColor = ml::RGBColor::Red;
		unsigned int numMatchesToPrint = filtered ? numMatches[idx] : std::min(GlobalBundlingState::get().s_minNumMatchesGlobal, (unsigned int)numMatches[idx]);
		for (unsigned int i = 0; i < numMatchesToPrint; i++) {
			const SIFTKeyPoint& key1 = keys[matchKeyIndices[idx*OFFSET + i].x];
			const SIFTKeyPoint& key2 = keys[matchKeyIndices[idx*OFFSET + i].y];
			if (matchDists[idx*OFFSET + i] > maxMatchDistance) maxMatchDistance = matchDists[idx*OFFSET + i];

			RGBColor c = RGBColor::interpolate(lowColor, highColor, matchDists[idx*OFFSET + i] / distMax);
			vec3f color(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
			vec2i p0 = ml::math::round(ml::vec2f(key1.pos.x, key1.pos.y));
			vec2i p1 = ml::math::round(ml::vec2f(key2.pos.x + image1.getWidth(), key2.pos.y));
			ImageHelper::drawCircle(matchImage, p0, ml::math::round(key1.scale), color);
			ImageHelper::drawCircle(matchImage, p1, ml::math::round(key2.scale), color);
			ImageHelper::drawLine(matchImage, p0, p1, color);
		}
		//std::cout << "(" << imageIndices << "): max match distance = " << maxMatchDistance << std::endl;
		FreeImageWrapper::saveImage(filename, matchImage);
	}
}

void TestMatching::loadColorImagesFromSensor(const std::string& filename, unsigned int skip)
{
	std::cout << "loading color images from sensor... ";
	CalibratedSensorData cs;
	BinaryDataStreamFile s(filename, false);
	s >> cs;
	s.closeStream();

	m_debugColorImages.resize((cs.m_ColorNumFrames - 1) / skip + 1);
	MLIB_ASSERT((cs.m_ColorNumFrames - 1) / skip == m_debugColorImages.size() - 1);
	for (unsigned int i = 0; i < cs.m_ColorNumFrames; i += skip) {
		MLIB_ASSERT(i / skip < m_debugColorImages.size());
		m_debugColorImages[i / skip] = ColorImageR8G8B8A8(cs.m_ColorImageWidth, cs.m_ColorImageHeight, cs.m_ColorImages[i]);
	}
	std::cout << "done! (" << m_debugColorImages.size() << " of " << cs.m_ColorNumFrames << ")" << std::endl;
}

void TestMatching::saveColorImages(const std::string& filename) const
{
	if (m_debugColorImages.empty()) return;
	std::cout << "saving color images... ";

	BinaryDataStreamFile s(filename, true);
	s << m_debugColorImages.size();
	for (unsigned int i = 0; i < m_debugColorImages.size(); i++) {
		s << m_debugColorImages[i];
	}
	s.closeStream();
	std::cout << "done!" << std::endl;
}

void TestMatching::loadColorImages(const std::string& filename)
{
	std::cout << "loading color images... ";

	BinaryDataStreamFile s(filename, false);
	size_t numColorImages;
	s >> numColorImages;
	m_debugColorImages.resize(numColorImages);
	for (unsigned int i = 0; i < m_debugColorImages.size(); i++) {
		s >> m_debugColorImages[i];
	}
	s.closeStream();
	std::cout << "done!" << std::endl;
}

void TestMatching::saveReferenceTrajectory(const std::string& filename) const
{
	if (m_referenceTrajectory.empty()) return;
	std::cout << "saving reference trajectory... ";

	BinaryDataStreamFile s(filename, true);
	s << m_referenceTrajectory;
	s.closeStream();
	std::cout << "done!" << std::endl;
}

void TestMatching::loadReferenceTrajectory(const std::string& filename, unsigned int skip /*= 1*/)
{
	std::cout << "loading reference trajectory... ";

	BinaryDataStreamFile s(filename, false);
	s >> m_referenceTrajectory;
	s.closeStream();
	if (skip > 1) {
		std::vector<mat4f> trajectory((m_referenceTrajectory.size() - 1) / skip + 1);
		for (unsigned int i = 0; i < m_referenceTrajectory.size(); i += skip) {
			trajectory[i / skip] = m_referenceTrajectory[i];
		}
		m_referenceTrajectory = trajectory;
	}

	std::cout << "done!" << std::endl;
}



