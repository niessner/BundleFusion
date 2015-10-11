
#include "stdafx.h"
#include "SIFTImageManager.h"
#include "SiftGPU/SiftMatch.h"

#include "testMatching.h"

TestMatching::TestMatching()
{
	m_siftManager = new SIFTImageManager(GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumKeysPerImage);
}

TestMatching::~TestMatching()
{
	SAFE_DELETE(m_siftManager);
}

void TestMatching::load(const std::string& filename)
{
	const std::string siftFile = util::removeExtensions(filename);
	if (!util::fileExists(siftFile)) throw MLIB_EXCEPTION(siftFile + " does not exist");
	m_siftManager->loadFromFile(siftFile);

	if (!util::fileExists(filename)) {
		std::cout << "warning: " << filename << " does not exist, need to re-compute" << std::endl;
		return;
	}
	BinaryDataStreamFile s(filename, false);
	size_t numImages;
	s >> numImages;
	m_numMatchesPerImagePair.resize(numImages);
	m_matchDistancesPerImagePair.resize(numImages);
	m_matchKeyIndicesPerImagePair.resize(numImages);
	for (unsigned int i = 0; i < numImages; i++) {
		s >> m_numMatchesPerImagePair[i];
		s >> m_matchDistancesPerImagePair[i];
		s >> m_matchKeyIndicesPerImagePair[i];
	}
	s.closeStream();
}

void TestMatching::save(const std::string& filename) const
{
	BinaryDataStreamFile s(filename, true);
	size_t numImages = m_numMatchesPerImagePair.size();
	s << numImages;
	for (unsigned int i = 0; i < numImages; i++) {
		s << m_numMatchesPerImagePair[i];
		s << m_matchDistancesPerImagePair[i];
		s << m_matchKeyIndicesPerImagePair[i];
	}
	s.closeStream();
}

void TestMatching::test()
{
	if (m_numMatchesPerImagePair.empty()) {
		matchAll();
	}
	unsigned int numImages = m_siftManager->getNumImages();
	MLIB_ASSERT(m_numMatchesPerImagePair.size() == numImages &&
		m_matchDistancesPerImagePair.size() == numImages &&
		m_matchKeyIndicesPerImagePair.size() == numImages);


}

void TestMatching::matchAll()
{
	SiftMatchGPU* siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	siftMatcher->InitSiftMatch();
	const float ratioMax = GlobalBundlingState::get().s_siftMatchRatioMaxGlobal;
	const float matchThresh = GlobalBundlingState::get().s_siftMatchThresh;
	unsigned int numImages = m_siftManager->getNumImages();

	clearMatching();
	resizeMatching(numImages);

	const std::vector<int>& valid = m_siftManager->getValidImages();
	for (unsigned int cur = 1; cur < numImages; cur++) {
		if (valid[cur] == 0) continue;

		SIFTImageGPU& curImage = m_siftManager->getImageGPU(cur);
		int num2 = (int)m_siftManager->getNumKeyPointsPerImage(cur);

		// match to all previous
		m_numMatchesPerImagePair[cur].resize(cur);
		m_matchDistancesPerImagePair[cur].resize(cur);
		m_matchKeyIndicesPerImagePair[cur].resize(cur);
		for (unsigned int prev = 0; prev < cur; prev++) {
			SIFTImageGPU& prevImage = m_siftManager->getImageGPU(prev);
			int num1 = (int)m_siftManager->getNumKeyPointsPerImage(prev);

			uint2 keyPointOffset = make_uint2(0, 0);
			ImagePairMatch& imagePairMatch = m_siftManager->getImagePairMatch(prev, keyPointOffset);

			siftMatcher->SetDescriptors(0, num1, (unsigned char*)prevImage.d_keyPointDescs);
			siftMatcher->SetDescriptors(1, num2, (unsigned char*)curImage.d_keyPointDescs);
			siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset, matchThresh, ratioMax);

			// save
		}
	}
	SAFE_DELETE(siftMatcher);
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



