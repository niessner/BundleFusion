#pragma once

#include "GlobalBundlingState.h"

class SIFTImageManager;

class TestMatching {
public:
	TestMatching();
	~TestMatching();

	void save(const std::string& filename) const;
	void load(const std::string& filename);

	void test();

	static void initDebugVerifyMatches(std::unordered_set<vec2ui>& unsureMatches, std::unordered_set<vec2ui>& closeMatches, std::unordered_set<vec2ui>& badMatches);

private:
	void matchAll();
	void clearMatching() {
		m_numMatchesPerImagePair.clear();
		m_matchDistancesPerImagePair.clear();
		m_matchKeyIndicesPerImagePair.clear();
	}
	void resizeMatching(unsigned int n) {
		m_numMatchesPerImagePair.resize(n);
		m_matchDistancesPerImagePair.resize(n);
		m_matchKeyIndicesPerImagePair.resize(n);
	}

	SIFTImageManager* m_siftManager;

	std::vector< std::vector<unsigned int> >	m_numMatchesPerImagePair; // vector for each image (up to image)
	std::vector< std::vector<float> >			m_matchDistancesPerImagePair;
	std::vector< std::vector<uint2> >			m_matchKeyIndicesPerImagePair; // index into d_keyPoints of m_siftManager
};