#pragma once

#include "GlobalBundlingState.h"
#include "GlobalDefines.h"

class SIFTImageManager;

template<>
struct std::hash<vec2ui> : public std::unary_function < vec2ui, size_t > {
	size_t operator()(const vec2ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		//const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);// ^ ((size_t)v.z * p2);
		return res;
	}
};


class TestMatching {
public:
	TestMatching();
	~TestMatching();

	//! debug hack
	void loadIntrinsics(const std::string& filename);
	void loadFromSensor(const std::string& sensorFile, const std::string& trajectoryFile, unsigned int skip);
	void saveImages(const std::string& filename) const;
	void loadImages(const std::string& filename);

	void save(const std::string& filename) const;
	void load(const std::string& filename, const std::string siftFile);

	void test();

	void saveToPointCloud(const std::string& filename, const std::vector<unsigned int>& frameIndices = std::vector<unsigned int>()) const;
	void saveMatchToPLY(const std::string& dir, const vec2ui& imageIndices, bool filtered) const;

private:
	int* getNumMatchesCUDA(unsigned int curFrame, bool filtered) {
		return const_cast<int*>(static_cast<const TestMatching*>(this)->getNumMatchesCUDA(curFrame, filtered));
	}
	float* getMatchDistsCUDA(unsigned int curFrame, bool filtered) {
		return const_cast<float*>(static_cast<const TestMatching*>(this)->getMatchDistsCUDA(curFrame, filtered));
	}
	uint2* getMatchKeyIndicesCUDA(unsigned int curFrame, bool filtered) {
		return const_cast<uint2*>(static_cast<const TestMatching*>(this)->getMatchKeyIndicesCUDA(curFrame, filtered));
	}
	const int* getNumMatchesCUDA(unsigned int curFrame, bool filtered) const {
		MLIB_ASSERT(curFrame > 0);
		unsigned int offset = (curFrame - 1) * curFrame / 2;
		if (filtered) return d_filtNumMatchesPerImagePair + offset;
		else return d_numMatchesPerImagePair + offset;
	}
	const float* getMatchDistsCUDA(unsigned int curFrame, bool filtered) const {
		MLIB_ASSERT(curFrame > 0);
		unsigned int offset = (curFrame - 1) * curFrame / 2;
		if (filtered) return d_filtMatchDistancesPerImagePair + offset * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED;
		else return d_matchDistancesPerImagePair + offset * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	}
	const uint2* getMatchKeyIndicesCUDA(unsigned int curFrame, bool filtered) const {
		MLIB_ASSERT(curFrame > 0);
		unsigned int offset = (curFrame - 1) * curFrame / 2;
		if (filtered) return d_filtMatchKeyIndicesPerImagePair + offset * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED;
		else return d_matchKeyIndicesPerImagePair + offset * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	}

	void matchAll();
	void filter(std::vector<vec2ui>& imagePairMatchesFiltered);
	void checkFiltered(const std::vector<vec2ui>& imagePairMatchesFiltered, std::vector<vec2ui>& falsePositives, std::vector<vec2ui>& falseNegatives) const;
	void printMatches(const std::string& outDir, const std::vector<vec2ui>& imagePairMatches, bool filtered) const;

	void getSrcAndTgtPts(const SIFTKeyPoint* keyPoints, const uint2* keyIndices, unsigned int numMatches,
		float3* srcPts, float3* tgtPts, const float4x4& colorIntrinsicsInv) const {
		for (unsigned int i = 0; i < numMatches; i++) {
			// source points
			const SIFTKeyPoint& key0 = keyPoints[keyIndices[i].x];
			srcPts[i] = colorIntrinsicsInv * (key0.depth * make_float3(key0.pos.x, key0.pos.y, 1.0f));

			// target points
			const SIFTKeyPoint& key1 = keyPoints[keyIndices[i].y];
			tgtPts[i] = colorIntrinsicsInv * (key1.depth * make_float3(key1.pos.x, key1.pos.y, 1.0f));
		}
	}

	void recordPointCloud(PointCloudf& pc, unsigned int frame) const;
	void recordKeysMeshData(MeshDataf& keys0, MeshDataf& keys1, const vec2ui& imageIndices, bool filtered, const vec4f& color0, const vec4f& color1) const;

	SIFTImageManager* m_siftManager;

	std::vector<vec2ui> m_origMatches;

	int*	d_numMatchesPerImagePair;
	float*	d_matchDistancesPerImagePair;
	uint2*	d_matchKeyIndicesPerImagePair;

	int*	d_filtNumMatchesPerImagePair;
	float*	d_filtMatchDistancesPerImagePair;
	uint2*	d_filtMatchKeyIndicesPerImagePair;

	mat4f	m_siftIntrinsics;
	mat4f	m_siftIntrinsicsInv;
	bool	m_bHasMatches;

	std::vector<ColorImageR8G8B8A8> m_debugColorImages;
	std::vector<DepthImage32>		m_debugDepthImages;
	mat4f							m_depthIntrinsicsInv;
	std::vector<mat4f>				m_referenceTrajectory;
};