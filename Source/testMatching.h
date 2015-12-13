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
	enum STAGE {
		NONE,
		INITIALIZED,
		HAS_MATCHES,
		FILTERED_IMPAIR_MATCHES,
		FILTERED_SA,
		FILTERED_DV,
		NUM_STAGES
	};

	TestMatching();
	~TestMatching();

	//test sparse+dense opt
	void runOpt();
	void analyzeLocalOpts();

	// match within first numFrames of sensorFile
	void match(const std::string& loadFile, const std::string& outDir, const std::string& sensorFile, const vec2ui& frames = vec2ui((unsigned int)-1));

	void matchFrame(unsigned int frame, bool print, bool checkReference);
	// matches last frame and optimizes
	void debugOptimizeGlobal();
	void checkCorrespondences();
	void printKeys();
	void debugMatchInfo();

	//! debug hack
	void loadFromSensor(const std::string& sensorFile, const std::string& trajectoryFile, unsigned int skip, const vec2ui& frames = vec2ui((unsigned int)-1));

	void save(const std::string& filename) const;
	void load(const std::string& filename, const std::string siftFile);

	void test();

	void saveToPointCloud(const std::string& filename, const std::vector<unsigned int>& frameIndices = std::vector<unsigned int>()) const;
	void saveMatchToPLY(const std::string& dir, const vec2ui& imageIndices, bool filtered) const;

	void printCacheFrames(const std::string& dir) const;

private:
	// for global offline matching
	void detectKeys(const std::vector<ColorImageR8G8B8A8> &colorImages, const std::vector<DepthImage32> &depthImages, SIFTImageManager *siftManager) const;
	void initSiftParams(unsigned int widthDepth, unsigned int heightDepth, unsigned int widthColor, unsigned int heightColor);

	// for online
	void constructSparseSystem(const std::vector<ColorImageR8G8B8A8> &colorImages, const std::vector<DepthImage32> &depthImages, SIFTImageManager *siftManager, const CUDACache* cudaCache);

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

	void matchAll(bool print = false, const std::string outDir = "");
	void filterImagePairMatches(std::vector<vec2ui>& imagePairMatchesFiltered, bool print = false, const std::string& outDir = "");
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

	void filterBySurfaceArea(bool print = false, const std::string& outDir = "");
	void filterByDenseVerify(bool print = false, const std::string& outDir = "");
	void createCachedFrames();

	void freeCachedFrames() {
		for (CUDACachedFrame& f : m_cachedFrames) {
			f.free();
		}
		m_cachedFrames.clear();

		MLIB_CUDA_SAFE_FREE(d_cachedFrames);
	}
	void allocCachedFrames(unsigned int num, unsigned int width, unsigned int height) {
		freeCachedFrames();
		m_cachedFrames.resize(num);
		for (CUDACachedFrame& f : m_cachedFrames) {
			f.alloc(width, height);
		}

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cachedFrames, sizeof(CUDACachedFrame)*num));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_cachedFrames, m_cachedFrames.data(), sizeof(CUDACachedFrame)*num, cudaMemcpyHostToDevice));
	}

	SIFTImageManager* m_siftManager;

	std::vector<vec2ui> m_origMatches;

	int*	d_numMatchesPerImagePair;
	float*	d_matchDistancesPerImagePair;
	uint2*	d_matchKeyIndicesPerImagePair;

	int*	d_filtNumMatchesPerImagePair;
	float*	d_filtMatchDistancesPerImagePair;
	uint2*	d_filtMatchKeyIndicesPerImagePair;

	STAGE	m_stage;

	// params
	unsigned int m_widthSift;
	unsigned int m_heightSift;
	unsigned int m_widthDepth;
	unsigned int m_heightDepth;
	CalibrationData m_colorCalibration;
	CalibrationData m_depthCalibration;

	//! debug stuff
	std::vector<ColorImageR8G8B8A8> m_colorImages;
	std::vector<DepthImage32>		m_depthImages;
	std::vector<mat4f>				m_referenceTrajectory;

	CUDACachedFrame*			 d_cachedFrames;
	std::vector<CUDACachedFrame> m_cachedFrames;
	mat4f m_intrinsicsDownsampled;
};