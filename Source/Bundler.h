#pragma once

#include "SubmapManager.h"
#include "SBA.h"

#include "RGBDSensor.h"
#include "BinaryDumpReader.h"

class CUDAImageManager;
class SIFTImageManager;
class CUDACache;

class SiftGPU;
class SiftMatchGPU;

class Bundler
{
public:
	Bundler() {
		m_CudaImageManager = NULL;
		m_sift = NULL;
		m_siftMatcher = NULL;
		m_submapSize = 0;
	}
	~Bundler() {
		destroy();
	}
	
	void init(RGBDSensor* sensor);

	// sensor is only for debug point cloud record
	bool process(RGBDSensor* sensor);



	void evaluateTimings() {
		m_SubmapManager.evaluateTimings();
		m_SparseBundler.evaluateSolverTimings();
	}

	//! debug vis functions
	void printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame);
	void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered,
		unsigned int frameStart, unsigned int frameSkip);
	void saveKeysToPointCloud(RGBDSensor* sensor, const std::string& filename = "refined.ply");

private:
	void destroy();
	void matchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, unsigned int frameStart, unsigned int frameSkip, bool print = false);

	void solve(float4x4* transforms, SIFTImageManager* siftManager, bool isLocal);

	void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered);

	CUDAImageManager*		m_CudaImageManager;
	SubmapManager			m_SubmapManager;
	SBA						m_SparseBundler;

	SiftGPU*				m_sift;
	SiftMatchGPU*			m_siftMatcher;

	unsigned int			m_submapSize;

	static Timer			s_timer;
};


