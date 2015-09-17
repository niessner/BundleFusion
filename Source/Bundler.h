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
	Bundler(RGBDSensor* sensor, CUDAImageManager* imageManager);
	~Bundler();
	

	//! takes last frame from imageManager: runs sift and sift matching (including filtering)
	void processInput() {}

	struct A {
		const float* depth;
		const uchar4* color;
		mat4f transform;
	};

	//! returns the last frame-to-frame aligned matrix; could be invalid
	bool getCurrentIntegrationFrame(mat4f& siftTransform, const float* & d_depth, const uchar4* & d_color) {
		siftTransform.setIdentity();
		d_depth = m_CudaImageManager->getLastIntegrateDepth();
		d_color = m_CudaImageManager->getLastIntegrateColor();
		return true;
	}
	//! optimize current local submap (nextLocal in submapManager)
	void optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations) {}
	//! optimize global keys
	void optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations) {}



	bool process();


	void evaluateTimings() {
		m_SubmapManager.evaluateTimings();
		m_SparseBundler.evaluateSolverTimings();
	}

	//! debug vis functions
	void printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame);
	void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered,
		unsigned int frameStart, unsigned int frameSkip);
	void saveKeysToPointCloud(const std::string& filename = "refined.ply");



private:
	void matchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, unsigned int frameStart, unsigned int frameSkip, bool print = false);

	void solve(float4x4* transforms, SIFTImageManager* siftManager, bool isLocal);

	void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered);

	CUDAImageManager*		m_CudaImageManager;		//managed outside
	RGBDSensor*				m_RGBDSensor;			//managed outside

	SubmapManager			m_SubmapManager;
	SBA						m_SparseBundler;

	SiftGPU*				m_sift;
	SiftMatchGPU*			m_siftMatcher;

	unsigned int			m_submapSize;

	bool					m_bOptimizeLocal; // ready to optimize local
	bool					m_bOptimizeGlobal; // ready to optimize global

	static Timer			s_timer;
};


