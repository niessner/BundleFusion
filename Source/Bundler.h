#pragma once

#include "SiftGPU/SiftCameraParams.h"
#include "SubmapManager.h"
#include "SBA.h"

#include "RGBDSensor.h"
#include "BinaryDumpReader.h"
#include "TrajectoryManager.h"

class CUDAImageManager;
class SIFTImageManager;
class CUDACache;

class SiftGPU;
class SiftMatchGPU;

class Bundler
{
public:
	struct BundlerState {
		int						m_localToSolve;		// index of local submap to solve (-1) if none
		int						m_lastLocalSolved; // to check if can fuse to global
		bool					m_bOptimizeGlobal; // ready to optimize global

		//TODO do we really need
		unsigned int			m_lastFrameProcessed;
		bool					m_bLastFrameValid;

		unsigned int			m_lastNumLocalFrames;
		unsigned int			m_numCompleteTransforms;
		unsigned int			m_lastValidCompleteTransform;

		BundlerState() {
			m_localToSolve = -1;
			m_bOptimizeGlobal = false;
			m_lastLocalSolved = -1;

			m_lastFrameProcessed = 0;
			m_bLastFrameValid = false;
			m_lastNumLocalFrames = 0;
			m_numCompleteTransforms = 0;
			m_lastValidCompleteTransform = 0;
		}
	};


	Bundler(RGBDSensor* sensor, CUDAImageManager* imageManager);
	~Bundler();


	//! takes last frame from imageManager: runs sift and sift matching (including filtering)
	void processInput();

	//! returns the last frame-to-frame aligned matrix; could be invalid
	bool getCurrentIntegrationFrame(mat4f& siftTransform, unsigned int& frameIdx);
	//! optimize current local submap (nextLocal in submapManager)
	void optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations);
	//! optimize global keys
	void optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations);

	//! global key fuse and sift matching/filtering
	void processGlobal();



	//bool process();


	//void evaluateTimings() {
	//	m_SubmapManager.evaluateTimings();
	//	m_SparseBundler.evaluateSolverTimings();
	//}
	void saveConvergence(const std::string& filename) {
		m_SparseBundler.printConvergence(filename);
	}

	//! debug vis functions
	void printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame) const;
	void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered,
		unsigned int frameStart, unsigned int frameSkip) const;
	void saveKeysToPointCloud(const std::string& filename = "refined.ply") const;
	//void saveDEBUG();

	void saveCompleteTrajectory(const std::string& filename) const;
	void saveSiftTrajectory(const std::string& filename) const;
	void saveIntegrateTrajectory(const std::string& filename);

	TrajectoryManager* getTrajectoryManager() {
		return m_trajectoryManager;
	}

	bool hasProcssedInputFrame() const {
		return m_bHasProcessedInputFrame;
	}

	void setProcessedInputFrame() {
		m_bHasProcessedInputFrame = true;
	}

	void confirmProcessedInputFrame() {
		m_bHasProcessedInputFrame = false;
	}

	void exitBundlingThread() {
		m_bExitBundlingThread = true;
	}
	bool getExitBundlingThread() const {
		return m_bExitBundlingThread;
	}

private:
	bool m_bHasProcessedInputFrame;
	bool m_bExitBundlingThread;

	void matchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, unsigned int frameStart, unsigned int frameSkip, bool print = false);

	void solve(float4x4* transforms, SIFTImageManager* siftManager, unsigned int numNonLinIters, unsigned int numLinIters, bool isLocal, bool recordConvergence);

	void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered) const;

	void initCurrFrameBuffers();
	void getCurrentFrame();

	CUDAImageManager*		m_CudaImageManager;		//managed outside
	RGBDSensor*				m_RGBDSensor;			//managed outside

	SubmapManager			m_SubmapManager;
	SBA						m_SparseBundler;

	SiftGPU*				m_sift;
	SiftMatchGPU*			m_siftMatcher;
	TrajectoryManager*		m_trajectoryManager;

	unsigned int			m_submapSize;

	//tmp buffers from cuda image manager
	unsigned int			m_inputDepthWidth, m_inputDepthHeight;
	unsigned int			m_inputColorWidth, m_inputColorHeight;
	unsigned int			m_widthSIFT, m_heightSIFT;
	float*					d_inputDepth;
	uchar4*					d_inputColor;
	float*					d_intensitySIFT;
	mat4f					m_SIFTIntrinsics;
	mat4f					m_SIFTIntrinsicsInv;

	static Timer			s_timer;

	SiftCameraParams		m_siftCameraParams;

	// state of processing/optimization
	BundlerState			m_currentState;
};


