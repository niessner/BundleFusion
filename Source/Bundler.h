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

class Bundler
{
public:
	struct BundlerInputData {
		unsigned int			m_inputDepthWidth, m_inputDepthHeight;
		unsigned int			m_inputColorWidth, m_inputColorHeight;
		unsigned int			m_widthSIFT, m_heightSIFT;
		float*					d_inputDepth;
		uchar4*					d_inputColor;
		float*					d_intensitySIFT;
		mat4f					m_SIFTIntrinsics;
		mat4f					m_SIFTIntrinsicsInv;
		float*					d_depthErodeHelper;

		bool m_bFilterDepthValues;
		float m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR;

		BundlerInputData() {
			m_inputDepthWidth = 0;	m_inputDepthHeight = 0;
			m_inputColorWidth = 0;	m_inputColorHeight = 0;
			m_widthSIFT = 0;		m_heightSIFT = 0;
			d_inputDepth = NULL;
			d_inputColor = NULL;
			d_intensitySIFT = NULL;
			d_depthErodeHelper = NULL;
		}
		void alloc(const RGBDSensor* sensor) {
			m_inputDepthWidth = sensor->getDepthWidth();
			m_inputDepthHeight = sensor->getDepthHeight();
			m_inputColorWidth = sensor->getColorWidth();
			m_inputColorHeight = sensor->getColorHeight();
			m_widthSIFT = GlobalBundlingState::get().s_widthSIFT;
			m_heightSIFT = GlobalBundlingState::get().s_heightSIFT;
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputDepth, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthErodeHelper, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputColor, sizeof(uchar4)*m_inputColorWidth*m_inputColorHeight));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensitySIFT, sizeof(float)*m_widthSIFT*m_heightSIFT));

			const float scaleWidthSIFT = (float)m_widthSIFT / (float)m_inputColorWidth;
			const float scaleHeightSIFT = (float)m_heightSIFT / (float)m_inputColorHeight;
			m_SIFTIntrinsics = sensor->getColorIntrinsics();
			m_SIFTIntrinsics._m00 *= scaleWidthSIFT;  m_SIFTIntrinsics._m02 *= scaleWidthSIFT;
			m_SIFTIntrinsics._m11 *= scaleHeightSIFT; m_SIFTIntrinsics._m12 *= scaleHeightSIFT;
			m_SIFTIntrinsicsInv = sensor->getColorIntrinsicsInv();
			m_SIFTIntrinsicsInv._m00 /= scaleWidthSIFT; m_SIFTIntrinsicsInv._m11 /= scaleHeightSIFT;

			m_bFilterDepthValues = GlobalBundlingState::get().s_depthFilter;
			m_fBilateralFilterSigmaR = GlobalBundlingState::get().s_depthSigmaR;
			m_fBilateralFilterSigmaD = GlobalBundlingState::get().s_depthSigmaD;
		}
		~BundlerInputData() {
			MLIB_CUDA_SAFE_CALL(cudaFree(d_inputDepth));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_inputColor));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_intensitySIFT));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_depthErodeHelper));
		}
	};
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

		bool					m_bProcessGlobal;

		BundlerState() {
			m_localToSolve = -1;
			m_bOptimizeGlobal = false;
			m_lastLocalSolved = -1;

			m_lastFrameProcessed = 0;
			m_bLastFrameValid = false;
			m_lastNumLocalFrames = 0;
			m_numCompleteTransforms = 0;
			m_lastValidCompleteTransform = 0;
			m_bProcessGlobal = false;
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
	void optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations, bool isStart = true, bool isEnd = true);

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

	unsigned int getNumProcessedFrames() const {
		return m_currentState.m_lastFrameProcessed;
	}

	unsigned int getSubMapSize() const {
		return m_submapSize;
	}

	void setScanDoneGlobalOpt() {
		m_bIsScanDoneGlobalOpt = true;
	}

private:
	bool m_bHasProcessedInputFrame;
	bool m_bExitBundlingThread;
	bool m_bIsScanDoneGlobalOpt;

	void solve(float4x4* transforms, SIFTImageManager* siftManager, unsigned int numNonLinIters, unsigned int numLinIters, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt);

	void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered) const;

	void getCurrentFrame();

	CUDAImageManager*		m_CudaImageManager;		//managed outside
	RGBDSensor*				m_RGBDSensor;			//managed outside

	SubmapManager			m_SubmapManager;
	SBA						m_SparseBundler;

	TrajectoryManager*		m_trajectoryManager;

	unsigned int			m_submapSize;

	//tmp buffers from cuda image manager
	BundlerInputData		m_bundlerInputData;

	static Timer			s_timer;
	static Timer			s_timerOpt;

	SiftCameraParams		m_siftCameraParams;

	// state of processing/optimization
	BundlerState			m_currentState;
};


