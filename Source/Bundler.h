#pragma once

#include "SiftGPU/SiftCameraParams.h"
#include "SubmapManager.h"

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
		float*					d_inputDepthFilt, *d_inputDepthRaw;
		uchar4*					d_inputColor;
		float*					d_intensitySIFT;
		mat4f					m_SIFTIntrinsics;
		mat4f					m_SIFTIntrinsicsInv;
		float*					d_intensityFilterHelper;

		bool m_bFilterDepthValues;
		float m_fBilateralFilterSigmaD, m_fBilateralFilterSigmaR;

		BundlerInputData() {
			m_inputDepthWidth = 0;	m_inputDepthHeight = 0;
			m_inputColorWidth = 0;	m_inputColorHeight = 0;
			m_widthSIFT = 0;		m_heightSIFT = 0;
			d_inputDepthFilt = NULL;d_inputDepthRaw = NULL;
			d_inputColor = NULL;
			d_intensitySIFT = NULL;
			d_intensityFilterHelper = NULL;
		}
		void alloc(const RGBDSensor* sensor) {
			m_inputDepthWidth = sensor->getDepthWidth();
			m_inputDepthHeight = sensor->getDepthHeight();
			m_inputColorWidth = sensor->getColorWidth();
			m_inputColorHeight = sensor->getColorHeight();
			m_widthSIFT = GlobalBundlingState::get().s_widthSIFT;
			m_heightSIFT = GlobalBundlingState::get().s_heightSIFT;
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputDepthFilt, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputDepthRaw, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityFilterHelper, sizeof(float)*m_widthSIFT*m_heightSIFT));
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
			MLIB_CUDA_SAFE_CALL(cudaFree(d_inputDepthFilt));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_inputDepthRaw));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_inputColor));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_intensitySIFT));
			MLIB_CUDA_SAFE_CALL(cudaFree(d_intensityFilterHelper));
		}
	};
	struct BundlerState {
		enum PROCESS_STATE {
			DO_NOTHING,
			PROCESS,
			INVALIDATE
		};
		//shared
		int						m_localToSolve;		// index of local submap to solve (-1) if none

		//only in process input
		unsigned int			m_lastFrameProcessed;
		bool					m_bLastFrameValid;

		//only in opt
		int						m_lastLocalSolved; // to check if can fuse to global
		PROCESS_STATE			m_bOptimizeGlobal; // ready to optimize global

		//unsigned int			m_lastNumLocalFrames;
		unsigned int			m_numCompleteTransforms;
		unsigned int			m_lastValidCompleteTransform;

		PROCESS_STATE			m_bProcessGlobal;

		unsigned int			m_totalNumOptLocalFrames;

		//shared but constant
		static int				s_markOffset;

		BundlerState() {
			m_localToSolve = -1;
			m_bOptimizeGlobal = DO_NOTHING;
			m_lastLocalSolved = -1;

			m_lastFrameProcessed = 0;
			m_bLastFrameValid = false;
			//m_lastNumLocalFrames = 0;
			m_numCompleteTransforms = 0;
			m_lastValidCompleteTransform = 0;
			m_bProcessGlobal = DO_NOTHING;

			m_totalNumOptLocalFrames = 0;
		}
	};


	Bundler(const RGBDSensor* sensor, const CUDAImageManager* imageManager);
	~Bundler();


	//! takes last frame from imageManager: runs sift and sift matching (including filtering)
	void processInput();

	void prepareLocalSolve(unsigned int curFrame, bool isLastFrame = false);

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
	void saveConvergence(const std::string& filename) const {
		m_SubmapManager.printConvergence(filename);
	}

	//! debug functions only call at end
	void saveCompleteTrajectory(const std::string& filename) const;
	void saveSiftTrajectory(const std::string& filename) const;
	void saveIntegrateTrajectory(const std::string& filename);
	void saveGlobalSiftManagerAndCacheToFile(const std::string& prefix) const { m_SubmapManager.saveGlobalSiftManagerAndCache(prefix); }
#ifdef DEBUG_PRINT_MATCHING
	void saveLogsToFile() const {
		m_SubmapManager.saveLogImImCorrsToFile("debug/_logs/log");
	}
#endif

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

	unsigned int getCurrProcessedFrame() const {
		return m_currentState.m_lastFrameProcessed;
	}
	unsigned int getNumProcessedFrames() const {
		return m_currentState.m_lastFrameProcessed + 1;
	}

	unsigned int getSubMapSize() const {
		return m_submapSize;
	}

	void setScanDoneGlobalOpt() {
		m_bIsScanDoneGlobalOpt = true;
	}

	//! fake finish local opt without actually optimizing anything
	void resetDEBUG(bool updateTrajectory) {
		m_currentState.m_localToSolve = -1;
		unsigned int curFrame = (m_currentState.m_lastLocalSolved < 0) ? (unsigned int)-1 : m_currentState.m_totalNumOptLocalFrames;
		m_SubmapManager.resetDEBUG(updateTrajectory && m_currentState.m_bProcessGlobal == BundlerState::PROCESS, m_currentState.m_lastLocalSolved, curFrame);
	}

private:
	bool m_bHasProcessedInputFrame;
	bool m_bExitBundlingThread;
	bool m_bIsScanDoneGlobalOpt;

	void getCurrentFrame();

	const CUDAImageManager*	m_CudaImageManager;		//managed outside
	const RGBDSensor*		m_RGBDSensor;			//managed outside

	SubmapManager			m_SubmapManager;

	TrajectoryManager*		m_trajectoryManager;

	unsigned int			m_submapSize;

	//tmp buffers from cuda image manager
	BundlerInputData		m_bundlerInputData;

	static Timer			s_timer;

	SiftCameraParams		m_siftCameraParams;

	// state of processing/optimization
	BundlerState			m_currentState;

	unsigned int			m_numFramesPastLast;
	unsigned int			m_numOptPerResidualRemoval;

	bool m_useSolve;
};


