#pragma once
#include "OnlineBundlerHelper.h"


class Bundler;
class RGBDSensor;
class CUDAImageManager;
class TrajectoryManager;

class OnlineBundler {
public:
	OnlineBundler(const RGBDSensor* sensor, const CUDAImageManager* imageManager);
	~OnlineBundler();


	bool getCurrentIntegrationFrame(mat4f& siftTransform, unsigned int& frameIdx, bool& bGlobalTrackingLost);

	//feature detect/match for current frame
	void processInput();

	//local opt and global match/opt
	void process(unsigned int numNonLinItersLocal, unsigned int numLinItersLocal, unsigned int numNonLinItersGlobal, unsigned int numLinItersGlobal);

	TrajectoryManager* getTrajectoryManager()	{ return m_trajectoryManager; }
	bool hasProcssedInputFrame() const			{ return m_bHasProcessedInputFrame; }
	void setProcessedInputFrame()				{ m_bHasProcessedInputFrame = true; }
	void confirmProcessedInputFrame()			{ m_bHasProcessedInputFrame = false; }
	void exitBundlingThread()					{ m_bExitBundlingThread = true; }
	bool getExitBundlingThread() const			{ return m_bExitBundlingThread; }

	unsigned int getCurrProcessedFrame() const	{ return m_state.m_lastFrameProcessed; }

	// -- various logging
	void saveGlobalSparseCorrsToFile(const std::string& filename) const;

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	void finishCorrespondenceEvaluatorLogging();
#endif

private:

	bool isLastLocalFrame(unsigned int curFrame) const { return (curFrame >= m_submapSize && (curFrame % m_submapSize) == 0); }
	void getCurrentFrame();
	void computeCurrentSiftTransform(bool bIsValid, unsigned int frameIdx, unsigned int localFrameIdx, unsigned int lastValidCompleteTransform);

	void prepareLocalSolve(unsigned int curFrame, bool isSequenceEnd);
	void initializeNextGlobalTransform(unsigned int lastMatchedIdx, unsigned int lastValidLocal);

	void processGlobal();
	void optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations);
	void optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations);

	void updateTrajectory(unsigned int curFrame);
	void invalidateImages(unsigned int startFrame, unsigned int endFrame = -1) {
		if (endFrame == -1) m_invalidImagesList[startFrame] = 0;
		else {
			for (unsigned int i = startFrame; i < endFrame; i++)
				m_invalidImagesList[i] = 0;
		}
	}
	void validateImages(unsigned int startFrame, unsigned int endFrame = -1) {
		if (endFrame == -1) m_invalidImagesList[startFrame] = 1;
		else {
			for (unsigned int i = startFrame; i < endFrame; i++)
				m_invalidImagesList[i] = 1;
		}
	}

	//*********** for interfacing with recon ************
	bool m_bHasProcessedInputFrame;
	bool m_bExitBundlingThread;
	const CUDAImageManager*		m_cudaImageManager; //managed outside

	//*********** input data ************
	BundlerInputData			m_input;
	unsigned int				m_submapSize;

	BundlerState				m_state;

	//*********** local/global ************
	Bundler*					m_local;
	Bundler*					m_optLocal;
	Bundler*					m_global;

	std::mutex					mutex_optLocal;
	std::mutex					mutex_siftMatcher; //TODO why can't this run multithreaded??
	unsigned int				m_numOptPerResidualRemoval;

	//*********** TRAJECTORIES ************
	TrajectoryManager*			m_trajectoryManager;

	std::mutex					mutex_completeTrajectory;
	float4x4*					d_completeTrajectory;
	float4x4*					d_localTrajectories;
	std::vector<std::vector<int>> m_localTrajectoriesValid;

	float4x4*					d_siftTrajectory; // frame-to-frame sift tracking for all frames in sequence
	//************************************

	std::vector<unsigned int>	m_invalidImagesList;   //cumulative over global and local
	int*						d_imageInvalidateList; // for updateTrajectory

	float4x4*					d_currIntegrateTransform;
	std::vector<mat4f>			m_currIntegrateTransform;

	Timer						m_timer;
};
