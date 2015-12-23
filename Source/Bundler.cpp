#include "stdafx.h"
#include "CUDAImageManager.h"
#include "CUDACache.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/MatrixConversion.h"
#include "ImageHelper.h"
#include "CUDAImageUtil.h"

#include "mLibCuda.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"
#include "Bundler.h"


extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params);

Timer Bundler::s_timer;
Timer Bundler::s_timerOpt;
int Bundler::BundlerState::s_markOffset = 2;


Bundler::Bundler(RGBDSensor* sensor, CUDAImageManager* imageManager)
{
	m_CudaImageManager = imageManager;
	m_RGBDSensor = sensor;

	// init CUDA
	m_bundlerInputData.alloc(m_RGBDSensor);

	m_submapSize = GlobalBundlingState::get().s_submapSize;
	m_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, m_submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage,
		m_submapSize, m_CudaImageManager);
	//TODO fix
	if (GlobalAppState::get().s_sensorIdx == 3) {
		m_SubmapManager.setTotalNumFrames(((BinaryDumpReader*)m_RGBDSensor)->getNumTotalFrames());
	}

	m_trajectoryManager = new TrajectoryManager(GlobalBundlingState::get().s_maxNumImages * m_submapSize);

	// init sift camera constant params
	m_siftCameraParams.m_depthWidth = m_bundlerInputData.m_inputDepthWidth;
	m_siftCameraParams.m_depthHeight = m_bundlerInputData.m_inputDepthHeight;
	m_siftCameraParams.m_intensityWidth = m_bundlerInputData.m_widthSIFT;
	m_siftCameraParams.m_intensityHeight = m_bundlerInputData.m_heightSIFT;
	m_siftCameraParams.m_siftIntrinsics = MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsics);
	m_siftCameraParams.m_siftIntrinsicsInv = MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsicsInv);
	m_SubmapManager.getCacheIntrinsics(m_siftCameraParams.m_downSampIntrinsics, m_siftCameraParams.m_downSampIntrinsicsInv);
	m_siftCameraParams.m_minKeyScale = GlobalBundlingState::get().s_minKeyScale;
	updateConstantSiftCameraParams(m_siftCameraParams);

	m_bHasProcessedInputFrame = false;
	m_bExitBundlingThread = false;
	m_bIsScanDoneGlobalOpt = false;
}

Bundler::~Bundler()
{
	SAFE_DELETE(m_trajectoryManager);
}

void Bundler::processInput()
{
	const unsigned int curFrame = m_CudaImageManager->getCurrFrameNumber();
	if (curFrame > 0 && m_currentState.m_lastFrameProcessed == curFrame) { // special case the last local solve (needs to run once)
#ifdef RUN_MULTITHREADED 
		if (m_RGBDSensor->isReceivingFrames()) { //debugging
			std::cout << "WHY IS processInput called on same frame but still receiving frames???" << std::endl;
			getchar();
		}
		MLIB_ASSERT(!m_RGBDSensor->isReceivingFrames());
#endif
		static unsigned int framePastLast = 0;
		if (framePastLast == 0 && m_currentState.m_localToSolve == -1) {
			if (!m_SubmapManager.isLastLocalFrame(curFrame)) prepareLocalSolve(curFrame, true);
			framePastLast++;
		}
		else {
			if (framePastLast == 10) {
#ifdef USE_GLOBAL_DENSE_AT_END
				m_SubmapManager.setEndSolveGlobalDenseWeights();
				//saveGlobalSiftManagerAndCacheToFile("debug/global");
				//saveCompleteTrajectory("debug/curTrajectory.bin");
				//std::cout << "waiting..." << std::endl;
				//getchar();
#endif
			}
			framePastLast++;
		}
		return; // nothing new to process
	}

	if (GlobalBundlingState::get().s_verbose) std::cout << "[ frame " << curFrame << " ]" << std::endl;

	getCurrentFrame();

	// run SIFT & process cuda cache
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
	const unsigned int curLocalFrame = m_SubmapManager.runSIFT(curFrame, m_bundlerInputData.d_intensitySIFT, m_bundlerInputData.d_inputDepth,
		m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight, m_bundlerInputData.d_inputColor, m_bundlerInputData.m_inputColorWidth, m_bundlerInputData.m_inputColorHeight);
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(true).timeSiftDetection = s_timer.getElapsedTimeMS(); }

	// match with every other local
	m_currentState.m_bLastFrameValid = 1;
	if (curLocalFrame > 0) {
		//const bool debugLocalMatching = false;
		//const unsigned int stop = 140;
		//if (debugLocalMatching && curFrame > stop && curFrame <= stop+m_submapSize) {
		//	m_SubmapManager.setPrintMatchesDEBUG(true);
		//}
		m_currentState.m_bLastFrameValid = m_SubmapManager.localMatchAndFilter(MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsicsInv));
		//if (debugLocalMatching && curFrame > stop && curFrame <= stop+m_submapSize) {
		//	m_SubmapManager.setPrintMatchesDEBUG(false);
		//	if (curFrame == stop + m_submapSize) {
		//		std::cout << "waiting..." << std::endl;
		//		getchar();
		//	}
		//}

		m_SubmapManager.computeCurrentSiftTransform(curFrame, curLocalFrame, m_currentState.m_lastValidCompleteTransform);
	}
	m_currentState.m_lastFrameProcessed = curFrame;

	// global frame
	if (m_SubmapManager.isLastLocalFrame(curFrame)) { // global frame
		prepareLocalSolve(curFrame);
	} // global
}

bool Bundler::getCurrentIntegrationFrame(mat4f& siftTransform, unsigned int& frameIdx)
{
	if (m_currentState.m_bLastFrameValid) {
		//cutilSafeCall(cudaMemcpy(&siftTransform, m_SubmapManager.getCurrIntegrateTransform(m_currentState.m_lastFrameProcessed), sizeof(float4x4), cudaMemcpyDeviceToHost));	//TODO MT needs to be copied from the other GPU...
		siftTransform = m_SubmapManager.getCurrentIntegrateTransform(m_currentState.m_lastFrameProcessed);
		frameIdx = m_currentState.m_lastFrameProcessed;
		//m_trajectoryManager->addFrame(TrajectoryManager::TrajectoryFrame::Integrated, siftTransform, m_currentState.m_lastFrameProcessed);
		return true;
	}
	else {
		//m_trajectoryManager->addFrame(TrajectoryManager::TrajectoryFrame::NotIntegrated_NoTransform, mat4f::zero(), m_currentState.m_lastFrameProcessed);
		return false;
	}
}

void Bundler::optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	if (m_currentState.m_localToSolve == -1) {
		return; // nothing to solve
	}

	unsigned int curNumLocalFrames = m_SubmapManager.getNumNextLocalFrames();
	//m_currentState.m_lastNumLocalFrames = m_SubmapManager.getNumNextLocalFrames();
	m_currentState.m_bProcessGlobal = BundlerState::DO_NOTHING;
	unsigned int currLocalIdx;
	if (m_currentState.m_localToSolve >= 0) {
		currLocalIdx = m_currentState.m_localToSolve;

		bool valid = m_SubmapManager.optimizeLocal(currLocalIdx, numNonLinIterations, numLinIterations);
		if (valid) m_currentState.m_bProcessGlobal = BundlerState::PROCESS;
		else m_currentState.m_bProcessGlobal = BundlerState::INVALIDATE;
	}
	else {
		currLocalIdx = -m_currentState.m_localToSolve - m_currentState.s_markOffset;
		m_currentState.m_bProcessGlobal = BundlerState::INVALIDATE; //invalidate
	}
	m_currentState.m_localToSolve = -1;
	m_currentState.m_lastLocalSolved = currLocalIdx;
	m_currentState.m_totalNumOptLocalFrames = m_submapSize * m_currentState.m_lastLocalSolved + curNumLocalFrames; //last local solved is 0-indexed so this doesn't overcount
}


void Bundler::processGlobal()
{
	if (m_currentState.m_bProcessGlobal == BundlerState::DO_NOTHING) {
		if (!m_RGBDSensor->isReceivingFrames()) m_currentState.m_bOptimizeGlobal = BundlerState::PROCESS;
		return;
	}

	if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addGlobalFrameTiming();
	if (m_currentState.m_bProcessGlobal == BundlerState::PROCESS) {
		m_currentState.m_bOptimizeGlobal = (BundlerState::PROCESS_STATE)m_SubmapManager.computeAndMatchGlobalKeys(m_currentState.m_lastLocalSolved,
			MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsics), MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsicsInv));
		//printKey("debug/keysGlobal/key" + std::to_string(m_currentState.m_lastLocalSolved) + ".png", m_currentState.m_lastLocalSolved*m_submapSize, m_SubmapManager.getGlobalDEBUG(), m_currentState.m_lastLocalSolved);
	}
	else {
		// cache
		m_SubmapManager.incrementGlobalCache();
		m_currentState.m_bOptimizeGlobal = BundlerState::INVALIDATE;

		//getchar();
		m_SubmapManager.invalidateImages(m_submapSize * m_currentState.m_lastLocalSolved, m_currentState.m_totalNumOptLocalFrames);
		//add invalidated (fake) global frame
		m_SubmapManager.addInvalidGlobalKey();
	}
	m_currentState.m_bProcessGlobal = BundlerState::DO_NOTHING;
}



void Bundler::optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations, bool isStart /*= true*/, bool isEnd /*= true*/)
{
	if (m_currentState.m_bOptimizeGlobal == BundlerState::DO_NOTHING) {
		return; // nothing to solve
	}

	MLIB_ASSERT(m_currentState.m_lastLocalSolved >= 0);
	unsigned int numFrames = m_currentState.m_totalNumOptLocalFrames;

	if (m_currentState.m_bOptimizeGlobal == BundlerState::PROCESS) {
		bool valid = m_SubmapManager.optimizeGlobal(numFrames, numNonLinIterations, numLinIterations, isStart, isEnd, m_bIsScanDoneGlobalOpt);

		if (isEnd) {
			m_SubmapManager.updateTrajectory(numFrames);
			m_trajectoryManager->updateOptimizedTransform(m_SubmapManager.getCompleteTrajectory(), numFrames);
			m_currentState.m_numCompleteTransforms = numFrames;
			if (valid) m_currentState.m_lastValidCompleteTransform = m_submapSize * m_currentState.m_lastLocalSolved; //TODO over-conservative but easier
			m_currentState.m_bOptimizeGlobal = BundlerState::DO_NOTHING;
		}
	}
	else {
		if (isStart) {
			m_SubmapManager.invalidateLastGlobalFrame();
			m_SubmapManager.updateTrajectory(numFrames);
			m_trajectoryManager->updateOptimizedTransform(m_SubmapManager.getCompleteTrajectory(), numFrames);
			m_currentState.m_numCompleteTransforms = numFrames;

			m_currentState.m_bOptimizeGlobal = BundlerState::DO_NOTHING;
		}
	}

}

void Bundler::saveCompleteTrajectory(const std::string& filename) const
{
	m_SubmapManager.saveCompleteTrajectory(filename, m_currentState.m_numCompleteTransforms);
}

void Bundler::saveSiftTrajectory(const std::string& filename) const
{
	m_SubmapManager.saveSiftTrajectory(filename, m_currentState.m_numCompleteTransforms);
}

void Bundler::saveIntegrateTrajectory(const std::string& filename)
{
	const std::vector<mat4f>& integrateTrajectory = m_SubmapManager.getAllIntegrateTransforms();
	std::vector<mat4f> saveIntegrateTrajectory(integrateTrajectory.begin(), integrateTrajectory.begin() + m_currentState.m_numCompleteTransforms);
	BinaryDataStreamFile s(filename, true);
	s << saveIntegrateTrajectory;
	s.closeStream();
}

void Bundler::getCurrentFrame()
{
	m_CudaImageManager->copyToBundling(m_bundlerInputData.d_inputDepth, m_bundlerInputData.d_inputColor);
	CUDAImageUtil::resampleToIntensity(m_bundlerInputData.d_intensitySIFT, m_bundlerInputData.m_widthSIFT, m_bundlerInputData.m_heightSIFT,
		m_bundlerInputData.d_inputColor, m_bundlerInputData.m_inputColorWidth, m_bundlerInputData.m_inputColorHeight);

	//if (GlobalBundlingState::get().s_erodeSIFTdepth) {
	//	unsigned int numIter = 2;

	//	numIter = 2 * ((numIter + 1) / 2);
	//	for (unsigned int i = 0; i < numIter; i++) {
	//		if (i % 2 == 0) {
	//			CUDAImageUtil::erodeDepthMap(m_bundlerInputData.d_depthErodeHelper, m_bundlerInputData.d_inputDepth, 3,
	//				m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight, 0.05f, 0.3f);
	//		}
	//		else {
	//			CUDAImageUtil::erodeDepthMap(m_bundlerInputData.d_inputDepth, m_bundlerInputData.d_depthErodeHelper, 3,
	//				m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight, 0.05f, 0.3f);
	//		}
	//	}
	//}
	//if (m_bundlerInputData.m_bFilterDepthValues) {
	//	CUDAImageUtil::gaussFilterFloatMap(m_bundlerInputData.d_depthErodeHelper, m_bundlerInputData.d_inputDepth,
	//		m_bundlerInputData.m_fBilateralFilterSigmaD, m_bundlerInputData.m_fBilateralFilterSigmaR,
	//		m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight);
	//	std::swap(m_bundlerInputData.d_inputDepth, m_bundlerInputData.d_depthErodeHelper);
	//}
}

void Bundler::saveDEBUG()
{
	m_SubmapManager.saveVerifyDEBUG("debug/");
}

void Bundler::prepareLocalSolve(unsigned int curFrame, bool isLastFrame /*= false*/)
{
	unsigned int curLocalIdx = m_SubmapManager.getCurrLocal(curFrame);
	if (isLastFrame && (curFrame % m_submapSize) == 0) { // only the overlap frame
		// invalidate
		curLocalIdx++;
		m_currentState.m_localToSolve = -((int)curLocalIdx + m_currentState.s_markOffset);
		if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: last local submap 1 frame -> invalidating" << curFrame << std::endl;
	}

	// if valid local
	if (m_SubmapManager.isCurrentLocalValidChunk()) {
		// ready to solve local
		MLIB_ASSERT(m_currentState.m_localToSolve == -1);
		m_currentState.m_localToSolve = curLocalIdx;
	}
	else {
		// invalidate the local
		m_currentState.m_localToSolve = -((int)curLocalIdx + m_currentState.s_markOffset);
		if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: invalid local submap " << curFrame << std::endl;
	}

	// switch local submaps
	m_SubmapManager.switchLocal();
}

