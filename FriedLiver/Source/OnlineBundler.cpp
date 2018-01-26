
#include "stdafx.h"
#include "OnlineBundler.h"

#include "RGBDSensor.h"
#include "CUDAImageManager.h"
#include "Bundler.h"
#include "TrajectoryManager.h"
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
#include "SensorDataReader.h"
#endif

#include "SiftGPU/SiftCameraParams.h"
#include "SiftGPU/MatrixConversion.h"

extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params);

extern "C" void computeSiftTransformCU(const float4x4* d_currFilteredTransformsInv, const int* d_currNumFilteredMatchesPerImagePair,
	const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform,
	float4x4* d_siftTrajectory, unsigned int curFrameIndexAll, unsigned int curFrameIndex, float4x4* d_currIntegrateTrans);
extern "C" void initNextGlobalTransformCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, unsigned int initGlobalIdx,
	float4x4* d_localTrajectories, unsigned int lastValidLocal, unsigned int numLocalTransformsPerTrajectory);
extern "C" void updateTrajectoryCU(
	const float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	const float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory, unsigned int numLocalTrajectories,
	int* d_imageInvalidateList);

#define ID_MARK_OFFSET 2

OnlineBundler::OnlineBundler(const RGBDSensor* sensor, const CUDAImageManager* imageManager)
{
	//init input data
	m_cudaImageManager = imageManager;
	m_input.alloc(sensor);
	m_submapSize = GlobalBundlingState::get().s_submapSize;
	m_numOptPerResidualRemoval = GlobalBundlingState::get().s_numOptPerResidualRemoval;

	const unsigned int maxNumImages = GlobalBundlingState::get().s_maxNumImages;
	const unsigned int maxNumKeysPerImage = GlobalBundlingState::get().s_maxNumKeysPerImage;
	m_local = new Bundler(m_submapSize + 1, maxNumKeysPerImage, m_input.m_SIFTIntrinsicsInv, imageManager, true);
	m_optLocal = new Bundler(m_submapSize + 1, maxNumKeysPerImage, m_input.m_SIFTIntrinsicsInv, imageManager, true);
	m_global = new Bundler(maxNumImages, maxNumKeysPerImage, m_input.m_SIFTIntrinsicsInv, imageManager, false);

	// init sift camera constant params
	SiftCameraParams siftCameraParams;
	siftCameraParams.m_depthWidth = m_input.m_inputDepthWidth;
	siftCameraParams.m_depthHeight = m_input.m_inputDepthHeight;
	siftCameraParams.m_intensityWidth = m_input.m_widthSIFT;
	siftCameraParams.m_intensityHeight = m_input.m_heightSIFT;
	siftCameraParams.m_siftIntrinsics = MatrixConversion::toCUDA(m_input.m_SIFTIntrinsics);
	siftCameraParams.m_siftIntrinsicsInv = MatrixConversion::toCUDA(m_input.m_SIFTIntrinsicsInv);
	m_global->getCacheIntrinsics(siftCameraParams.m_downSampIntrinsics, siftCameraParams.m_downSampIntrinsicsInv);
	siftCameraParams.m_minKeyScale = GlobalBundlingState::get().s_minKeyScale;
	updateConstantSiftCameraParams(siftCameraParams);

	//trajectories
	m_trajectoryManager = new TrajectoryManager(maxNumImages * m_submapSize);
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_completeTrajectory, sizeof(float4x4)*maxNumImages*m_submapSize));

	std::vector<mat4f> identityTrajectory((m_submapSize + 1) * maxNumImages, mat4f::identity());
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_localTrajectories, sizeof(float4x4)*maxNumImages*(m_submapSize + 1)));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_localTrajectories, identityTrajectory.data(), sizeof(float4x4) * identityTrajectory.size(), cudaMemcpyHostToDevice));
	m_localTrajectoriesValid.resize(maxNumImages);

	float4x4 id; id.setIdentity();
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_siftTrajectory, sizeof(float4x4)*maxNumImages*m_submapSize));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_siftTrajectory, &id, sizeof(float4x4), cudaMemcpyHostToDevice)); // set first to identity

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currIntegrateTransform, sizeof(float4x4)*maxNumImages*m_submapSize));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currIntegrateTransform, &id, sizeof(float4x4), cudaMemcpyHostToDevice)); // set first to identity

	m_currIntegrateTransform.resize(maxNumImages*m_submapSize);
	m_currIntegrateTransform[0].setIdentity();

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_imageInvalidateList, sizeof(int)*maxNumImages*m_submapSize));
	m_invalidImagesList.resize(maxNumImages*m_submapSize, 1);

	m_bHasProcessedInputFrame = false;
	m_bExitBundlingThread = false;
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	if (GlobalAppState::get().s_sensorIdx != 8) throw MLIB_EXCEPTION("unable to evaluate sparse corrs for non sens-data input");
	std::vector<mat4f> trajectory; 
	{ // only want global trajectory
		std::vector<mat4f> completeTrajectory; 
		((SensorDataReader*)sensor)->getTrajectory(completeTrajectory);
		for (unsigned int i = 0; i < completeTrajectory.size(); i += m_submapSize) trajectory.push_back(completeTrajectory[i]);
	}
	m_global->initializeCorrespondenceEvaluator(trajectory, "debug/_corr-evaluation");
#endif
}

OnlineBundler::~OnlineBundler()
{
	SAFE_DELETE(m_local);
	SAFE_DELETE(m_optLocal);
	SAFE_DELETE(m_global);

	MLIB_CUDA_SAFE_FREE(d_completeTrajectory);
	MLIB_CUDA_SAFE_FREE(d_localTrajectories);
	MLIB_CUDA_SAFE_FREE(d_siftTrajectory);
	MLIB_CUDA_SAFE_FREE(d_currIntegrateTransform);
	MLIB_CUDA_SAFE_FREE(d_imageInvalidateList);
}

void OnlineBundler::getCurrentFrame()
{
	m_cudaImageManager->copyToBundling(m_input.d_inputDepthRaw, m_input.d_inputDepthFilt, m_input.d_inputColor);
	CUDAImageUtil::resampleToIntensity(m_input.d_intensitySIFT, m_input.m_widthSIFT, m_input.m_heightSIFT,
		m_input.d_inputColor, m_input.m_inputColorWidth, m_input.m_inputColorHeight);

	if (m_input.m_bFilterIntensity) {
		CUDAImageUtil::gaussFilterIntensity(m_input.d_intensityFilterHelper, m_input.d_intensitySIFT, m_input.m_intensitySigmaD, m_input.m_widthSIFT, m_input.m_heightSIFT);
		std::swap(m_input.d_intensityFilterHelper, m_input.d_intensitySIFT);
	}
}

void OnlineBundler::computeCurrentSiftTransform(bool bIsValid, unsigned int frameIdx, unsigned int localFrameIdx, unsigned int lastValidCompleteTransform)
{
	if (!bIsValid) {
		MLIB_ASSERT(frameIdx > 1);
		m_currIntegrateTransform[frameIdx].setZero(-std::numeric_limits<float>::infinity());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_siftTrajectory + frameIdx, d_siftTrajectory + frameIdx - 1, sizeof(float4x4), cudaMemcpyDeviceToDevice)); //set invalid
	}
	else if (frameIdx > 0) {
		mutex_completeTrajectory.lock();
		computeSiftTransformCU(m_local->getCurrentSiftTransformsGPU(), m_local->getNumFiltMatchesGPU(),
			d_completeTrajectory, lastValidCompleteTransform, d_siftTrajectory, frameIdx, localFrameIdx, d_currIntegrateTransform + frameIdx);
		mutex_completeTrajectory.unlock();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&m_currIntegrateTransform[frameIdx], d_currIntegrateTransform + frameIdx, sizeof(float4x4), cudaMemcpyDeviceToHost));
	}
}

void OnlineBundler::prepareLocalSolve(unsigned int curFrame, bool isSequenceEnd)
{
	m_state.m_processState = BundlerState::DO_NOTHING;
	unsigned int curLocalIdx = (std::max(curFrame, 1u) - 1) / m_submapSize;
	if (isSequenceEnd && (curFrame % m_submapSize) == 0) { // only the overlap frame
		// invalidate //TODO how annoying is it to keep this frame?
		curLocalIdx++;
		m_state.m_localToSolve = -((int)curLocalIdx + ID_MARK_OFFSET);
		m_state.m_processState = BundlerState::INVALIDATE;
		if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: last local submap 1 frame -> invalidating" << curFrame << std::endl;
	}
	else {
		// if valid local
		if (m_local->isValid()) {
			// ready to solve local
			MLIB_ASSERT(m_state.m_localToSolve == -1);
			m_state.m_localToSolve = curLocalIdx;
			m_state.m_processState = BundlerState::PROCESS;
		}
		else {
			// invalidate the local
			m_state.m_localToSolve = -((int)curLocalIdx + ID_MARK_OFFSET);
			m_state.m_processState = BundlerState::INVALIDATE;
			if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: invalid local submap " << curFrame << " (idx = " << curLocalIdx << ")" << std::endl;
		}
	}

	// switch local submaps
	mutex_optLocal.lock();
	std::swap(m_local, m_optLocal);
	mutex_optLocal.unlock();
}

void OnlineBundler::processInput()
{
	const unsigned int curFrame = m_cudaImageManager->getCurrFrameNumber();
	const bool bIsLastLocal = isLastLocalFrame(curFrame);
	if (curFrame > 0 && m_state.m_lastFrameProcessed == curFrame) { //sequence has ended (no new frames from cudaimagemanager)
		if (m_state.m_numFramesPastEnd == 0 && m_state.m_localToSolve == -1) {
			if (!bIsLastLocal) prepareLocalSolve(curFrame, true);
		}
		const unsigned int numSolveFramesBeforeExit = GlobalAppState::get().s_numSolveFramesBeforeExit;
		if (numSolveFramesBeforeExit != (unsigned int)-1) {
#ifdef USE_GLOBAL_DENSE_AT_END
			if (m_state.m_numFramesPastEnd == numSolveFramesBeforeExit) {
				if (m_state.m_lastFrameProcessed < 10000) { //TODO fix here
					GlobalBundlingState::get().s_numGlobalNonLinIterations = 3;
					const unsigned int maxNumIts = GlobalBundlingState::get().s_numGlobalNonLinIterations;
					std::vector<float> sparseWeights(maxNumIts, 1.0f);
					std::vector<float> denseDepthWeights(maxNumIts, 15.0f);
					std::vector<float> denseColorWeights(maxNumIts, 0.0f);

					m_global->setSolveWeights(sparseWeights, denseDepthWeights, denseColorWeights);
				}
			}
#endif
			if (m_state.m_numFramesPastEnd == numSolveFramesBeforeExit + 1) {
				std::cout << "stopping solve" << std::endl;
				m_state.m_bUseSolve = false;
			}
		}
		m_state.m_numFramesPastEnd++;
		return; //nothing new to process
	}
	//get depth/color data
	getCurrentFrame();

	// feature detect
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
	m_local->detectFeatures(m_input.d_intensitySIFT, m_input.d_inputDepthFilt);
	m_local->storeCachedFrame(m_input.m_inputDepthWidth, m_input.m_inputDepthHeight, m_input.d_inputColor, m_input.m_inputColorWidth, m_input.m_inputColorHeight, m_input.d_inputDepthRaw);
	const unsigned int curLocalFrame = m_local->getCurrFrameNumber();
	if (bIsLastLocal) {
		mutex_optLocal.lock();
		m_optLocal->copyFrame(m_local, curLocalFrame);
		mutex_optLocal.unlock();
	}
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(true).timeSiftDetection = m_timer.getElapsedTimeMS(); }

	//feature match
	m_state.m_bLastFrameValid = true;
	if (curLocalFrame > 0) {
		mutex_siftMatcher.lock();
		m_state.m_bLastFrameValid = m_local->matchAndFilter() != ((unsigned int)-1);
		mutex_siftMatcher.unlock();
		computeCurrentSiftTransform(m_state.m_bLastFrameValid, curFrame, curLocalFrame, m_state.m_lastValidCompleteTransform);
	}

	if (bIsLastLocal) { //prepare for solve
		prepareLocalSolve(curFrame, false);
	}

	m_state.m_lastFrameProcessed = curFrame;
}

bool OnlineBundler::getCurrentIntegrationFrame(mat4f& siftTransform, unsigned int& frameIdx, bool& bGlobalTrackingLost)
{
	bGlobalTrackingLost = m_state.m_bGlobalTrackingLost;
	if (m_state.m_bLastFrameValid) {
		siftTransform = m_currIntegrateTransform[m_state.m_lastFrameProcessed];
		frameIdx = m_state.m_lastFrameProcessed;
		return true;
	}
	else {
		return false;
	}
}

void OnlineBundler::optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	MLIB_ASSERT(m_state.m_bUseSolve);
	if (m_state.m_processState == BundlerState::DO_NOTHING) return;

	mutex_optLocal.lock();
	BundlerState::PROCESS_STATE optLocalState = m_state.m_processState;
	m_state.m_processState = BundlerState::DO_NOTHING;
	unsigned int curLocalIdx = (unsigned int)-1;
	unsigned int numLocalFrames = std::min(m_submapSize, m_optLocal->getNumFrames());
	if (optLocalState == BundlerState::PROCESS) {
		curLocalIdx = m_state.m_localToSolve;
		bool removed = false;
		bool valid = m_optLocal->optimize(numNonLinIterations, numLinIterations, GlobalBundlingState::get().s_useLocalVerify,
			false, m_state.m_numFramesPastEnd != 0, removed); // no max res removal
		if (valid) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_localTrajectories + (m_submapSize + 1)*curLocalIdx, m_optLocal->getTrajectoryGPU(), sizeof(float4x4)*(m_submapSize + 1), cudaMemcpyDeviceToDevice));
			m_state.m_processState = BundlerState::PROCESS;
		}
		else m_state.m_processState = BundlerState::INVALIDATE;
	}
	else if (optLocalState == BundlerState::INVALIDATE) {
		curLocalIdx = -m_state.m_localToSolve - ID_MARK_OFFSET;
		m_state.m_processState = BundlerState::INVALIDATE;
	}
	m_state.m_localToSolve = (unsigned int)-1;
	m_state.m_lastLocalSolved = curLocalIdx;
	m_state.m_totalNumOptLocalFrames = m_submapSize * m_state.m_lastLocalSolved + numLocalFrames; //last local solved is 0-indexed so this doesn't overcount
	mutex_optLocal.unlock();
}

void OnlineBundler::initializeNextGlobalTransform(unsigned int lastMatchedIdx, unsigned int lastValidLocal)
{
	const unsigned int numFrames = m_global->getNumFrames();
	MLIB_ASSERT(numFrames >= 1);
	initNextGlobalTransformCU(m_global->getTrajectoryGPU(), numFrames, lastMatchedIdx, d_localTrajectories, lastValidLocal, m_submapSize + 1);
}

void OnlineBundler::processGlobal()
{
	//global match/filter
	MLIB_ASSERT(m_state.m_bUseSolve);

	BundlerState::PROCESS_STATE processState = m_state.m_processState;
	if (processState == BundlerState::DO_NOTHING) {
		if (m_state.m_numFramesPastEnd != 0) { //sequence is over, try revalidation still
			unsigned int idx = m_global->tryRevalidation(m_state.m_lastLocalSolved, true);
			if (idx != (unsigned int)-1) { //validate chunk images
				const std::vector<int>& validLocal = m_localTrajectoriesValid[idx];
				for (unsigned int i = 0; i < validLocal.size(); i++) {
					if (validLocal[i] == 1)	validateImages(idx * m_submapSize + i);
				}
				m_state.m_processState = BundlerState::PROCESS;
			}
		}
		return;
	}

	if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addGlobalFrameTiming();
	m_state.m_processState = BundlerState::DO_NOTHING;
	if (processState == BundlerState::PROCESS) {
		//if (m_global->getNumFrames() <= m_state.m_lastLocalSolved) {
			MLIB_ASSERT((int)m_global->getNumFrames() <= m_state.m_lastLocalSolved);
			//fuse
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
			mutex_optLocal.lock();
			m_optLocal->fuseToGlobal(m_global);//TODO GPU version of this??

			const unsigned int curGlobalFrame = m_global->getCurrFrameNumber();
			const std::vector<int>& validImagesLocal = m_optLocal->getValidImages(); 
			const unsigned int numLocalFrames = std::min(m_submapSize, m_optLocal->getNumFrames());
			unsigned int lastValidLocal = 0; 
			for (int i = (int)m_optLocal->getNumFrames() - 1; i >= 0; i--) {
				if (validImagesLocal[i]) { lastValidLocal = i; break; }
			}
			for (unsigned int i = 0; i < numLocalFrames; i++) {
				if (validImagesLocal[i] == 0)
					invalidateImages(curGlobalFrame * m_submapSize + i);
			}
			m_localTrajectoriesValid[curGlobalFrame] = validImagesLocal; m_localTrajectoriesValid[curGlobalFrame].resize(numLocalFrames);
			initializeNextGlobalTransform(curGlobalFrame, lastValidLocal); //(initializes 2 ahead) 
			//done with local data
			m_optLocal->reset();
			mutex_optLocal.unlock();
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(false).timeSiftDetection = m_timer.getElapsedTimeMS(); }

			//match!
			if (m_global->getNumFrames() > 1) {
				mutex_siftMatcher.lock();
				unsigned int lastMatchedGlobal = m_global->matchAndFilter();
				mutex_siftMatcher.unlock();
				if (lastMatchedGlobal == (unsigned int)-1) {
					m_state.m_bGlobalTrackingLost = true;
					m_state.m_processState = BundlerState::INVALIDATE;
				}
				else {
					m_state.m_bGlobalTrackingLost = false;
					const unsigned int revalidateIdx = m_global->getRevalidatedIdx();
					if (revalidateIdx != (unsigned int)-1) { //validate chunk images
						const std::vector<int>& validLocal = m_localTrajectoriesValid[revalidateIdx];
						for (unsigned int i = 0; i < validLocal.size(); i++) {
							if (validLocal[i] == 1)	validateImages(revalidateIdx * m_submapSize + i);
						}
					}
					m_state.m_processState = BundlerState::PROCESS;
				}
			}
		//}
	}
	else if (processState == BundlerState::INVALIDATE) {
		// cache
		m_state.m_processState = BundlerState::INVALIDATE; 
		m_global->addInvalidFrame(); //add invalidated (fake) global frame
		//finish local opt
		mutex_optLocal.lock();
		m_optLocal->reset();
		mutex_optLocal.unlock();
		invalidateImages(m_submapSize * m_state.m_lastLocalSolved, m_state.m_totalNumOptLocalFrames); 
	}
}

void OnlineBundler::updateTrajectory(unsigned int curFrame)
{
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_imageInvalidateList, m_invalidImagesList.data(), sizeof(int)*curFrame, cudaMemcpyHostToDevice));
	mutex_completeTrajectory.lock();
	updateTrajectoryCU(m_global->getTrajectoryGPU(), m_global->getNumFrames(),
		d_completeTrajectory, curFrame, d_localTrajectories, m_submapSize + 1,
		m_global->getNumFrames(), d_imageInvalidateList);
	mutex_completeTrajectory.unlock();
}

void OnlineBundler::optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	MLIB_ASSERT(m_state.m_bUseSolve);
	const bool isSequenceDone = m_state.m_numFramesPastEnd > 0;
	if (!isSequenceDone && m_state.m_processState == BundlerState::DO_NOTHING) return; //always solve after end of sequence
	MLIB_ASSERT(m_state.m_lastLocalSolved >= 0);

	const BundlerState::PROCESS_STATE state = isSequenceDone ? BundlerState::PROCESS : m_state.m_processState; //always solve after end of sequence
	unsigned int numTotalFrames = m_state.m_totalNumOptLocalFrames;
	if (state == BundlerState::PROCESS) {
		const unsigned int countNumFrames = (m_state.m_numFramesPastEnd > 0) ? m_state.m_numFramesPastEnd : numTotalFrames / m_submapSize;
		bool bRemoveMaxResidual = (countNumFrames % m_numOptPerResidualRemoval) == (m_numOptPerResidualRemoval - 1);
		bool removed = false;
		bool valid = m_global->optimize(numNonLinIterations, numLinIterations, false, bRemoveMaxResidual, m_state.m_numFramesPastEnd > 0, removed);//no verify
		if (removed) { // may invalidate already invalidated images
			for (unsigned int i = 0; i < m_global->getNumFrames(); i++) {
				if (m_global->getValidImages()[i] == 0)
					invalidateImages(i * m_submapSize, std::min((i + 1)*m_submapSize, numTotalFrames));
			}
		}

		updateTrajectory(numTotalFrames);
		m_trajectoryManager->updateOptimizedTransform(d_completeTrajectory, numTotalFrames);
		m_state.m_numCompleteTransforms = numTotalFrames;
		if (valid) m_state.m_lastValidCompleteTransform = m_submapSize * m_state.m_lastLocalSolved; //TODO over-conservative but easier
	}
	else if (state == BundlerState::INVALIDATE) {
		m_global->invalidateLastFrame();
		invalidateImages(m_submapSize * m_state.m_lastLocalSolved, m_state.m_totalNumOptLocalFrames);
		updateTrajectory(numTotalFrames);
		m_trajectoryManager->updateOptimizedTransform(d_completeTrajectory, numTotalFrames);
		m_state.m_numCompleteTransforms = numTotalFrames;
	}

	m_state.m_processState = BundlerState::DO_NOTHING;
}

void OnlineBundler::process(unsigned int numNonLinItersLocal, unsigned int numLinItersLocal, unsigned int numNonLinItersGlobal, unsigned int numLinItersGlobal)
{
	if (!m_state.m_bUseSolve) return; //solver off

	optimizeLocal(numNonLinItersLocal, numLinItersLocal);
	processGlobal();
	optimizeGlobal(numNonLinItersGlobal, numLinItersGlobal);

	//{ //no opt
	//	m_state.m_localToSolve = -1;
	//	m_state.m_processState = BundlerState::DO_NOTHING;
	//	mutex_optLocal.lock();
	//	m_optLocal->reset();
	//	mutex_optLocal.unlock();
	//}
	//{ //local solve only
	//	optimizeLocal(numNonLinItersLocal, numLinItersLocal);
	//	mutex_optLocal.lock();
	//	unsigned int curFrame = (m_state.m_lastLocalSolved < 0) ? (unsigned int)-1 : m_state.m_totalNumOptLocalFrames;
	//	if (m_state.m_lastLocalSolved >= 0) {
	//		float4x4 relativeTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&relativeTransform, d_localTrajectories + m_state.m_lastLocalSolved*(m_submapSize + 1) + m_submapSize, sizeof(float4x4), cudaMemcpyDeviceToHost));
	//		float4x4 prevTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&prevTransform, m_global->getTrajectoryGPU() + m_state.m_lastLocalSolved, sizeof(float4x4), cudaMemcpyDeviceToHost));
	//		float4x4 newTransform = prevTransform * relativeTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_global->getTrajectoryGPU() + m_state.m_lastLocalSolved + 1, &newTransform, sizeof(float4x4), cudaMemcpyHostToDevice));
	//	}
	//	if (m_state.m_lastLocalSolved > 0) {
	//		// update trajectory
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_imageInvalidateList, m_invalidImagesList.data(), sizeof(int)*curFrame, cudaMemcpyHostToDevice));
	//		updateTrajectoryCU(m_global->getTrajectoryGPU(), m_state.m_lastLocalSolved,
	//			d_completeTrajectory, curFrame,
	//			d_localTrajectories, m_submapSize + 1, m_state.m_lastLocalSolved,
	//			d_imageInvalidateList);
	//	}
	//	m_optLocal->reset();
	//	mutex_optLocal.unlock();
	//	m_state.m_localToSolve = -1;
	//	m_state.m_processState = BundlerState::DO_NOTHING;
	//}
	//{ //local solve + glob match only
	//	optimizeLocal(numNonLinItersLocal, numLinItersLocal);
	//	processGlobal();
	//	unsigned int curFrame = (m_state.m_lastLocalSolved < 0) ? (unsigned int)-1 : m_state.m_totalNumOptLocalFrames;
	//	if (m_state.m_lastLocalSolved >= 0) {
	//		float4x4 relativeTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&relativeTransform, d_localTrajectories + m_state.m_lastLocalSolved*(m_submapSize + 1) + m_submapSize, sizeof(float4x4), cudaMemcpyDeviceToHost));
	//		float4x4 prevTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&prevTransform, m_global->getTrajectoryGPU() + m_state.m_lastLocalSolved, sizeof(float4x4), cudaMemcpyDeviceToHost));
	//		float4x4 newTransform = prevTransform * relativeTransform;
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_global->getTrajectoryGPU() + m_state.m_lastLocalSolved + 1, &newTransform, sizeof(float4x4), cudaMemcpyHostToDevice));
	//	}
	//	if (m_state.m_lastLocalSolved > 0) {
	//		// update trajectory
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_imageInvalidateList, m_invalidImagesList.data(), sizeof(int)*curFrame, cudaMemcpyHostToDevice));
	//		updateTrajectoryCU(m_global->getTrajectoryGPU(), m_state.m_lastLocalSolved,
	//			d_completeTrajectory, curFrame,
	//			d_localTrajectories, m_submapSize + 1, m_state.m_lastLocalSolved,
	//			d_imageInvalidateList);
	//	}
	//	m_state.m_localToSolve = -1;
	//	m_state.m_processState = BundlerState::DO_NOTHING;
	//}
}

void OnlineBundler::saveGlobalSparseCorrsToFile(const std::string& filename) const
{
	m_global->saveSparseCorrsToFile(filename);
}

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
void OnlineBundler::finishCorrespondenceEvaluatorLogging()
{
	m_global->finishCorrespondenceEvaluatorLogging();
	//these ones shouldn't have it anyways...
	m_local->finishCorrespondenceEvaluatorLogging();
	m_optLocal->finishCorrespondenceEvaluatorLogging();
}
#endif
