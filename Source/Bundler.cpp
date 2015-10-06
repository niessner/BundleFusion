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
	m_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, m_submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage, m_submapSize, m_CudaImageManager);
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
	if (curFrame > 0 && m_currentState.m_lastFrameProcessed == curFrame) {
		static bool processLastFrame = true;
		if (processLastFrame && m_currentState.m_localToSolve == -1) {
			if (!m_SubmapManager.isLastLocalFrame(curFrame)) prepareLocalSolve(curFrame, true);
			processLastFrame = false;
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

	//printKey("key" + std::to_string(curLocalFrame) + ".png", curFrame, g_SubmapManager.currentLocal, curLocalFrame);
	
	// match with every other local
	m_currentState.m_bLastFrameValid = 1;
	if (curLocalFrame > 0) {
		//matchAndFilter(m_SubmapManager.currentLocal, m_SubmapManager.currentLocalCache, curFrame - curLocalFrame, 1);
		m_currentState.m_bLastFrameValid = m_SubmapManager.localMatchAndFilter(MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsicsInv));

		m_SubmapManager.computeCurrentSiftTransform(curFrame, curLocalFrame, m_currentState.m_lastValidCompleteTransform);
	}
	m_currentState.m_lastFrameProcessed = curFrame;

	////!!!DEBUGGING
	//const mat4f& intTransform = m_SubmapManager.getCurrentIntegrateTransform(curFrame);
	//if (m_currentState.m_bLastFrameValid && ((intTransform(0, 0) == 0.0f && intTransform(1, 1) == 0.0f && intTransform(2, 2) == 0.0f && intTransform(3, 3) == 0.0f) ||
	//	intTransform[0] == -std::numeric_limits<float>::infinity())) {
	//	std::cout << "valid but transform = " << std::endl << intTransform << std::endl;
	//	getchar();
	//}
	////!!!DEBUGGING

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

	m_currentState.m_lastNumLocalFrames = m_SubmapManager.getNumNextLocalFrames();
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
}


void Bundler::processGlobal()
{
	if (m_currentState.m_bProcessGlobal == BundlerState::DO_NOTHING) {
		if (!m_RGBDSensor->isReceivingFrames()) m_currentState.m_bOptimizeGlobal = BundlerState::PROCESS;
		return;
	}

	if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addGlobalFrameTiming();
	if (m_currentState.m_bProcessGlobal == BundlerState::PROCESS) {
		// cache
		m_SubmapManager.copyToGlobalCache(); 

		m_currentState.m_bOptimizeGlobal = (BundlerState::PROCESS_STATE)m_SubmapManager.computeAndMatchGlobalKeys(m_currentState.m_lastLocalSolved,
			MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsics), MatrixConversion::toCUDA(m_bundlerInputData.m_SIFTIntrinsicsInv));
	}
	else {
		// cache
		m_SubmapManager.incrementGlobalCache();
		m_currentState.m_bOptimizeGlobal = BundlerState::INVALIDATE;

		//getchar();
		m_SubmapManager.invalidateImages(m_submapSize * m_currentState.m_lastLocalSolved, m_submapSize * m_currentState.m_lastLocalSolved + m_currentState.m_lastNumLocalFrames);
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

	unsigned int numFrames = m_submapSize * m_currentState.m_lastLocalSolved + m_currentState.m_lastNumLocalFrames;

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
			m_currentState.m_numCompleteTransforms = numFrames;
			m_SubmapManager.updateTrajectory(m_currentState.m_numCompleteTransforms);
			m_trajectoryManager->updateOptimizedTransform(m_SubmapManager.getCompleteTrajectory(), m_currentState.m_numCompleteTransforms);

			m_currentState.m_bOptimizeGlobal = BundlerState::DO_NOTHING;
		}
	}

}

void Bundler::printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame) const
{
	//TODO get color cpu for these functions
	CUDAImageManager::ManagedRGBDInputFrame& integrateFrame = m_CudaImageManager->getIntegrateFrame(allFrame);

	ColorImageR8G8B8A8 im(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(im.getPointer(), integrateFrame.getColorFrameGPU(), sizeof(uchar4) * m_CudaImageManager->getIntegrationWidth() * m_CudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
	im.reSample(m_bundlerInputData.m_widthSIFT, m_bundlerInputData.m_heightSIFT);

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];
		RGBColor c = RGBColor::randomColor();
		vec4uc color(c.r, c.g, c.b, c.a);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}

void Bundler::printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices, const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered) const
{
	// get data
	std::vector<SIFTKeyPoint> keys;
	siftManager->getSIFTKeyPointsDEBUG(keys); // prev frame

	std::vector<uint2> keyPointIndices;
	std::vector<float> matchDistances;
	if (filtered) {
		siftManager->getFiltKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	else {
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	if (keyPointIndices.size() == 0) return;

	ColorImageR32G32B32 matchImage(image1.getWidth() * 2, image1.getHeight());
	ColorImageR32G32B32 im1(image1);
	ColorImageR32G32B32 im2(image2);
	matchImage.copyIntoImage(im1, 0, 0);
	matchImage.copyIntoImage(im2, image1.getWidth(), 0);

	float maxMatchDistance = 0.0f;
	RGBColor lowColor = ml::RGBColor::Blue;
	RGBColor highColor = ml::RGBColor::Red;
	for (unsigned int i = 0; i < keyPointIndices.size(); i++) {
		const SIFTKeyPoint& key1 = keys[keyPointIndices[i].x];
		const SIFTKeyPoint& key2 = keys[keyPointIndices[i].y];
		if (matchDistances[i] > maxMatchDistance) maxMatchDistance = matchDistances[i];

		RGBColor c = RGBColor::interpolate(lowColor, highColor, matchDistances[i] / distMax);
		vec3f color(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
		vec2i p0 = ml::math::round(ml::vec2f(key1.pos.x, key1.pos.y));
		vec2i p1 = ml::math::round(ml::vec2f(key2.pos.x + image1.getWidth(), key2.pos.y));
		ImageHelper::drawCircle(matchImage, p0, ml::math::round(key1.scale), color);
		ImageHelper::drawCircle(matchImage, p1, ml::math::round(key2.scale), color);
		ImageHelper::drawLine(matchImage, p0, p1, color);
	}
	std::cout << "(" << imageIndices << "): max match distance = " << maxMatchDistance << std::endl;
	FreeImageWrapper::saveImage(filename, matchImage);
}

void Bundler::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered, unsigned int frameStart, unsigned int frameSkip) const
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	unsigned int curFrame = numFrames - 1; //TODO get color cpu for these functions
	CUDAImageManager::ManagedRGBDInputFrame& curIntegrateFrame = m_CudaImageManager->getIntegrateFrame(curFrame * frameSkip + frameStart);
	ColorImageR8G8B8A8 curImage(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curImage.getPointer(), curIntegrateFrame.getColorFrameGPU(),
		sizeof(uchar4) * curImage.getNumPixels(), cudaMemcpyDeviceToHost));
	curImage.reSample(m_bundlerInputData.m_widthSIFT, m_bundlerInputData.m_heightSIFT);

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		CUDAImageManager::ManagedRGBDInputFrame& prevIntegrateFrame = m_CudaImageManager->getIntegrateFrame(prev * frameSkip + frameStart);
		ColorImageR8G8B8A8 prevImage(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevImage.getPointer(), prevIntegrateFrame.getColorFrameGPU(),
			sizeof(uchar4) * prevImage.getNumPixels(), cudaMemcpyDeviceToHost));
		prevImage.reSample(m_bundlerInputData.m_widthSIFT, m_bundlerInputData.m_heightSIFT);

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered);
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

	if (GlobalBundlingState::get().s_erodeSIFTdepth) {
		unsigned int numIter = 2;

		numIter = 2 * ((numIter + 1) / 2);
		for (unsigned int i = 0; i < numIter; i++) {
			if (i % 2 == 0) {
				CUDAImageUtil::erodeDepthMap(m_bundlerInputData.d_depthErodeHelper, m_bundlerInputData.d_inputDepth, 3,
					m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight, 0.05f, 0.3f);
			}
			else {
				CUDAImageUtil::erodeDepthMap(m_bundlerInputData.d_inputDepth, m_bundlerInputData.d_depthErodeHelper, 3,
					m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight, 0.05f, 0.3f);
			}
		}
	}
	if (m_bundlerInputData.m_bFilterDepthValues) {
		CUDAImageUtil::gaussFilterFloatMap(m_bundlerInputData.d_depthErodeHelper, m_bundlerInputData.d_inputDepth,
			m_bundlerInputData.m_fBilateralFilterSigmaD, m_bundlerInputData.m_fBilateralFilterSigmaR,
			m_bundlerInputData.m_inputDepthWidth, m_bundlerInputData.m_inputDepthHeight);
		std::swap(m_bundlerInputData.d_inputDepth, m_bundlerInputData.d_depthErodeHelper);
	}
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

