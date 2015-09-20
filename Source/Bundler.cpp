#include "stdafx.h"
#include "CUDAImageManager.h"
#include "CUDACache.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/MatrixConversion.h"
#include "SiftGPU/SIFTMatchFilter.h"
#include "ImageHelper.h"

#include "mLibCuda.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"
#include "Bundler.h"


extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params);

Timer Bundler::s_timer;


Bundler::Bundler(RGBDSensor* sensor, CUDAImageManager* imageManager)
{
	m_CudaImageManager = imageManager;
	m_RGBDSensor = sensor;

	// init CUDA
	m_submapSize = GlobalBundlingState::get().s_submapSize;
	m_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, m_submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage, m_submapSize, m_CudaImageManager);
	//TODO fix
	if (GlobalAppState::get().s_sensorIdx == 3) {
		m_SubmapManager.setTotalNumFrames(((BinaryDumpReader*)m_RGBDSensor)->getNumTotalFrames());
	}
	m_SparseBundler.init(GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumCorrPerImage);

	m_trajectoryManager = new TrajectoryManager(GlobalBundlingState::get().s_maxNumImages * m_submapSize);

	// init sift camera constant params
	m_siftCameraParams.m_depthWidth = m_CudaImageManager->getIntegrationWidth();
	m_siftCameraParams.m_depthHeight = m_CudaImageManager->getIntegrationHeight();
	m_siftCameraParams.m_intensityWidth = m_CudaImageManager->getSIFTWidth();
	m_siftCameraParams.m_intensityHeight = m_CudaImageManager->getSIFTHeight();
	m_siftCameraParams.m_siftIntrinsics = MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsics());
	m_siftCameraParams.m_siftIntrinsicsInv = MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsicsInv());
	m_siftCameraParams.m_downSampIntrinsics = MatrixConversion::toCUDA(m_SubmapManager.currentLocalCache->getIntrinsics());
	m_siftCameraParams.m_downSampIntrinsicsInv = MatrixConversion::toCUDA(m_SubmapManager.currentLocalCache->getIntrinsicsInv());
	updateConstantSiftCameraParams(m_siftCameraParams);

	m_sift = new SiftGPU;
	m_siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	m_sift->SetParams(0, GlobalBundlingState::get().s_enableDetailedTimings, 150);
	m_sift->InitSiftGPU(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight(), m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());
	m_siftMatcher->InitSiftMatch();

	m_bHasProcessedInputFrame = false;
	m_bExitBundlingThread = false;
}

Bundler::~Bundler()
{
	SAFE_DELETE(m_sift);
	SAFE_DELETE(m_siftMatcher);
	SAFE_DELETE(m_trajectoryManager);
}

void Bundler::processInput()
{

	const unsigned int curFrame = m_CudaImageManager->getCurrFrameNumber();
	std::cout << "[ frame " << curFrame << " ]" << std::endl;

	if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addLocalFrameTiming();

	CUDAImageManager::ManagedRGBDInputFrame& integrateFrame = m_CudaImageManager->getLastIntegrateFrame();

	// run SIFT
	SIFTImageGPU& cur = m_SubmapManager.currentLocal->createSIFTImageGPU();
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
	int success = m_sift->RunSIFT(m_CudaImageManager->getIntensityImageSIFT(), integrateFrame.getDepthFrameGPU());
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection on frame " + std::to_string(curFrame));
	unsigned int numKeypoints = m_sift->GetKeyPointsAndDescriptorsCUDA(cur, integrateFrame.getDepthFrameGPU());
	m_SubmapManager.currentLocal->finalizeSIFTImageGPU(numKeypoints);
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(true).timeSiftDetection = s_timer.getElapsedTimeMS(); }

	// process cuda cache
	const unsigned int curLocalFrame = m_SubmapManager.currentLocal->getNumImages() - 1;
	m_SubmapManager.currentLocalCache->storeFrame(integrateFrame.getDepthFrameGPU(), integrateFrame.getColorFrameGPU(), m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	if (GlobalBundlingState::get().s_recordKeysPointCloud && (curLocalFrame == 0 || m_SubmapManager.isLastLocalFrame(curFrame))) {
		m_RGBDSensor->recordPointCloud();
	}
	//printKey("key" + std::to_string(curLocalFrame) + ".png", curFrame, g_SubmapManager.currentLocal, curLocalFrame);
	// sift/cuda cache for next local
	if (m_SubmapManager.isLastLocalFrame(curFrame)) {
		MLIB_ASSERT(m_currentState.m_lastLocalSolved + 1 == m_SubmapManager.getCurrLocal(curFrame)); // otherwise this will overwrite the data needed by the local solve!
		MLIB_ASSERT(m_SubmapManager.nextLocal->getNumImages() == 0);

		SIFTImageGPU& curNext = m_SubmapManager.nextLocal->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(curNext.d_keyPoints, cur.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(curNext.d_keyPointDescs, cur.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		m_SubmapManager.nextLocal->finalizeSIFTImageGPU(numKeypoints);
		m_SubmapManager.nextLocalCache->copyCacheFrameFrom(m_SubmapManager.currentLocalCache, curLocalFrame);
	}
	// match with every other local
	SIFTImageManager* currentLocal = m_SubmapManager.currentLocal;
	if (curLocalFrame > 0) {
		matchAndFilter(currentLocal, m_SubmapManager.currentLocalCache, curFrame - curLocalFrame, 1);

		currentLocal->computeSiftTransformCU(m_SubmapManager.d_completeTrajectory, m_currentState.m_lastValidCompleteTransform,
			m_SubmapManager.d_siftTrajectory, curFrame, curLocalFrame, m_SubmapManager.getCurrIntegrateTransform(curFrame));
	}
	m_currentState.m_lastFrameProcessed = curFrame;
	m_currentState.m_bLastFrameValid = (currentLocal->getValidImages()[curLocalFrame] != 0);

	// global frame
	if (m_SubmapManager.isLastFrame(curFrame) || m_SubmapManager.isLastLocalFrame(curFrame)) { // end frame or global frame
		if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addGlobalFrameTiming();
		// cache
		m_SubmapManager.globalCache->copyCacheFrameFrom(m_SubmapManager.currentLocalCache, 0);

		// if valid local
		const std::vector<int>& validImagesLocal = currentLocal->getValidImages();
		const unsigned int curLocalIdx = m_SubmapManager.getCurrLocal(curFrame);
		if (validImagesLocal[1]) {
			// ready to solve local
			m_currentState.m_localToSolve = curLocalIdx;

			// switch local submaps
			m_SubmapManager.switchLocal();

			
		}
		else {
			m_currentState.m_localToSolve = -1; // no need to solve local
			m_currentState.m_lastLocalSolved = curLocalIdx; // already "solved"
			m_currentState.m_bOptimizeGlobal = false;
			m_currentState.m_lastNumLocalFrames = std::min(m_submapSize, m_SubmapManager.currentLocal->getNumImages());
			m_currentState.m_numCompleteTransforms = curFrame;

			//add invalidated (fake) global frame
			std::cout << "WARNING: invalid local submap " << curFrame << " (" << m_SubmapManager.global->getNumImages() << ", " << m_SubmapManager.currentLocal->getNumImages() << ")" << std::endl;
			//getchar();
			SIFTImageGPU& curGlobalImage = m_SubmapManager.global->createSIFTImageGPU();
			m_SubmapManager.global->finalizeSIFTImageGPU(0);
			m_SubmapManager.switchLocal();
			m_SubmapManager.finishLocalOpt();
			m_SubmapManager.global->invalidateFrame(m_SubmapManager.global->getNumImages() - 1);

			m_SubmapManager.updateTrajectory(curFrame);
			m_SubmapManager.initializeNextGlobalTransform(true);
		}
	} // global
}

bool Bundler::getCurrentIntegrationFrame(mat4f& siftTransform, const float* & d_depth, const uchar4* & d_color)
{
	if (m_currentState.m_bLastFrameValid) {
		cutilSafeCall(cudaMemcpy(&siftTransform, m_SubmapManager.getCurrIntegrateTransform(m_currentState.m_lastFrameProcessed), sizeof(float4x4), cudaMemcpyDeviceToHost));	//TODO MT needs to be copied from the other GPU...
		d_depth = m_CudaImageManager->getLastIntegrateFrame().getDepthFrameGPU();
		d_color = m_CudaImageManager->getLastIntegrateFrame().getColorFrameGPU();

		m_trajectoryManager->addFrame(TrajectoryManager::TrajectoryFrame::Integrated, siftTransform, m_currentState.m_lastFrameProcessed);
		return true;
	}
	else {
		m_trajectoryManager->addFrame(TrajectoryManager::TrajectoryFrame::NotIntegrated_NoTransform, mat4f::zero(), m_currentState.m_lastFrameProcessed);
		return false;
	}
}

void Bundler::optimizeLocal(unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	if (m_currentState.m_localToSolve == -1) return; // nothing to solve

	const unsigned int currLocalIdx = m_currentState.m_localToSolve;
	m_currentState.m_localToSolve = -1; 
	m_currentState.m_lastNumLocalFrames = std::min(m_submapSize, m_SubmapManager.nextLocal->getNumImages());

	solve(m_SubmapManager.getLocalTrajectoryGPU(currLocalIdx), m_SubmapManager.nextLocal, numNonLinIterations, numLinIterations, true, false);
	// still need this for global key fuse

	m_currentState.m_lastLocalSolved = currLocalIdx;
}

void Bundler::processGlobal()
{
	SIFTImageManager* global = m_SubmapManager.global;
	if ((int)global->getNumImages() <= m_currentState.m_lastLocalSolved) {
		// fuse to global
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		SIFTImageGPU& curGlobalImage = global->createSIFTImageGPU();
		unsigned int numGlobalKeys = m_SubmapManager.nextLocal->FuseToGlobalKeyCU(curGlobalImage, m_SubmapManager.getLocalTrajectoryGPU(m_currentState.m_lastLocalSolved),
			MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsics()), MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsicsInv()));
		global->finalizeSIFTImageGPU(numGlobalKeys);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(false).timeSiftDetection = s_timer.getElapsedTimeMS(); }

		const std::vector<int>& validImagesLocal = m_SubmapManager.nextLocal->getValidImages();
		for (unsigned int i = 0; i < std::min(m_submapSize, m_SubmapManager.nextLocal->getNumImages()); i++) {
			if (validImagesLocal[i] == 0)
				m_SubmapManager.invalidateImages((global->getNumImages() - 1) * m_submapSize + i);
		}
		m_SubmapManager.initializeNextGlobalTransform(false);
		// done with local data!
		m_SubmapManager.finishLocalOpt();

		//unsigned int gframe = (unsigned int)global->getNumImages() - 1;
		//printKey("debug/keys/" + std::to_string(gframe) + ".png", gframe*submapSize, global, gframe);

		// match with every other global
		if (global->getNumImages() > 1) {
			matchAndFilter(global, m_SubmapManager.globalCache, 0, m_submapSize);
			//printCurrentMatches("output/matches/", binaryDumpReader, global, true, 0, submapSize);

			if (global->getValidImages()[global->getNumImages() - 1]) {
				// ready to solve global
				m_currentState.m_bOptimizeGlobal = true;
			}
			else {
				std::cout << "WARNING: last image not valid! no new global images for solve" << std::endl;
				std::cout << "#global images = " << global->getNumImages() << std::endl;
				//getchar();
			}
		}
	}
}

void Bundler::optimizeGlobal(unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	if (!m_currentState.m_bOptimizeGlobal) return; // nothing to solve

	//solve!
	m_currentState.m_bOptimizeGlobal = false;
	solve(m_SubmapManager.d_globalTrajectory, m_SubmapManager.global, numNonLinIterations, numLinIterations, false, GlobalBundlingState::get().s_recordSolverConvergence);

	const unsigned int numGlobalFrames = m_SubmapManager.global->getNumImages();
	unsigned int numFrames = (numGlobalFrames > 0) ? ((numGlobalFrames - 1) * m_submapSize + m_currentState.m_lastNumLocalFrames) : m_currentState.m_lastNumLocalFrames;

	// may invalidate already invalidated images
	const std::vector<int>& validImagesGlobal = m_SubmapManager.global->getValidImages();
	for (unsigned int i = 0; i < numGlobalFrames; i++) {
		if (validImagesGlobal[i] == 0) {
			m_SubmapManager.invalidateImages(i * m_submapSize, std::min((i + 1)*m_submapSize, numFrames));
		}
	}

	m_SubmapManager.updateTrajectory(numFrames);
	m_trajectoryManager->updateOptimizedTransform(m_SubmapManager.d_completeTrajectory, numFrames);
	m_currentState.m_numCompleteTransforms = numFrames;
	m_currentState.m_lastValidCompleteTransform = numFrames - 1;
}

void Bundler::solve(float4x4* transforms, SIFTImageManager* siftManager, unsigned int numNonLinIters, unsigned int numLinIters, bool isLocal, bool recordConvergence)
{
	bool useVerify = false; //TODO do we need verify?
	m_SparseBundler.align(siftManager, transforms, numNonLinIters, numLinIters, useVerify, isLocal, recordConvergence);
	//if (useVerify) bundle->verifyTrajectory();
}

void Bundler::matchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, 
	unsigned int frameStart, unsigned int frameSkip, bool print /*= false*/) // frameStart/frameSkip for debugging (printing matches)
{
	bool isLocal = (frameSkip == 1);
	const std::vector<int>& validImages = siftManager->getValidImages();

	// match with every other
	const unsigned int curFrame = siftManager->getNumImages() - 1;
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		uint2 keyPointOffset = make_uint2(0, 0);
		ImagePairMatch& imagePairMatch = siftManager->getImagePairMatch(prev, keyPointOffset);

		SIFTImageGPU& image_i = siftManager->getImageGPU(prev);
		SIFTImageGPU& image_j = siftManager->getImageGPU(curFrame);
		int num1 = (int)siftManager->getNumKeyPointsPerImage(prev);
		int num2 = (int)siftManager->getNumKeyPointsPerImage(curFrame);

		if (validImages[prev] == 0 || num1 == 0 || num2 == 0) {
			unsigned int numMatch = 0;
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(imagePairMatch.d_numMatches, &numMatch, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
		else {
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
			m_siftMatcher->SetDescriptors(0, num1, (unsigned char*)image_i.d_keyPointDescs);
			m_siftMatcher->SetDescriptors(1, num2, (unsigned char*)image_j.d_keyPointDescs);
			m_siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset);
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeSiftMatching = s_timer.getElapsedTimeMS(); }
		}
	}

	if (curFrame > 0) { // can have a match to another frame

		// --- sort the current key point matches
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		siftManager->SortKeyPointMatchesCU(curFrame);
		//if (print) printCurrentMatches("debug/", siftManager, false, frameStart, frameSkip);

		// --- filter matches
		//SIFTMatchFilter::filterKeyPointMatches(siftManager);
		siftManager->FilterKeyPointMatchesCU(curFrame);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterKeyPoint = s_timer.getElapsedTimeMS(); }

		// --- surface area filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		//const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
		//SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		siftManager->FilterMatchesBySurfaceAreaCU(curFrame, MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsicsInv()), GlobalBundlingState::get().s_surfAreaPcaThresh);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterSurfaceArea = s_timer.getElapsedTimeMS(); }

		// --- dense verify filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		//SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);
		const CUDACachedFrame* cachedFramesCUDA = cudaCache->getCacheFramesGPU();
		siftManager->FilterMatchesByDenseVerifyCU(curFrame, cudaCache->getWidth(), cudaCache->getHeight(), MatrixConversion::toCUDA(cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifySiftErrThresh, GlobalBundlingState::get().s_verifySiftCorrThresh);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterDenseVerify = s_timer.getElapsedTimeMS(); }

		// --- filter frames
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		//!!!
		//SIFTMatchFilter::filterFrames(siftManager);
		siftManager->filterFrames(curFrame);
		//!!!
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeFilterFrames = s_timer.getElapsedTimeMS(); }
		if (print) printCurrentMatches("debug/filt", siftManager, true, frameStart, frameSkip);

		// --- add to global correspondences
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		siftManager->AddCurrToResidualsCU(curFrame, MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsicsInv()));
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeAddCurrResiduals = s_timer.getElapsedTimeMS(); }
	}
}

void Bundler::printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame) const
{
	//TODO get color cpu for these functions
	CUDAImageManager::ManagedRGBDInputFrame& integrateFrame = m_CudaImageManager->getIntegrateFrame(allFrame);

	ColorImageR8G8B8A8 im(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(im.getPointer(), integrateFrame.getColorFrameGPU(), sizeof(uchar4) * m_CudaImageManager->getIntegrationWidth() * m_CudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
	im.reSample(m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());

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
	curImage.reSample(m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		CUDAImageManager::ManagedRGBDInputFrame& prevIntegrateFrame = m_CudaImageManager->getIntegrateFrame(prev * frameSkip + frameStart);
		ColorImageR8G8B8A8 prevImage(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevImage.getPointer(), prevIntegrateFrame.getColorFrameGPU(),
			sizeof(uchar4) * prevImage.getNumPixels(), cudaMemcpyDeviceToHost));
		prevImage.reSample(m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered);
	}
}

void Bundler::saveKeysToPointCloud(const std::string& filename /*= "refined.ply"*/) const
{
	if (GlobalBundlingState::get().s_recordKeysPointCloud) {
		const std::vector<int>& validImagesGlobal = m_SubmapManager.global->getValidImages();
		std::vector<mat4f> globalTrajectory(m_SubmapManager.global->getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(globalTrajectory.data(), m_SubmapManager.d_globalTrajectory, sizeof(float4x4)*globalTrajectory.size(), cudaMemcpyDeviceToHost));
		
		m_RGBDSensor->saveRecordedPointCloud(filename, validImagesGlobal, globalTrajectory);
		//!!!
		//unsigned int numFrames = (m_SubmapManager.global->getNumImages() > 0) ? (m_SubmapManager.global->getNumImages() - 1)*m_submapSize + m_currentState.m_lastNumLocalFrames : m_currentState.m_lastNumLocalFrames;
		//std::cout << "found " << numFrames << " total frames" << std::endl;
		//std::vector<mat4f> completeTrajectory(numFrames);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(completeTrajectory.data(), m_SubmapManager.d_completeTrajectory, sizeof(float4x4)*completeTrajectory.size(), cudaMemcpyDeviceToHost));
		//m_RGBDSensor->saveRecordedPointCloudDEBUG(filename, validImagesGlobal, completeTrajectory, m_submapSize);
		//!!!
	}
}

void Bundler::saveCompleteTrajectory(const std::string& filename) const
{
	std::vector<mat4f> completeTrajectory(m_currentState.m_numCompleteTransforms);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(completeTrajectory.data(), m_SubmapManager.d_completeTrajectory, sizeof(mat4f)*completeTrajectory.size(), cudaMemcpyDeviceToHost));

	BinaryDataStreamFile s(filename, true);
	s << completeTrajectory;
	s.closeStream();
}

void Bundler::saveSiftTrajectory(const std::string& filename) const
{
	std::vector<mat4f> siftTrjectory(m_currentState.m_numCompleteTransforms);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(siftTrjectory.data(), m_SubmapManager.d_siftTrajectory, sizeof(mat4f)*siftTrjectory.size(), cudaMemcpyDeviceToHost));

	BinaryDataStreamFile s(filename, true);
	s << siftTrjectory;
	s.closeStream();
}

void Bundler::saveIntegrateTrajectory(const std::string& filename)
{
	std::vector<mat4f> integrateTrajectory(m_currentState.m_numCompleteTransforms);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(integrateTrajectory.data(), m_SubmapManager.getCurrIntegrateTransform(0), sizeof(mat4f)*integrateTrajectory.size(), cudaMemcpyDeviceToHost));

	BinaryDataStreamFile s(filename, true);
	s << integrateTrajectory;
	s.closeStream();
}

//void Bundler::saveDEBUG()
//{
//	const std::vector<int> validImages(m_SubmapManager.global->getNumImages() * m_submapSize, 1);
//	std::vector<mat4f> siftTrajectory(validImages.size());
//	MLIB_CUDA_SAFE_CALL(cudaMemcpy(siftTrajectory.data(), m_SubmapManager.d_siftTrajectory, sizeof(float4x4)*siftTrajectory.size(), cudaMemcpyDeviceToHost));
//	m_RGBDSensor->saveRecordedPointCloud("test.ply", validImages, siftTrajectory);
//}

