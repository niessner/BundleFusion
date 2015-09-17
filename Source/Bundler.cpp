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


Timer Bundler::s_timer;


void Bundler::init(RGBDSensor* sensor)
{
	// init CUDA
	m_CudaImageManager = new CUDAImageManager(GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
		GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, sensor);
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	m_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage, submapSize, m_CudaImageManager);
	//TODO fix
	if (GlobalAppState::get().s_sensorIdx == 3) {
		m_SubmapManager.setTotalNumFrames(((BinaryDumpReader*)sensor)->getNumTotalFrames());
	}
	m_SparseBundler.init(GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumCorrPerImage);

	m_sift = new SiftGPU;
	m_siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	m_sift->SetParams(0, GlobalBundlingState::get().s_enableDetailedTimings, 150);
	m_sift->InitSiftGPU();
	m_siftMatcher->InitSiftMatch();

	m_submapSize = GlobalBundlingState::get().s_submapSize;
}

void Bundler::destroy()
{
	SAFE_DELETE(m_sift);
	SAFE_DELETE(m_siftMatcher);
	SAFE_DELETE(m_CudaImageManager);
}

bool Bundler::process(RGBDSensor* sensor)
{
	if (!m_CudaImageManager->process()) return false;

	const unsigned int curFrame = m_CudaImageManager->getCurrFrameNumber();
	std::cout << "[ frame " << curFrame << " ]" << std::endl;
		
	if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addLocalFrameTiming();

	// run SIFT
	SIFTImageGPU& cur = m_SubmapManager.currentLocal->createSIFTImageGPU();
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
	int success = m_sift->RunSIFT(m_CudaImageManager->getIntensityImage(), m_CudaImageManager->getDepthInput());
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection on frame " + std::to_string(curFrame));
	unsigned int numKeypoints = m_sift->GetKeyPointsAndDescriptorsCUDA(cur, m_CudaImageManager->getDepthInput());
	m_SubmapManager.currentLocal->finalizeSIFTImageGPU(numKeypoints);
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(true).timeSiftDetection = s_timer.getElapsedTimeMS(); }

	// process cuda cache
	const unsigned int curLocalFrame = m_SubmapManager.currentLocal->getNumImages() - 1;
	m_SubmapManager.currentLocalCache->storeFrame(m_CudaImageManager->getLastIntegrateDepth(), m_CudaImageManager->getLastIntegrateColor(), m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	if (GlobalBundlingState::get().s_recordKeysPointCloud && curLocalFrame == 0 || m_SubmapManager.isLastLocalFrame(curFrame)) {
		sensor->recordPointCloud();
	}
	//printKey("key" + std::to_string(curLocalFrame) + ".png", curFrame, g_SubmapManager.currentLocal, curLocalFrame);

	// local submaps
	if (m_SubmapManager.isLastLocalFrame(curFrame)) {
		SIFTImageGPU& curNext = m_SubmapManager.nextLocal->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(curNext.d_keyPoints, cur.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(curNext.d_keyPointDescs, cur.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		m_SubmapManager.nextLocal->finalizeSIFTImageGPU(numKeypoints);

		m_SubmapManager.nextLocalCache->copyCacheFrameFrom(m_SubmapManager.currentLocalCache, curLocalFrame);
	}

	// match with every other local
	SIFTImageManager* currentLocal = m_SubmapManager.currentLocal;
	std::vector<int> validImagesLocal; currentLocal->getValidImagesDEBUG(validImagesLocal);
	if (curLocalFrame > 0) {
		matchAndFilter(currentLocal, m_SubmapManager.currentLocalCache, validImagesLocal, curFrame - curLocalFrame, 1);
	}

	// global frame
	if (m_SubmapManager.isLastFrame(curFrame) || m_SubmapManager.isLastLocalFrame(curFrame)) { // end frame or global frame
		if (GlobalBundlingState::get().s_enableGlobalTimings) TimingLog::addGlobalFrameTiming();

		// cache
		m_SubmapManager.globalCache->copyCacheFrameFrom(m_SubmapManager.currentLocalCache, 0);

		// if valid local
		if (validImagesLocal[1]) {
			const unsigned int curLocalIdx = m_SubmapManager.getCurrLocal(curFrame);
			// solve local
			solve(m_SubmapManager.getLocalTrajectoryGPU(curLocalIdx), m_SubmapManager.currentLocal, true);

			// fuse to global
			SIFTImageManager* global = m_SubmapManager.global;
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
			SIFTImageGPU& curGlobalImage = global->createSIFTImageGPU();
			unsigned int numGlobalKeys = m_SubmapManager.currentLocal->FuseToGlobalKeyCU(curGlobalImage, m_SubmapManager.getLocalTrajectoryGPU(curLocalIdx),
				MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsics()), MatrixConversion::toCUDA(m_CudaImageManager->getSIFTIntrinsicsInv()));
			global->finalizeSIFTImageGPU(numGlobalKeys);
			if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(false).timeSiftDetection = s_timer.getElapsedTimeMS(); }

			//unsigned int gframe = (unsigned int)global->getNumImages() - 1;
			//printKey("debug/keys/" + std::to_string(gframe) + ".png", gframe*submapSize, global, gframe);

			// switch local submaps
			m_SubmapManager.switchLocal();

			// match with every other global
			std::vector<int> validImagesGlobal; global->getValidImagesDEBUG(validImagesGlobal);
			if (global->getNumImages() > 1) {
				matchAndFilter(global, m_SubmapManager.globalCache, validImagesGlobal, 0, m_submapSize);
				//printCurrentMatches("output/matches/", binaryDumpReader, global, true, 0, submapSize);

				if (validImagesGlobal.back()) {
					// solve global
					solve(m_SubmapManager.d_globalTrajectory, global, false);
				}
			}

			// complete trajectory
			m_SubmapManager.updateTrajectory(curFrame);
			m_SubmapManager.initializeNextGlobalTransform(false);
		}
		else {
			m_SubmapManager.updateTrajectory(curFrame);
			m_SubmapManager.initializeNextGlobalTransform(true);
		}
	} // global
	
	return true;
}

void Bundler::solve(float4x4* transforms, SIFTImageManager* siftManager, bool isLocal)
{
	bool useVerify = false; //TODO do we need verify?
	m_SparseBundler.align(siftManager, transforms, GlobalBundlingState::get().s_numNonLinIterations, GlobalBundlingState::get().s_numLinIterations, useVerify, isLocal);
	//if (useVerify) bundle->verifyTrajectory();
}

void Bundler::matchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, const std::vector<int>& validImages,
	unsigned int frameStart, unsigned int frameSkip, bool print /*= false*/) // frameStart/frameSkip for debugging (printing matches)
{
	bool isLocal = (frameSkip == 1);

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
		SIFTMatchFilter::filterFrames(siftManager);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeFilterFrames = s_timer.getElapsedTimeMS(); }
		if (print) printCurrentMatches("debug/filt", siftManager, true, frameStart, frameSkip);

		// --- add to global correspondences
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
		siftManager->AddCurrToResidualsCU(curFrame);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeAddCurrResiduals = s_timer.getElapsedTimeMS(); }
	}
}

void Bundler::printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame)
{
	ColorImageR8G8B8A8 im(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(im.getPointer(), m_CudaImageManager->getIntegrateColor(allFrame), sizeof(uchar4) * m_CudaImageManager->getIntegrationWidth() * m_CudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
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

void Bundler::printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices, const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered)
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

void Bundler::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered, unsigned int frameStart, unsigned int frameSkip)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	unsigned int curFrame = numFrames - 1;
	ColorImageR8G8B8A8 curImage(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curImage.getPointer(), m_CudaImageManager->getIntegrateColor(curFrame * frameSkip + frameStart),
		sizeof(uchar4) * curImage.getNumPixels(), cudaMemcpyDeviceToHost));
	curImage.reSample(m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		ColorImageR8G8B8A8 prevImage(m_CudaImageManager->getIntegrationWidth(), m_CudaImageManager->getIntegrationHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevImage.getPointer(), m_CudaImageManager->getIntegrateColor(prev * frameSkip + frameStart),
			sizeof(uchar4) * prevImage.getNumPixels(), cudaMemcpyDeviceToHost));
		prevImage.reSample(m_CudaImageManager->getSIFTWidth(), m_CudaImageManager->getSIFTHeight());

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered);
	}
}

void Bundler::saveKeysToPointCloud(RGBDSensor* sensor, const std::string& filename /*= "refined.ply"*/)
{
	if (GlobalBundlingState::get().s_recordKeysPointCloud) {
		std::vector<int> validImagesGlobal; m_SubmapManager.global->getValidImagesDEBUG(validImagesGlobal);
		std::vector<mat4f> globalTrajectory(m_SubmapManager.global->getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(globalTrajectory.data(), m_SubmapManager.d_globalTrajectory, sizeof(float4x4)*globalTrajectory.size(), cudaMemcpyDeviceToHost));
		sensor->saveRecordedPointCloud(filename, validImagesGlobal, globalTrajectory);
	}
}
