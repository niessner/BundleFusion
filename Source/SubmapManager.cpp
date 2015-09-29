
#include "stdafx.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/MatrixConversion.h"
#include "SiftGPU/SIFTMatchFilter.h"
#include "GlobalAppState.h"

#include "SubmapManager.h"



SubmapManager::SubmapManager()
{
	m_sift = NULL;
	m_siftMatcher = NULL;

	currentLocal = NULL;
	nextLocal = NULL;
	optLocal = NULL;
	global = NULL;
	m_numTotalFrames = 0;
	m_submapSize = 0;

	currentLocalCache = NULL;
	nextLocalCache = NULL;
	globalCache = NULL;
	optLocalCache = NULL;
	//m_globalTimer = NULL;

	d_globalTrajectory = NULL;
	d_completeTrajectory = NULL;
	d_localTrajectories = NULL;

	d_siftTrajectory = NULL;
}

void SubmapManager::initSIFT(unsigned int widthSift, unsigned int heightSift)
{
	m_sift = new SiftGPU;
	m_siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);

	m_sift->SetParams(widthSift, heightSift, false, 150, GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax);
	m_sift->InitSiftGPU();
	m_siftMatcher->InitSiftMatch();
}

void SubmapManager::init(unsigned int maxNumGlobalImages, unsigned int maxNumLocalImages, unsigned int maxNumKeysPerImage, unsigned int submapSize, const CUDAImageManager* imageManager, unsigned int numTotalFrames /*= (unsigned int)-1*/)
{
	initSIFT(GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT);

	// cache
	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;

	const float scaleWidth = (float)downSampWidth / (float)imageManager->getIntegrationWidth();
	const float scaleHeight = (float)downSampHeight / (float)imageManager->getIntegrationHeight();
	mat4f intrinsicsDownsampled = imageManager->getIntrinsics();
	intrinsicsDownsampled._m00 *= scaleWidth;  intrinsicsDownsampled._m02 *= scaleWidth;
	intrinsicsDownsampled._m11 *= scaleHeight; intrinsicsDownsampled._m12 *= scaleHeight;

	currentLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	nextLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	optLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	globalCache = new CUDACache(downSampWidth, downSampHeight, maxNumGlobalImages, intrinsicsDownsampled);

	m_numTotalFrames = numTotalFrames;
	m_submapSize = submapSize;

	// sift manager
	currentLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	nextLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	optLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	global = new SIFTImageManager(m_submapSize, maxNumGlobalImages, maxNumKeysPerImage);

	m_invalidImagesList.resize(maxNumGlobalImages * m_submapSize, 1);

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_globalTrajectory, sizeof(float4x4)*maxNumGlobalImages));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_completeTrajectory, sizeof(float4x4)*maxNumGlobalImages*m_submapSize));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_localTrajectories, sizeof(float4x4)*maxNumLocalImages*maxNumGlobalImages));

	float4x4 id;	id.setIdentity();
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globalTrajectory, &id, sizeof(float4x4), cudaMemcpyHostToDevice)); // set first to identity
	std::vector<mat4f> initialLocalTrajectories(maxNumLocalImages * maxNumGlobalImages, mat4f::identity());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_localTrajectories, initialLocalTrajectories.data(), sizeof(float4x4) * initialLocalTrajectories.size(), cudaMemcpyHostToDevice));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_siftTrajectory, sizeof(float4x4)*maxNumGlobalImages*m_submapSize));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_siftTrajectory, &id, sizeof(float4x4), cudaMemcpyHostToDevice)); // set first to identity

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_currIntegrateTransform, sizeof(float4x4)*maxNumGlobalImages*m_submapSize));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_currIntegrateTransform, &id, sizeof(float4x4), cudaMemcpyHostToDevice)); // set first to identity
	m_currIntegrateTransform.resize(maxNumGlobalImages*m_submapSize);
	m_currIntegrateTransform[0].setIdentity();

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_imageInvalidateList, sizeof(int) * maxNumGlobalImages * maxNumLocalImages));
}

SubmapManager::~SubmapManager()
{
	SAFE_DELETE(m_sift);
	SAFE_DELETE(m_siftMatcher);

	SAFE_DELETE(currentLocal);
	SAFE_DELETE(nextLocal);
	SAFE_DELETE(optLocal);
	SAFE_DELETE(global);

	SAFE_DELETE(currentLocalCache);
	SAFE_DELETE(nextLocalCache);
	SAFE_DELETE(optLocalCache);
	SAFE_DELETE(globalCache);

	MLIB_CUDA_SAFE_FREE(d_globalTrajectory);
	MLIB_CUDA_SAFE_FREE(d_completeTrajectory);
	MLIB_CUDA_SAFE_FREE(d_localTrajectories);

	MLIB_CUDA_SAFE_FREE(d_imageInvalidateList);
	MLIB_CUDA_SAFE_FREE(d_siftTrajectory);
	MLIB_CUDA_SAFE_FREE(d_currIntegrateTransform);
}

std::pair<SIFTImageManager*, CUDACache*> SubmapManager::get(TYPE type)
{
	switch (type) {
	case LOCAL_CURRENT:
		mutex_curLocal.lock();
		return std::make_pair(currentLocal, currentLocalCache);
		break;
	case LOCAL_NEXT:
		mutex_nextLocal.lock();
		return std::make_pair(nextLocal, nextLocalCache);
		break;
	case GLOBAL:
		mutex_global.lock();
		return std::make_pair(global, globalCache);
		break;
	default:
		throw MLIB_EXCEPTION("invalid siftimagemanager query");
	}
}

void SubmapManager::finish(TYPE type)
{
	switch (type) {
	case LOCAL_CURRENT:
		mutex_curLocal.unlock();
		break;
	case LOCAL_NEXT:
		mutex_nextLocal.unlock();
		break;
	case GLOBAL:
		mutex_global.unlock();
		break;
	default:
		throw MLIB_EXCEPTION("invalid siftimagemanager query");
	}
}

unsigned int SubmapManager::runSIFT(unsigned int curFrame, float* d_intensitySIFT, const float* d_inputDepth, unsigned int depthWidth, unsigned int depthHeight, const uchar4* d_inputColor, unsigned int colorWidth, unsigned int colorHeight)
{
	auto& cur = get(LOCAL_CURRENT);

	SIFTImageGPU& curImage = cur.first->createSIFTImageGPU();
	int success = m_sift->RunSIFT(d_intensitySIFT, d_inputDepth);
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection");
	unsigned int numKeypoints = m_sift->GetKeyPointsAndDescriptorsCUDA(curImage, d_inputDepth);
	cur.first->finalizeSIFTImageGPU(numKeypoints);

	// process cuda cache
	const unsigned int curLocalFrame = cur.first->getNumImages() - 1;
	cur.second->storeFrame(d_inputDepth, depthWidth, depthHeight, d_inputColor, colorWidth, colorHeight);

	// init next
	if (isLastLocalFrame(curFrame)) {
		auto& next = get(LOCAL_NEXT);
		SIFTImageGPU& nextImage = next.first->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(nextImage.d_keyPoints, curImage.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(nextImage.d_keyPointDescs, curImage.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		next.first->finalizeSIFTImageGPU(numKeypoints);
		next.second->copyCacheFrameFrom(cur.second, curLocalFrame);
		finish(LOCAL_NEXT);
	}

	finish(LOCAL_CURRENT);
	return curLocalFrame;
}

bool SubmapManager::matchAndFilter(TYPE type, const float4x4& siftIntrinsicsInv)
{
	auto& cur = get(type);
	bool isLocal = (type != GLOBAL);

	SIFTImageManager* siftManager = cur.first;
	const CUDACache* cudaCache = cur.second;
	const std::vector<int>& validImages = siftManager->getValidImages();
	Timer timer;

	// match with every other
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
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
			m_siftMatcher->SetDescriptors(0, num1, (unsigned char*)image_i.d_keyPointDescs);
			m_siftMatcher->SetDescriptors(1, num2, (unsigned char*)image_j.d_keyPointDescs);
			m_siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset);
		}
	}
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeSiftMatching = timer.getElapsedTimeMS(); }

	bool lastValid = true;
	if (curFrame > 0) { // can have a match to another frame

		// --- sort the current key point matches
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		siftManager->SortKeyPointMatchesCU(curFrame);
		//if (print) printCurrentMatches("debug/", siftManager, false, frameStart, frameSkip);

		// --- filter matches
		//SIFTMatchFilter::filterKeyPointMatches(siftManager);
		const unsigned int minNumMatches = isLocal ? GlobalBundlingState::get().s_minNumMatchesLocal : GlobalBundlingState::get().s_minNumMatchesGlobal;
		siftManager->FilterKeyPointMatchesCU(curFrame, siftIntrinsicsInv, minNumMatches);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterKeyPoint = timer.getElapsedTimeMS(); }

		// --- surface area filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
		//SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		siftManager->FilterMatchesBySurfaceAreaCU(curFrame, siftIntrinsicsInv, GlobalBundlingState::get().s_surfAreaPcaThresh);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterSurfaceArea = timer.getElapsedTimeMS(); }

		// --- dense verify filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);
		const CUDACachedFrame* cachedFramesCUDA = cudaCache->getCacheFramesGPU();
		siftManager->FilterMatchesByDenseVerifyCU(curFrame, cudaCache->getWidth(), cudaCache->getHeight(), MatrixConversion::toCUDA(cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifySiftErrThresh, GlobalBundlingState::get().s_verifySiftCorrThresh,
			GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterDenseVerify = timer.getElapsedTimeMS(); }

		// --- filter frames
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//!!!
		//SIFTMatchFilter::filterFrames(siftManager);
		siftManager->filterFrames(curFrame);
		//!!!
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeFilterFrames = timer.getElapsedTimeMS(); }
		//if (print) printCurrentMatches("debug/filt", siftManager, true, frameStart, frameSkip);

		// --- add to global correspondences
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		if (siftManager->getValidImages()[curFrame] != 0)
			siftManager->AddCurrToResidualsCU(curFrame, siftIntrinsicsInv);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeAddCurrResiduals = timer.getElapsedTimeMS(); }

		if (siftManager->getValidImages()[curFrame] == 0) lastValid = false;
	}
	finish(type);

	return lastValid;
}

bool SubmapManager::isCurrentLocalValidChunk()
{
	bool valid = false;
	auto& cur = get(LOCAL_CURRENT);
	if (cur.first->getValidImages()[1] != 0) valid = true;
	finish(LOCAL_CURRENT);
	return valid;
}

unsigned int SubmapManager::getNumCurrentLocalFrames()
{
	auto& cur = get(LOCAL_CURRENT);
	unsigned int numFrames = std::min(m_submapSize, cur.first->getNumImages());
	finish(LOCAL_CURRENT);
	return numFrames;
}

void SubmapManager::getCacheIntrinsics(float4x4& intrinsics, float4x4& intrinsicsInv)
{
	auto& cur = get(LOCAL_CURRENT); 
	intrinsics = MatrixConversion::toCUDA(cur.second->getIntrinsics());
	intrinsicsInv = MatrixConversion::toCUDA(cur.second->getIntrinsicsInv());
	finish(LOCAL_CURRENT);
}

void SubmapManager::copyToGlobalCache()
{
	auto& mCur = get(LOCAL_CURRENT);
	auto& mGlobal = get(GLOBAL);
	mGlobal.second->copyCacheFrameFrom(mCur.second, 0);
	finish(LOCAL_CURRENT);
	finish(GLOBAL);
}
