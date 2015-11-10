
#include "stdafx.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/MatrixConversion.h"
#include "SiftGPU/SIFTMatchFilter.h"
#include "GlobalAppState.h"
#include "SiftVisualization.h"

#include "SubmapManager.h"



SubmapManager::SubmapManager()
{
	m_sift = NULL;
	m_siftMatcher = NULL;

	m_currentLocal = NULL;
	m_nextLocal = NULL;
	m_optLocal = NULL;
	m_global = NULL;
	m_numTotalFrames = 0;
	m_submapSize = 0;

	m_currentLocalCache = NULL;
	m_nextLocalCache = NULL;
	m_globalCache = NULL;
	m_optLocalCache = NULL;
	//m_globalTimer = NULL;

	d_globalTrajectory = NULL;
	d_completeTrajectory = NULL;
	d_localTrajectories = NULL;

	d_siftTrajectory = NULL;

	_debugPrintMatches = false;
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
	m_SparseBundler.init(GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumCorrPerImage);

	// cache
	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;

	const float scaleWidth = (float)downSampWidth / (float)imageManager->getIntegrationWidth();
	const float scaleHeight = (float)downSampHeight / (float)imageManager->getIntegrationHeight();
	mat4f intrinsicsDownsampled = imageManager->getIntrinsics();
	intrinsicsDownsampled._m00 *= scaleWidth;  intrinsicsDownsampled._m02 *= scaleWidth;
	intrinsicsDownsampled._m11 *= scaleHeight; intrinsicsDownsampled._m12 *= scaleHeight;

	m_currentLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	m_nextLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	m_optLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages, intrinsicsDownsampled);
	m_globalCache = new CUDACache(downSampWidth, downSampHeight, maxNumGlobalImages, intrinsicsDownsampled);

	m_numTotalFrames = numTotalFrames;
	m_submapSize = submapSize;

	// sift manager
	m_currentLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	m_nextLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	m_optLocal = new SIFTImageManager(m_submapSize, maxNumLocalImages, maxNumKeysPerImage);
	m_global = new SIFTImageManager(m_submapSize, maxNumGlobalImages, maxNumKeysPerImage);

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

	SAFE_DELETE(m_currentLocal);
	SAFE_DELETE(m_nextLocal);
	SAFE_DELETE(m_optLocal);
	SAFE_DELETE(m_global);

	SAFE_DELETE(m_currentLocalCache);
	SAFE_DELETE(m_nextLocalCache);
	SAFE_DELETE(m_optLocalCache);
	SAFE_DELETE(m_globalCache);

	MLIB_CUDA_SAFE_FREE(d_globalTrajectory);
	MLIB_CUDA_SAFE_FREE(d_completeTrajectory);
	MLIB_CUDA_SAFE_FREE(d_localTrajectories);

	MLIB_CUDA_SAFE_FREE(d_imageInvalidateList);
	MLIB_CUDA_SAFE_FREE(d_siftTrajectory);
	MLIB_CUDA_SAFE_FREE(d_currIntegrateTransform);
}

unsigned int SubmapManager::runSIFT(unsigned int curFrame, float* d_intensitySIFT, const float* d_inputDepth, unsigned int depthWidth, unsigned int depthHeight, const uchar4* d_inputColor, unsigned int colorWidth, unsigned int colorHeight)
{
	SIFTImageGPU& curImage = m_currentLocal->createSIFTImageGPU();
	int success = m_sift->RunSIFT(d_intensitySIFT, d_inputDepth);
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection");
	unsigned int numKeypoints = m_sift->GetKeyPointsAndDescriptorsCUDA(curImage, d_inputDepth);
	m_currentLocal->finalizeSIFTImageGPU(numKeypoints);

	// process cuda cache
	const unsigned int curLocalFrame = m_currentLocal->getNumImages() - 1;
	m_currentLocalCache->storeFrame(d_inputDepth, depthWidth, depthHeight, d_inputColor, colorWidth, colorHeight);

	// init next
	if (isLastLocalFrame(curFrame)) {
		mutex_nextLocal.lock();
		SIFTImageGPU& nextImage = m_nextLocal->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(nextImage.d_keyPoints, curImage.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(nextImage.d_keyPointDescs, curImage.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		m_nextLocal->finalizeSIFTImageGPU(numKeypoints);
		m_nextLocalCache->copyCacheFrameFrom(m_currentLocalCache, curLocalFrame);
		mutex_nextLocal.unlock();
	}

	return curLocalFrame;
}

bool SubmapManager::matchAndFilter(bool isLocal, SIFTImageManager* siftManager, CUDACache* cudaCache, const float4x4& siftIntrinsicsInv)
{
	const std::vector<int>& validImages = siftManager->getValidImages();
	Timer timer;

	// match with every other
	const unsigned int curFrame = siftManager->getNumImages() - 1;
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
	if (isLocal) m_mutexMatcher.lock();
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
			if (!isLocal) m_mutexMatcher.lock();
			m_siftMatcher->SetDescriptors(0, num1, (unsigned char*)image_i.d_keyPointDescs);
			m_siftMatcher->SetDescriptors(1, num2, (unsigned char*)image_j.d_keyPointDescs);
			float ratioMax = isLocal ? GlobalBundlingState::get().s_siftMatchRatioMaxLocal : GlobalBundlingState::get().s_siftMatchRatioMaxGlobal;
			m_siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset, GlobalBundlingState::get().s_siftMatchThresh, ratioMax);
			if (!isLocal) m_mutexMatcher.unlock();
		}
	}
	if (isLocal) m_mutexMatcher.unlock();
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeSiftMatching = timer.getElapsedTimeMS(); }

	bool lastValid = true;
	if (curFrame > 0) { // can have a match to another frame

		// --- sort the current key point matches
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		siftManager->SortKeyPointMatchesCU(curFrame);

		//!!!DEBUGGING
		std::vector<unsigned int> _numRawMatches; 
		if (_debugPrintMatches) {
			siftManager->getNumRawMatchesDEBUG(_numRawMatches);
			if (isLocal) SiftVisualization::printCurrentMatches("debug/rawMatches/", siftManager, cudaCache, false);
			else SiftVisualization::printCurrentMatches("debug/rawMatchesGlobal/", siftManager, cudaCache, false);

			if (!isLocal) SiftVisualization::printCurrentMatches("debug/", siftManager, cudaCache, false, 5);
		}
		//!!!DEBUGGING

		// --- filter matches
		const unsigned int minNumMatches = isLocal ? GlobalBundlingState::get().s_minNumMatchesLocal : GlobalBundlingState::get().s_minNumMatchesGlobal;
		//!!!DEBUGGING
		SIFTMatchFilter::ransacKeyPointMatches(siftManager, siftIntrinsicsInv, minNumMatches, GlobalBundlingState::get().s_maxKabschResidual2, false);
		//!!!DEBUGGING
		//siftManager->FilterKeyPointMatchesCU(curFrame, siftIntrinsicsInv, minNumMatches, GlobalBundlingState::get().s_maxKabschResidual2, (_debugPrintMatches && !isLocal));
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterKeyPoint = timer.getElapsedTimeMS(); }

		//!!!DEBUGGING
		std::vector<unsigned int> _numFiltMatches;
		if (_debugPrintMatches) siftManager->getNumFiltMatchesDEBUG(_numFiltMatches);
		//!!!DEBUGGING

		// --- surface area filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
		//SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		siftManager->FilterMatchesBySurfaceAreaCU(curFrame, siftIntrinsicsInv, GlobalBundlingState::get().s_surfAreaPcaThresh);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterSurfaceArea = timer.getElapsedTimeMS(); }

		//!!!DEBUGGING
		std::vector<unsigned int> _numFiltMatchesSA;
		if (_debugPrintMatches) siftManager->getNumFiltMatchesDEBUG(_numFiltMatchesSA);
		//!!!DEBUGGING

		// --- dense verify filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);
		const CUDACachedFrame* cachedFramesCUDA = cudaCache->getCacheFramesGPU();
		siftManager->FilterMatchesByDenseVerifyCU(curFrame, cudaCache->getWidth(), cudaCache->getHeight(), MatrixConversion::toCUDA(cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifySiftErrThresh, GlobalBundlingState::get().s_verifySiftCorrThresh,
			//GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax); //TODO PARAMS
			0.1f, 3.0f);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeMatchFilterDenseVerify = timer.getElapsedTimeMS(); }

		//!!!DEBUGGING
		std::vector<unsigned int> _numFiltMatchesDV;
		if (_debugPrintMatches) {
			siftManager->getNumFiltMatchesDEBUG(_numFiltMatchesDV);
			if (isLocal) SiftVisualization::printCurrentMatches("debug/filtMatches/", siftManager, cudaCache, true);
			else SiftVisualization::printCurrentMatches("debug/filtMatchesGlobal/", siftManager, cudaCache, true);
		}
		//!!!DEBUGGING

		// --- filter frames
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//!!!DEBUGGING
		//SIFTMatchFilter::filterFrames(siftManager);
		siftManager->filterFrames(curFrame);
		//!!!DEBUGGING
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeFilterFrames = timer.getElapsedTimeMS(); }
		//if (print) SiftVisualization::printCurrentMatches("debug/filt", siftManager, true, frameStart, frameSkip);

		// --- add to global correspondences
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		if (siftManager->getValidImages()[curFrame] != 0)
			siftManager->AddCurrToResidualsCU(curFrame, siftIntrinsicsInv);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeAddCurrResiduals = timer.getElapsedTimeMS(); }

		if (siftManager->getValidImages()[curFrame] == 0) lastValid = false;
	}

	return lastValid;
}

bool SubmapManager::isCurrentLocalValidChunk()
{
	bool valid = false;
	if (m_currentLocal->getValidImages()[1] != 0) valid = true;
	return valid;
}

unsigned int SubmapManager::getNumNextLocalFrames()
{
	mutex_nextLocal.lock();
	unsigned int numFrames = std::min(m_submapSize, m_nextLocal->getNumImages());
	mutex_nextLocal.unlock();
	return numFrames;
}

void SubmapManager::getCacheIntrinsics(float4x4& intrinsics, float4x4& intrinsicsInv)
{
	intrinsics = MatrixConversion::toCUDA(m_currentLocalCache->getIntrinsics());
	intrinsicsInv = MatrixConversion::toCUDA(m_currentLocalCache->getIntrinsicsInv());
}

void SubmapManager::copyToGlobalCache()
{
	m_globalCache->copyCacheFrameFrom(m_nextLocalCache, 0);
}

bool SubmapManager::optimizeLocal(unsigned int curLocalIdx, unsigned int numNonLinIterations, unsigned int numLinIterations)
{
	//_idxsLocalOptimized.push_back(curLocalIdx);

	bool ret = false;

	//m_SubmapManager.optLocal->lock();
	mutex_nextLocal.lock();
	SIFTImageManager* siftManager = m_nextLocal;
	CUDACache* cudaCache = m_nextLocalCache;

	MLIB_ASSERT(m_nextLocal->getNumImages() > 1);
	bool useVerify = true;
	m_SparseBundler.align(siftManager, getLocalTrajectoryGPU(curLocalIdx), numNonLinIterations, numLinIterations,
		useVerify, true, false, true, true, false);
	// still need this for global key fuse

	//!!!DEBUGGING
	//if (curLocalIdx >= 20) {
	//	saveOptToPointCloud("debug/local-" + std::to_string(curLocalIdx) + ".ply", m_nextLocalCache, m_nextLocal->getValidImages(), getLocalTrajectoryGPU(curLocalIdx), m_nextLocal->getNumImages());
	//}
	//!!!DEBUGGING

	// verify
	if (m_SparseBundler.useVerification()) {
		const CUDACachedFrame* cachedFramesCUDA = cudaCache->getCacheFramesGPU();
		int valid = siftManager->VerifyTrajectoryCU(siftManager->getNumImages(), getLocalTrajectoryGPU(curLocalIdx),
			cudaCache->getWidth(), cudaCache->getHeight(), MatrixConversion::toCUDA(cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifyOptErrThresh, GlobalBundlingState::get().s_verifyOptCorrThresh,
			//GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax); //TODO PARAMS
			0.1f, 3.0f);

		if (valid == 0) {
			////!!!DEBUGGING
			//_idxsLocalInvalidVerify.push_back(curLocalIdx);
			saveOptToPointCloud("debug/opt-" + std::to_string(curLocalIdx) + ".ply", m_nextLocalCache, m_nextLocal->getValidImages(), getLocalTrajectoryGPU(curLocalIdx), m_nextLocal->getNumImages());
			//std::cout << "SAVED " << curLocalIdx << std::endl;
			////!!!DEBUGGING

			if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: invalid local submap from verify " << curLocalIdx << std::endl;
			//!!!
			std::cout << "WARNING: invalid local submap from verify " << curLocalIdx << std::endl;
			getchar();
			//!!!
			ret = false;
		}
		else
			ret = true;
	}
	else
		ret = true;
	//m_SubmapManager.optLocal->unlock();

	mutex_nextLocal.unlock();
	return ret;
}

int SubmapManager::computeAndMatchGlobalKeys(unsigned int lastLocalSolved, const float4x4& siftIntrinsics, const float4x4& siftIntrinsicsInv)
{
	int ret = 0; ////!!!TODO FIX
	if ((int)m_global->getNumImages() <= lastLocalSolved) {
		//m_SubmapManager.optLocal->lock();
		mutex_nextLocal.lock();
		SIFTImageManager* local = m_nextLocal;

		// fuse to global
		Timer timer;
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
		//SIFTImageGPU& curGlobalImage = m_global->createSIFTImageGPU();
		//unsigned int numGlobalKeys = local->FuseToGlobalKeyCU(curGlobalImage, getLocalTrajectoryGPU(lastLocalSolved),
		//	siftIntrinsics, siftIntrinsicsInv);
		//m_global->finalizeSIFTImageGPU(numGlobalKeys);
		local->fuseToGlobal(m_global, siftIntrinsics, getLocalTrajectoryGPU(lastLocalSolved));
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(false).timeSiftDetection = timer.getElapsedTimeMS(); }

		const std::vector<int>& validImagesLocal = local->getValidImages();
		for (unsigned int i = 0; i < std::min(m_submapSize, local->getNumImages()); i++) {
			if (validImagesLocal[i] == 0)
				invalidateImages((m_global->getNumImages() - 1) * m_submapSize + i);
		}
		initializeNextGlobalTransform(false);
		// done with local data!
		finishLocalOpt();
		//m_SubmapManager.optLocal->unlock();
		mutex_nextLocal.unlock();

		//unsigned int gframe = (unsigned int)global->getNumImages() - 1;
		//printKey("debug/keys/" + std::to_string(gframe) + ".png", gframe*submapSize, global, gframe);

		// match with every other global
		if (m_global->getNumImages() > 1) {
			//!!!DEBUGGING
			//if (m_global->getNumImages() == 7) {
			//	setPrintMatchesDEBUG(true);
			//	int a = 5;
			//}
			//!!!DEBUGGING
			matchAndFilter(false, m_global, m_globalCache, siftIntrinsicsInv);
			//!!!DEBUGGING
			//if (m_global->getNumImages() == 7) {
			//	setPrintMatchesDEBUG(false);
			//	int a = 5;
			//	std::cout << "waiting..." << std::endl;
			//	getchar();
			//}
			//!!!DEBUGGING

			if (m_global->getValidImages()[m_global->getNumImages() - 1]) {
				// ready to solve global
				ret = 1;
			}
			else {
				if (GlobalBundlingState::get().s_verbose) std::cout << "WARNING: last image (" << m_global->getNumImages() << ") not valid! no new global images for solve" << std::endl;
				//!!!
				std::cout << "WARNING: last image (" << m_global->getNumImages() << ") not valid! no new global images for solve" << std::endl;
				getchar();
				//!!!
				ret = 2;
			}
		}
		else {
			ret = 0;
		}
	}
	return ret;
}

void SubmapManager::addInvalidGlobalKey()
{
	SIFTImageGPU& curGlobalImage = m_global->createSIFTImageGPU();
	m_global->finalizeSIFTImageGPU(0);
	mutex_nextLocal.lock();
	finishLocalOpt();
	mutex_nextLocal.unlock();
	initializeNextGlobalTransform(true);
}

bool SubmapManager::optimizeGlobal(unsigned int numFrames, unsigned int numNonLinIterations, unsigned int numLinIterations, bool isStart, bool isEnd, bool isScanDone)
{
	//_idxsGlobalOptimized.push_back(m_global->getNumImages() - 1);

	bool ret = false;
	const unsigned int numGlobalFrames = m_global->getNumImages();

	const bool useVerify = false;
	m_SparseBundler.align(m_global, d_globalTrajectory, numNonLinIterations, numLinIterations,
		useVerify, false, GlobalBundlingState::get().s_recordSolverConvergence, isStart, isEnd, isScanDone);

	if (isEnd) {
		//!!!DEBUGGING
		//const std::vector<int> _valid = m_global->getValidImages();
		//if (numFrames >= 200) {
			//saveOptToPointCloud("debug/global-" + std::to_string(getCurrLocal(numFrames)) + ".ply", m_globalCache, m_global->getValidImages(), d_globalTrajectory, m_global->getNumImages());
		//	int a = 5;
		//}
		//!!!DEBUGGING

		// may invalidate already invalidated images
		const std::vector<int>& validImagesGlobal = m_global->getValidImages();
		for (unsigned int i = 0; i < numGlobalFrames; i++) {
			if (validImagesGlobal[i] == 0) {
				invalidateImages(i * m_submapSize, std::min((i + 1)*m_submapSize, numFrames));
			}
		}

		if (validImagesGlobal[numGlobalFrames - 1] != 0) ret = true;
	}
	return ret;
}

void SubmapManager::saveCompleteTrajectory(const std::string& filename, unsigned int numTransforms) const
{
	std::vector<mat4f> completeTrajectory(numTransforms);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(completeTrajectory.data(), d_completeTrajectory, sizeof(mat4f)*completeTrajectory.size(), cudaMemcpyDeviceToHost));

	BinaryDataStreamFile s(filename, true);
	s << completeTrajectory;
	s.closeStream();
}

void SubmapManager::saveSiftTrajectory(const std::string& filename, unsigned int numTransforms) const
{
	std::vector<mat4f> siftTrjectory(numTransforms);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(siftTrjectory.data(), d_siftTrajectory, sizeof(mat4f)*siftTrjectory.size(), cudaMemcpyDeviceToHost));

	BinaryDataStreamFile s(filename, true);
	s << siftTrjectory;
	s.closeStream();
}

void SubmapManager::saveOptToPointCloud(const std::string& filename, const CUDACache* cudaCache, const std::vector<int>& valid, const float4x4* d_transforms, unsigned int numFrames)
{
	// local transforms: d_transforms = getLocalTrajectoryGPU(localIdx);
	// local cudaCache: cudaCache = m_nextLocalCache
	// global transforms: d_transforms = d_globalTrajectory;
	// global cudaCache: cudaCache = m_globalCache

	//transforms
	std::vector<mat4f> transforms(numFrames);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*numFrames, cudaMemcpyDeviceToHost));
	//frames
	ColorImageR32G32B32A32 camPosition;
	ColorImageR8G8B8A8 color;
	camPosition.allocate(cudaCache->getWidth(), cudaCache->getHeight());
	color.allocate(cudaCache->getWidth(), cudaCache->getHeight());
	const std::vector<CUDACachedFrame>& cacheFrames = cudaCache->getCacheFrames();

	bool saveFrameByFrame = true;

	PointCloudf pc;
	for (unsigned int f = 0; f < numFrames; f++) {
		if (valid[f] == 0) continue;

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPosition.getPointer(), cacheFrames[f].d_cameraposDownsampled, sizeof(float4)*camPosition.getNumPixels(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(color.getPointer(), cacheFrames[f].d_colorDownsampled, sizeof(uchar4)*color.getNumPixels(), cudaMemcpyDeviceToHost));

		PointCloudf framePc;

		for (unsigned int i = 0; i < camPosition.getNumPixels(); i++) {
			const vec4f& p = camPosition.getPointer()[i];
			if (p.x != -std::numeric_limits<float>::infinity()) {
				pc.m_points.push_back(transforms[f] * p.getVec3());
				const vec4uc& c = color.getPointer()[i];
				pc.m_colors.push_back(vec4f(c.x, c.y, c.z, c.w) / 255.f);

				if (saveFrameByFrame) {
					framePc.m_points.push_back(pc.m_points.back());
					framePc.m_colors.push_back(pc.m_colors.back());
				}
			}
		}
		if (saveFrameByFrame) {
			PointCloudIOf::saveToFile("debug/frames/" + std::to_string(f) + ".ply", framePc);
		}
	}
	PointCloudIOf::saveToFile(filename, pc);
}

void SubmapManager::saveVerifyDEBUG(const std::string& prefix) const
{
	{
		std::ofstream s(prefix + "_localOpt.txt");
		s << _idxsLocalOptimized.size() << " local optimized" << std::endl;
		for (unsigned int i = 0; i < _idxsLocalOptimized.size(); i++)
			s << _idxsLocalOptimized[i] << std::endl;
		s.close();
	}
		{
			std::ofstream s(prefix + "_localFailVerify.txt");
			s << _idxsLocalInvalidVerify.size() << " local failed verify" << std::endl;
			for (unsigned int i = 0; i < _idxsLocalInvalidVerify.size(); i++)
				s << _idxsLocalInvalidVerify[i] << std::endl;
			s.close();
		}
		{
			std::ofstream s(prefix + "_globalOptimized.txt");
			s << _idxsGlobalOptimized.size() << " global optimized" << std::endl;
			for (unsigned int i = 0; i < _idxsGlobalOptimized.size(); i++)
				s << _idxsGlobalOptimized[i] << std::endl;
			s.close();
		}
}

