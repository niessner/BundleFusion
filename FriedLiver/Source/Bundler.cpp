#include "stdafx.h"
#include "Bundler.h"
#include "SiftGPU/SiftGPU.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/MatrixConversion.h"
#include "SiftGPU/SIFTMatchFilter.h"
#include "CUDAImageManager.h"
#include "CUDACache.h"

#include "mLibCuda.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"
#include "ConditionManager.h"
#include "TimingLog.h"

//for debugging
#include "SiftVisualization.h"

Bundler::Bundler(unsigned int maxNumImages, unsigned int maxNumKeysPerImage,
	const mat4f& siftIntrinsicsInv, const CUDAImageManager* manager, bool isLocal)
{
	//initialize sift
	initSift(GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, isLocal);
	m_siftIntrinsicsInv = MatrixConversion::toCUDA(siftIntrinsicsInv);
	m_siftIntrinsics = m_siftIntrinsicsInv.getInverse();
	m_bIsLocal = isLocal;

	//initialize optimizer
	const unsigned int maxNumResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (maxNumImages*(maxNumImages - 1)) / 2;
	m_optimizer.init(maxNumImages, maxNumResiduals);

	//dense tracking
	const unsigned int cacheInputWidth = manager->getSIFTDepthWidth();
	const unsigned int cacheInputHeight = manager->getSIFTDepthHeight();
	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;
	const mat4f cacheInputIntrinsics = manager->getSIFTDepthIntrinsics();
	m_cudaCache = new CUDACache(cacheInputWidth, cacheInputHeight, downSampWidth, downSampHeight, maxNumImages, cacheInputIntrinsics);

	//sparse tracking
	m_siftManager = new SIFTImageManager(maxNumImages, maxNumKeysPerImage);

	//trajectories
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_trajectory, sizeof(float4x4)*maxNumImages));
	std::vector<mat4f> identityTrajectory(maxNumImages, mat4f::identity()); //initialize transforms to identity
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory, identityTrajectory.data(), sizeof(float4x4)*maxNumImages, cudaMemcpyHostToDevice));

	m_continueRetry = 0;
	m_revalidatedIdx = (unsigned int)-1;

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	m_corrEvaluator = NULL;
#endif
}


void Bundler::initSift(unsigned int widthSift, unsigned int heightSift, bool isLocal)
{
	if (isLocal) {
		m_sift = new SiftGPU;
		m_sift->SetParams(widthSift, heightSift, false, 150, GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax);
		m_sift->InitSiftGPU();
	}
	else {
		m_sift = NULL; //don't need detection for global
	}
	m_siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	m_siftMatcher->InitSiftMatch();
}

Bundler::~Bundler()
{
	SAFE_DELETE(m_sift);
	SAFE_DELETE(m_siftMatcher);

	SAFE_DELETE(m_siftManager);
	SAFE_DELETE(m_cudaCache);

	MLIB_CUDA_SAFE_FREE(d_trajectory);
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	SAFE_DELETE(m_corrEvaluator);
#endif
}

void Bundler::getCacheIntrinsics(float4x4& intrinsics, float4x4& intrinsicsInv)
{
	intrinsics = MatrixConversion::toCUDA(m_cudaCache->getIntrinsics());
	intrinsicsInv = MatrixConversion::toCUDA(m_cudaCache->getIntrinsicsInv());
}

void Bundler::detectFeatures(float* d_intensitySift, const float* d_inputDepthFilt)
{
	SIFTImageGPU& cur = m_siftManager->createSIFTImageGPU();
	int success = m_sift->RunSIFT(d_intensitySift, d_inputDepthFilt);
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection");
	unsigned int numKeypoints = m_sift->GetKeyPointsAndDescriptorsCUDA(cur, d_inputDepthFilt, m_siftManager->getMaxNumKeyPointsPerImage());

	if (numKeypoints > GlobalBundlingState::get().s_maxNumKeysPerImage) throw MLIB_EXCEPTION("too many keypoints"); //should never happen

	m_siftManager->finalizeSIFTImageGPU(numKeypoints);
}

unsigned int Bundler::matchAndFilter()
{
	const unsigned int numFrames = m_siftManager->getNumImages();
	MLIB_ASSERT(numFrames > 1);
	const unsigned int curFrame = m_siftManager->getCurrentFrame();

	const std::vector<int>& validImages = m_siftManager->getValidImages();

	// match with every other //TODO CLASS for image match proposals
	const unsigned int startFrame = numFrames == curFrame + 1 ? 0 : curFrame + 1;
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
	int num2 = (int)m_siftManager->getNumKeyPointsPerImage(curFrame);
	if (num2 == 0) return (unsigned int)-1;

	for (unsigned int prev = startFrame; prev < numFrames; prev++) {
		if (prev == curFrame) continue;
		uint2 keyPointOffset = make_uint2(0, 0);
		ImagePairMatch& imagePairMatch = m_siftManager->getImagePairMatch(prev, curFrame, keyPointOffset);

		SIFTImageGPU& image_i = m_siftManager->getImageGPU(prev);
		SIFTImageGPU& image_j = m_siftManager->getImageGPU(curFrame);
		int num1 = (int)m_siftManager->getNumKeyPointsPerImage(prev);

		if (validImages[prev] == 0 || num1 == 0 || num2 == 0) {
			unsigned int numMatch = 0;
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(imagePairMatch.d_numMatches, &numMatch, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
		else {
			m_siftMatcher->SetDescriptors(0, num1, (unsigned char*)image_i.d_keyPointDescs);
			m_siftMatcher->SetDescriptors(1, num2, (unsigned char*)image_j.d_keyPointDescs);
			float ratioMax = m_bIsLocal ? GlobalBundlingState::get().s_siftMatchRatioMaxLocal : GlobalBundlingState::get().s_siftMatchRatioMaxGlobal; //TODO do we need two different here?
			m_siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset, GlobalBundlingState::get().s_siftMatchThresh, ratioMax);
		}
	}
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(m_bIsLocal).timeSiftMatching = m_timer.getElapsedTimeMS(); }

	unsigned int lastMatchedFrame = (unsigned int)-1;
	if (curFrame > 0) { // can have a match to another frame

		// --- sort the current key point matches
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
		m_siftManager->SortKeyPointMatchesCU(curFrame, startFrame, numFrames);
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
		if (m_corrEvaluator) m_corrEvaluator->evaluate(m_siftManager, m_cudaCache, MatrixConversion::toMlib(m_siftIntrinsicsInv), false, true, false, "raw");
#endif
		////debugging
		//const bool usedebug = true;//!m_bIsLocal;// && curFrame >= 49;
		//std::vector<unsigned int> numMatches;
		//if (usedebug) {
		//	m_siftManager->getNumRawMatchesDEBUG(numMatches);
		//	SiftVisualization::printCurrentMatches("debug/rawMatches/", m_siftManager, m_cudaCache, false);
		//	int a = 5;
		//}
		////debugging

		// --- filter matches
		const unsigned int minNumMatches = m_bIsLocal ? GlobalBundlingState::get().s_minNumMatchesLocal : GlobalBundlingState::get().s_minNumMatchesGlobal;
		//SIFTMatchFilter::ransacKeyPointMatches(siftManager, siftIntrinsicsInv, minNumMatches, GlobalBundlingState::get().s_maxKabschResidual2, false);
		//SIFTMatchFilter::filterKeyPointMatches(siftManager, siftIntrinsicsInv, minNumMatches);
		m_siftManager->FilterKeyPointMatchesCU(curFrame, startFrame, numFrames, m_siftIntrinsicsInv, minNumMatches, GlobalBundlingState::get().s_maxKabschResidual2);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(m_bIsLocal).timeMatchFilterKeyPoint = m_timer.getElapsedTimeMS(); }
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
		if (m_corrEvaluator) m_corrEvaluator->evaluate(m_siftManager, m_cudaCache, MatrixConversion::toMlib(m_siftIntrinsicsInv), true, false, false, "kabsch");
#endif
		////debugging
		//if (usedebug) {
		//	m_siftManager->getNumFiltMatchesDEBUG(numMatches);
		//	SiftVisualization::printCurrentMatches("debug/matchesKeyFilt/", m_siftManager, m_cudaCache, true);
		//	int a = 5;
		//}
		////debugging

		// --- surface area filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
		//const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
		//SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		m_siftManager->FilterMatchesBySurfaceAreaCU(curFrame, startFrame, numFrames, m_siftIntrinsicsInv, GlobalBundlingState::get().s_surfAreaPcaThresh);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(m_bIsLocal).timeMatchFilterSurfaceArea = m_timer.getElapsedTimeMS(); }
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
		if (m_corrEvaluator) m_corrEvaluator->evaluate(m_siftManager, m_cudaCache, MatrixConversion::toMlib(m_siftIntrinsicsInv), true, false, false, "sa");
#endif
		////debugging
		//if (usedebug) {
		//	m_siftManager->getNumFiltMatchesDEBUG(numMatches);
		//	SiftVisualization::printCurrentMatches("debug/matchesSAFilt/", m_siftManager, m_cudaCache, true);
		//	int a = 5;
		//}
		////debugging

		// --- dense verify filter
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
		//SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);
		const CUDACachedFrame* cachedFramesCUDA = m_cudaCache->getCacheFramesGPU();
		m_siftManager->FilterMatchesByDenseVerifyCU(curFrame, startFrame, numFrames, m_cudaCache->getWidth(), m_cudaCache->getHeight(), MatrixConversion::toCUDA(m_cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifySiftErrThresh, GlobalBundlingState::get().s_verifySiftCorrThresh,
			GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax);
		//0.1f, 3.0f);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(m_bIsLocal).timeMatchFilterDenseVerify = m_timer.getElapsedTimeMS(); }
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
		if (m_corrEvaluator) m_corrEvaluator->evaluate(m_siftManager, m_cudaCache, MatrixConversion::toMlib(m_siftIntrinsicsInv), true, false, true, "dense");
#endif
		////debugging
		//if (usedebug) {
		//	m_siftManager->getNumFiltMatchesDEBUG(numMatches);
		//	SiftVisualization::printCurrentMatches("debug/filtMatches/", m_siftManager, m_cudaCache, true);
		//	int a = 5;
		//}
		////debugging

		// --- filter frames
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
		lastMatchedFrame = m_siftManager->filterFrames(curFrame, startFrame, numFrames);
		// --- add to global correspondences
		MLIB_ASSERT((m_siftManager->getValidImages()[curFrame] != 0 && lastMatchedFrame != (unsigned int)-1) || (lastMatchedFrame == (unsigned int)-1 && m_siftManager->getValidImages()[curFrame] == 0)); //TODO REMOVE
		if (lastMatchedFrame != (unsigned int)-1)//if (siftManager->getValidImages()[curFrame] != 0)
			m_siftManager->AddCurrToResidualsCU(curFrame, startFrame, numFrames, m_siftIntrinsicsInv);
		//else lastValid = false;
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(m_bIsLocal).timeMisc = m_timer.getElapsedTimeMS(); }

		if (!m_bIsLocal) { //global only
			if (lastMatchedFrame != (unsigned int)-1 && lastMatchedFrame + 1 != curFrame) { //re-initialize to better location based off of last match
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory + curFrame, d_trajectory + lastMatchedFrame, sizeof(float4x4), cudaMemcpyDeviceToDevice));
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory + curFrame + 1, d_trajectory + lastMatchedFrame, sizeof(float4x4), cudaMemcpyDeviceToDevice));
			}
			//retry
			if (curFrame + 1 == numFrames) { // this is a current frame (and not a retry frame)
				if (lastMatchedFrame != (unsigned int)-1) //1 revalidation per frame 
					tryRevalidation(curFrame, false);
				else {
					if (GlobalBundlingState::get().s_verbose && curFrame + 1 == numFrames)
						std::cout << "WARNING: last image (" << curFrame << ") not valid! no new global images for solve" << std::endl;
					m_siftManager->addToRetryList(curFrame);
				}
			}
		} //global only

		////debugging
		//if (usedebug) {
		//	std::vector<EntryJ> corrs(m_siftManager->getNumGlobalCorrespondences());
		//	if (!corrs.empty()) MLIB_CUDA_SAFE_CALL(cudaMemcpy(corrs.data(), m_siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*corrs.size(), cudaMemcpyDeviceToHost));
		//	int a = 5;
		//}
		////debugging
	}
	return lastMatchedFrame;
}

bool Bundler::optimize(unsigned int numNonLinIterations, unsigned int numLinIterations, bool bUseVerify, bool bRemoveMaxResidual, bool bIsScanDone, bool& bOptRemoved)
{
	MLIB_ASSERT(m_siftManager->getNumImages() > 1);

	bool ret = false;
	bOptRemoved = m_optimizer.align(m_siftManager, m_cudaCache, d_trajectory, numNonLinIterations, numLinIterations, bUseVerify, m_bIsLocal,
		false, true, bRemoveMaxResidual, bIsScanDone, m_revalidatedIdx); //false -> record convergence, true -> buildjt

	if (m_optimizer.useVerification()) {
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.start(); }
		const CUDACachedFrame* cachedFramesCUDA = m_cudaCache->getCacheFramesGPU();
		int valid = m_siftManager->VerifyTrajectoryCU(m_siftManager->getNumImages(), d_trajectory,
			m_cudaCache->getWidth(), m_cudaCache->getHeight(), MatrixConversion::toCUDA(m_cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifyOptErrThresh, GlobalBundlingState::get().s_verifyOptCorrThresh,
			//GlobalAppState::get().s_sensorDepthMin, GlobalAppState::get().s_sensorDepthMax); //TODO PARAMS
			0.1f, 3.0f);
		if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); m_timer.stop(); TimingLog::getFrameTiming(true).timeSolve += m_timer.getElapsedTimeMS(); }
		if (valid > 0)
			ret = true;
		else if (GlobalBundlingState::get().s_verbose)
			std::cout << "WARNING: invalid local submap from verify" << std::endl;
	}
	else ret = true;

	return ret;
}

void Bundler::storeCachedFrame(unsigned int depthWidth, unsigned int depthHeight, const uchar4* d_inputColor, unsigned int colorWidth, unsigned int colorHeight, const float* d_inputDepthRaw)
{
	m_cudaCache->storeFrame(d_inputDepthRaw, depthWidth, depthHeight, d_inputColor, colorWidth, colorHeight);
}

void Bundler::copyFrame(const Bundler* b, unsigned int frame)
{
	SIFTImageGPU& next = m_siftManager->createSIFTImageGPU();
	const SIFTImageGPU& cur = b->m_siftManager->getImageGPU(frame);
	const unsigned int numKeys = b->m_siftManager->getNumKeyPointsPerImage(frame);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(next.d_keyPoints, cur.d_keyPoints, sizeof(SIFTKeyPoint)*numKeys, cudaMemcpyDeviceToDevice));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(next.d_keyPointDescs, cur.d_keyPointDescs, sizeof(SIFTKeyPointDesc)*numKeys, cudaMemcpyDeviceToDevice));
	m_siftManager->finalizeSIFTImageGPU(numKeys);
	m_cudaCache->copyCacheFrameFrom(b->m_cudaCache, frame);
}

bool Bundler::isValid() const
{
	const std::vector<int>& valid = m_siftManager->getValidImages();
	const unsigned int numImages = m_siftManager->getNumImages();
	for (unsigned int i = 1; i < numImages; i++) {//for (unsigned int i = 0; i < numImages; i++) { //TODO allow single frame valid for local
		if (valid[i] != 0)
			return true;
	}
	return false;
}

unsigned int Bundler::tryRevalidation(unsigned int curGlobalFrame, bool bIsScanDone)
{
#ifdef USE_RETRY
	m_revalidatedIdx = (unsigned int)-1;
	if (m_continueRetry < 0) return false; // nothing to do
	// see if have any invalid images which match
	unsigned int idx;
	if (m_siftManager->getTopRetryImage(idx)) {
		if (bIsScanDone) { //TODO CHECK THIS PART
			if (m_continueRetry == 0) {
				m_continueRetry = idx;
			}
			else if (m_continueRetry == idx) {
				m_continueRetry = -1;
				return m_revalidatedIdx; // nothing more to do (looped around again)
			}
		}

		m_siftManager->setCurrentFrame(idx);
		unsigned int lastMatchedGlobal = matchAndFilter();

		if (m_siftManager->getValidImages()[idx] != 0) { //validate
			//initialize to better transform
			MLIB_ASSERT(lastMatchedGlobal != (unsigned int)-1);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory + idx, d_trajectory + lastMatchedGlobal, sizeof(float4x4), cudaMemcpyDeviceToDevice));

			////debugging //TODO ENABLE THIS AND TEST
			//mat4f lastMatchedTransform; mat4f relativeSiftTransform;
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(lastMatchedTransform.getData(), d_trajectory + lastMatchedGlobal, sizeof(float4x4), cudaMemcpyDeviceToHost));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(relativeSiftTransform.getData(), m_siftManager->getFiltTransformsToWorldGPU() + lastMatchedGlobal, sizeof(float4x4), cudaMemcpyDeviceToHost));
			//mat4f initTransform = lastMatchedTransform * relativeSiftTransform;
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory + idx, initTransform.getData(), sizeof(float4x4), cudaMemcpyHostToDevice));
			////debugging

			m_revalidatedIdx = idx;
			if (GlobalBundlingState::get().s_verbose) std::cout << "re-validating " << idx << std::endl;
		}
		else
			m_siftManager->addToRetryList(idx);
		//reset
		m_siftManager->setCurrentFrame(curGlobalFrame);
	}
	return m_revalidatedIdx;
#else
	return (unsigned int)-1;
#endif
}

void Bundler::reset()
{
	std::vector<mat4f> trajectory(m_siftManager->getNumImages(), mat4f::identity());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory, trajectory.data(), sizeof(mat4f)*trajectory.size(), cudaMemcpyHostToDevice));
	m_siftManager->reset();
	m_cudaCache->reset();
}

void Bundler::addInvalidFrame()
{
	m_cudaCache->incrementCache(); //increment cache fake frame
	SIFTImageGPU& cur = m_siftManager->createSIFTImageGPU(); //0 sift keys
	m_siftManager->finalizeSIFTImageGPU(0);
	initializeNextTransformUnknown();
}

const std::vector<int>& Bundler::getValidImages() const
{
	return m_siftManager->getValidImages();
}

void Bundler::invalidateLastFrame()
{
	if (m_siftManager->getNumImages() <= 1) { // can't invalidate first chunk //TODO ALLOW INVALIDATION OF FIRST FRAME
		std::cout << "INVALID FIRST CHUNK" << std::endl;
		std::ofstream s(util::directoryFromPath(GlobalAppState::get().s_binaryDumpSensorFile) + "processed.txt");
		s << "valid = false" << std::endl;
		s << "INVALID_FIRST_CHUNK" << std::endl;
		s.close();
		ConditionManager::setExit();
	}
	m_siftManager->invalidateFrame(m_siftManager->getNumImages() - 1);
}

void Bundler::fuseToGlobal(Bundler* glob)
{
	m_siftManager->fuseToGlobal(glob->m_siftManager, m_siftIntrinsics, d_trajectory, m_siftIntrinsicsInv);	//sparse features
	glob->m_cudaCache->copyCacheFrameFrom(m_cudaCache, 0);													//dense frames
	//fuse local depth frames for global cache //TODO TRY THIS
	//m_cudaCache->fuseDepthFrames(glob->m_cudaCache, m_siftManager->getValidImagesGPU(), d_trajectory); //valid images have been updated in the solve
}

void Bundler::saveSparseCorrsToFile(const std::string& filename) const
{
	UINT64 numCorrs = (UINT64)m_siftManager->getNumGlobalCorrespondences();
	std::vector<EntryJ> corrs(numCorrs);
	if (corrs.empty()) {
		std::cout << "warning: no sparse correspondences to save" << std::endl;
		return;
	}
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(corrs.data(), m_siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*numCorrs, cudaMemcpyDeviceToHost));
	BinaryDataStreamFile s(filename, true);
	s << (UINT64)corrs.size();
	s.writeData((const BYTE*)corrs.data(), sizeof(EntryJ)*numCorrs);
	s.close();
}

