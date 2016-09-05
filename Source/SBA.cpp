
#include "stdafx.h"
#include "SBA.h"
#include "CUDACache.h"

#ifdef USE_GPU_SOLVE
#include "TimingLog.h"
#include "GlobalBundlingState.h"

#define POSESIZE 6

extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans, const int* d_validImages);

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms, const int* d_validImages);

Timer SBA::s_timer;


SBA::SBA()
{
	d_xRot = NULL;
	d_xTrans = NULL;
	m_solver = NULL;

	m_bUseComprehensiveFrameInvalidation = false;

	const unsigned int maxNumIts = std::max(GlobalBundlingState::get().s_numGlobalNonLinIterations, GlobalBundlingState::get().s_numLocalNonLinIterations);
	m_localWeightsSparse.resize(maxNumIts, 1.0f);
	m_localWeightsDenseDepth.resize(maxNumIts);
	for (unsigned int i = 0; i < maxNumIts; i++) m_localWeightsDenseDepth[i] = (i + 1.0f);
	m_localWeightsDenseColor.resize(maxNumIts, 0.0f); //no color
	//for (unsigned int i = 2; i < maxNumIts; i++) m_localWeightsDenseColor[i] = 1.0f;//fr3_nstn
	//// for tum data
	//std::cout << "using FR2 params" << std::endl;
	//m_localWeightsSparse.resize(maxNumIts, 1.0f);
	//m_localWeightsDenseDepth.resize(maxNumIts, 1.0f); //for (unsigned int i = 1; i < maxNumIts; i++) m_localWeightsDenseDepth[i] = 1.0f;
	//m_localWeightsDenseColor.resize(maxNumIts, 0.0f); //for (unsigned int i = 2; i < maxNumIts; i++) m_localWeightsDenseColor[i] = 1.0f;
	////for (unsigned int i = 0; i < 2; i++) m_localWeightsSparse[maxNumIts - i - 1] = 0.0f; // turn off sparse at end

	m_globalWeightsMutex.lock();
	m_globalWeightsSparse.resize(maxNumIts, 1.0f);
	m_globalWeightsDenseDepth.resize(maxNumIts, 1.0f);
	m_globalWeightsDenseColor.resize(maxNumIts, 0.0f); //off
	//// for tum data
	//m_globalWeightsSparse.resize(maxNumIts, 1.0f);
	//m_globalWeightsDenseDepth.resize(maxNumIts, 0.0f);
	//m_globalWeightsDenseColor.resize(maxNumIts, 0.1f); //off

#ifdef USE_GLOBAL_DENSE_EVERY_FRAME
	m_bUseGlobalDenseOpt = true;
#else
	m_bUseGlobalDenseOpt = false;
#endif
	m_globalWeightsMutex.unlock();

	m_bUseLocalDense = true;
}


void SBA::align(SIFTImageManager* siftManager, const CUDACache* cudaCache, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal,
	bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx /*= (unsigned int)-1*/)
{
	if (recordConvergence) m_recordedConvergence.push_back(std::vector<float>());

	m_bVerify = false;
	m_maxResidual = -1.0f;

	//dense opt params
	bool usePairwise; const CUDACache* cache = cudaCache;
	std::vector<float> weightsDenseDepth, weightsDenseColor, weightsSparse;
	if (isLocal) {
		weightsSparse = m_localWeightsSparse;
		usePairwise = true; //always use pairwise
		if (m_bUseLocalDense) {
			weightsDenseDepth = m_localWeightsDenseDepth; //turn on
			weightsDenseColor = m_localWeightsDenseColor;
		}
		else {
			cache = NULL; //to turn off
			weightsDenseDepth = std::vector<float>(m_localWeightsDenseDepth.size(), 0.0f); weightsDenseColor = weightsDenseDepth;
		}
	}
	else {
		usePairwise = true; //always global dense pairwise

		if (!m_bUseGlobalDenseOpt) {
			weightsSparse = m_globalWeightsSparse;
			cache = NULL;
			weightsDenseDepth = std::vector<float>(m_globalWeightsDenseDepth.size(), 0.0f); weightsDenseColor = weightsDenseDepth;
		}
		else {
			m_globalWeightsMutex.lock();
			weightsSparse = m_globalWeightsSparse;
			weightsDenseDepth = m_globalWeightsDenseDepth;
			weightsDenseColor = m_globalWeightsDenseColor;
			m_globalWeightsMutex.unlock();
		}
	}

	if (!isScanDoneOpt && GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }

	unsigned int numImages = siftManager->getNumImages();
	if (isStart) siftManager->updateGPUValidImages();
	const int* d_validImages = siftManager->getValidImagesGPU();
	convertMatricesToPosesCU(d_transforms, numImages, d_xRot, d_xTrans, d_validImages);

	bool removed = alignCUDA(siftManager, cache, usePairwise, weightsSparse, weightsDenseDepth, weightsDenseColor, maxNumIters, numPCGits, isStart, isEnd, revalidateIdx);
	if (recordConvergence) {
		const std::vector<float>& conv = m_solver->getConvergenceAnalysis();
		m_recordedConvergence.back().insert(m_recordedConvergence.back().end(), conv.begin(), conv.end());
	}

	if (useVerify) {
		if (weightsSparse.front() > 0) m_bVerify = m_solver->useVerification(siftManager->getGlobalCorrespondencesGPU(), siftManager->getNumGlobalCorrespondences());
		else m_bVerify = true; //TODO this should not happen except for debugging
	}

	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, d_validImages);

	if (!isScanDoneOpt && GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeSolve += s_timer.getElapsedTimeMS(); TimingLog::getFrameTiming(isLocal).numItersSolve += maxNumIters; }
}

bool SBA::alignCUDA(SIFTImageManager* siftManager, const CUDACache* cudaCache, bool useDensePairwise, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor,
	unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd, unsigned int revalidateIdx)
{
	EntryJ* d_correspondences = siftManager->getGlobalCorrespondencesGPU();
	m_numCorrespondences = siftManager->getNumGlobalCorrespondences();

	// transforms
	unsigned int numImages = siftManager->getNumImages();
	m_solver->solve(d_correspondences, m_numCorrespondences, siftManager->getValidImagesGPU(), numImages, numNonLinearIterations, numLinearIterations,
		cudaCache, weightsSparse, weightsDenseDepth, weightsDenseColor, useDensePairwise, d_xRot, d_xTrans, isStart, isEnd, revalidateIdx); //isStart -> rebuild jt, isEnd -> remove max residual

	bool removed = false;
	if (isEnd && weightsSparse.front() > 0) {
		removed = removeMaxResidualCUDA(siftManager, numImages);
	}

	return removed;
}

//!!!debugging
#include "GlobalAppState.h"

#include "MatrixConversion.h"
#ifdef NEW_GUIDED_REMOVE
template<>
struct std::hash<ml::vec2ui> : public std::unary_function < ml::vec2ui, size_t > {
	size_t operator()(const ml::vec2ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		//const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);// ^ ((size_t)v.z * p2);
		return res;
	}
};
#endif
//!!!debugging
bool SBA::removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages)
{
	ml::vec2ui imageIndices;
	bool remove = m_solver->getMaxResidual(siftManager->getGlobalCorrespondencesGPU(), imageIndices, m_maxResidual);
	if (remove) {
		if (GlobalBundlingState::get().s_verbose) std::cout << "\timages (" << imageIndices << "): invalid match " << m_maxResidual << std::endl;

#ifdef NEW_GUIDED_REMOVE
		const std::vector<vec2ui>& imPairsToRemove = m_solver->getGuidedMaxResImagesToRemove();
		if (imPairsToRemove.empty()) {
			siftManager->InvalidateImageToImageCU(make_uint2(imageIndices.x, imageIndices.y));
			//_logRemovedImImCorrs.push_back(std::make_pair(imageIndices, m_maxResidual));
		}
		else {
			std::cout << "guided remove (" << imPairsToRemove.size() << ")" << std::endl;
			//getchar();
			for (unsigned int i = 0; i < imPairsToRemove.size(); i++) {
				siftManager->InvalidateImageToImageCU(make_uint2(imPairsToRemove[i].x, imPairsToRemove[i].y));
				//_logRemovedImImCorrs.push_back(std::make_pair(imPairsToRemove[i], -1.0f)); //unknown res
			}
		}
#else
		////!!!debugging
		//{
		//	std::ofstream s("debug/_logs/" + std::to_string(siftManager->getNumImages()) + "-" + std::to_string(siftManager->getCurrentFrame()) + ".txt");
		//	s << imageIndices << " : " << m_maxResidual << std::endl;
		//	//if (siftManager->getCurrentFrame() >= 70) {
		//	//	float4x4* d_transforms = NULL;
		//	//	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_transforms, sizeof(float4x4)*numImages));
		//	//	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, siftManager->getValidImagesGPU());
		//	//	std::vector<mat4f> transforms(numImages);
		//	//	MLIB_CUDA_SAFE_CALL(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*numImages, cudaMemcpyDeviceToHost));
		//	//	const std::vector<int>& valid = siftManager->getValidImages();
		//	//	for (unsigned int i = 0; i < numImages; i++) if (valid[i] == 0) transforms[i].setZero(-std::numeric_limits<float>::infinity());
		//	//	BinaryDataStreamFile ofs("debug/_logs/" + std::to_string(siftManager->getCurrentFrame()) + ".trajectory", true);
		//	//	ofs << transforms; ofs.closeStream();
		//	//	MLIB_CUDA_SAFE_FREE(d_transforms);
		//	//	siftManager->saveToFile("debug/_logs/" + std::to_string(siftManager->getCurrentFrame()) + ".sift");
		//	//}
		//	//if (siftManager->getCurrentFrame() == 71) {
		//	//	std::cout << "waiting..." << std::endl;
		//	//	getchar();
		//	//}
		//}
		////!!!debugging

		// invalidate correspondence
		siftManager->InvalidateImageToImageCU(make_uint2(imageIndices.x, imageIndices.y));
		//_logRemovedImImCorrs.push_back(std::make_pair(imageIndices, m_maxResidual));
#endif
		if (m_bUseComprehensiveFrameInvalidation)
			siftManager->CheckForInvalidFramesCU(m_solver->getVarToCorrNumEntriesPerRow(), numImages); // need to re-adjust for removed matches
		else
			siftManager->CheckForInvalidFramesSimpleCU(m_solver->getVarToCorrNumEntriesPerRow(), numImages); // faster but not completely accurate

		return true;
	}
	//else std::cout << "\thighest residual " << m_maxResidual << " from images (" << imageIndices << ")" << std::endl;
	return false;
}

void SBA::printConvergence(const std::string& filename) const
{
	//if (m_recordedConvergence.empty()) return;
	//std::ofstream s(filename);
	//s << m_recordedConvergence.size() << " optimizations" << std::endl;
	//s << std::endl;
	//for (unsigned int i = 0; i < m_recordedConvergence.size(); i++) {
	//	s << "[ opt# " << i << " ]" << std::endl;
	//	for (unsigned int k = 0; k < m_recordedConvergence[i].size(); k++)
	//		s << "\titer " << k << ": " << m_recordedConvergence[i][k] << std::endl;
	//	s << std::endl;
	//}
	//s.close();

	if (m_recordedConvergence.empty()) return;
	std::ofstream s(filename);
	for (unsigned int i = 0; i < m_recordedConvergence.size(); i++) {
		for (unsigned int k = 0; k < m_recordedConvergence[i].size(); k++)
			s << m_recordedConvergence[i][k] << std::endl;
	}
	s.close();
}

#endif