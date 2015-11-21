
#include "stdafx.h"
#include "SBA.h"

#ifdef USE_GPU_SOLVE
#include "TimingLog.h"
#include "GlobalBundlingState.h"

#define POSESIZE 6

extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans);

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms);

Timer SBA::s_timer;

void SBA::align(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt)
{
	if (recordConvergence) m_recordedConvergence.push_back(std::vector<float>());

	m_bVerify = false;

	m_maxResidual = -1.0f;

	if (!isScanDoneOpt && GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }

	unsigned int numImages = siftManager->getNumImages();
	convertMatricesToPosesCU(d_transforms, numImages, d_xRot, d_xTrans);

	bool removed = false;
	const unsigned int maxIts = 1;//GlobalBundlingState::get().s_maxNumResidualsRemoved;
	unsigned int curIt = 0;
	do {
		removed = alignCUDA(siftManager, d_transforms, maxNumIters, numPCGits, isStart, isEnd);
		if (recordConvergence) {
			const std::vector<float>& conv = m_solver->getConvergenceAnalysis();
			m_recordedConvergence.back().insert(m_recordedConvergence.back().end(), conv.begin(), conv.end());
		}
		curIt++;
	} while (removed && curIt < maxIts);

	if (useVerify) m_bVerify = m_solver->useVerification(siftManager->getGlobalCorrespondencesDEBUG(), siftManager->getNumGlobalCorrespondences());

	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms);
	
	if (!isScanDoneOpt && GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeSolve += s_timer.getElapsedTimeMS(); TimingLog::getFrameTiming(isLocal).numItersSolve += curIt * maxNumIters; }
}

bool SBA::alignCUDA(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd)
{
	EntryJ* d_correspondences = siftManager->getGlobalCorrespondencesDEBUG();
	m_numCorrespondences = siftManager->getNumGlobalCorrespondences();

	// transforms
	unsigned int numImages = siftManager->getNumImages();

	m_solver->solve(d_correspondences, m_numCorrespondences, numImages, numNonLinearIterations, numLinearIterations, d_xRot, d_xTrans, isStart, isEnd);

	bool removed = false; 
	if (isEnd) {
		removed = removeMaxResidualCUDA(siftManager, numImages);
	}

	return removed;
}

bool SBA::removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages)
{
	ml::vec2ui imageIndices;
	bool remove = m_solver->getMaxResidual(siftManager->getGlobalCorrespondencesDEBUG(), imageIndices, m_maxResidual);

	if (remove) {
		if (GlobalBundlingState::get().s_verbose) std::cout << "\timages (" << imageIndices << "): invalid match " << m_maxResidual << std::endl;

		// invalidate correspondence
		siftManager->InvalidateImageToImageCU(make_uint2(imageIndices.x, imageIndices.y));
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