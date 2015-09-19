
#include "stdafx.h"
#include "SBA.h"
#include "TimingLog.h"
#include "GlobalBundlingState.h"

#define POSESIZE 6

extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans);

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms);

Timer SBA::s_timer;

void SBA::align(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal, bool recordConvergence /*= false*/)
{
	if (recordConvergence) m_recordedConvergence.push_back(std::vector<float>());

	m_bVerify = false;

	//std::cout << "[ align ]" << std::endl;
	m_maxResidual = -1.0f;

	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.start(); }
	bool removed = false;
	const unsigned int maxIts = 60;//GlobalAppState::get().s_maxNumResidualsRemoved; //!!!TODO PARAMS
	unsigned int curIt = 0;
	do {
		removed = alignCUDA(siftManager, d_transforms, maxNumIters, numPCGits, useVerify);
		if (recordConvergence) {
			const std::vector<float>& conv = m_solver->getConvergenceAnalysis();
			m_recordedConvergence.back().insert(m_recordedConvergence.back().end(), conv.begin(), conv.end());
		}
		curIt++;
	} while (removed && curIt < maxIts);
	
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(isLocal).timeSolve = s_timer.getElapsedTimeMS(); TimingLog::getFrameTiming(isLocal).numItersSolve = curIt * maxNumIters; }
	//std::cout << "[ align Time:] " << s_timer.getElapsedTimeMS() << " ms" << std::endl;

}

bool SBA::alignCUDA(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool useVerify)
{
	EntryJ* d_correspondences = siftManager->getGlobalCorrespondencesDEBUG();
	unsigned int numCorrespondences = siftManager->getNumGlobalCorrespondences();

	m_numCorrespondences = numCorrespondences;
	// transforms
	unsigned int numImages = siftManager->getNumImages();
	convertMatricesToPosesCU(d_transforms, numImages, d_xRot, d_xTrans);

	m_solver->solve(d_correspondences, m_numCorrespondences, numImages, numNonLinearIterations, numLinearIterations, d_xRot, d_xTrans);

	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms);

	bool removed = removeMaxResidualCUDA(siftManager, numImages);

	if (!removed && useVerify) m_bVerify = m_solver->useVerification(d_correspondences, m_numCorrespondences);

	return removed;
}

bool SBA::removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages)
{
	ml::vec2ui imageIndices;
	bool remove = m_solver->getMaxResidual(siftManager->getGlobalCorrespondencesDEBUG(), imageIndices, m_maxResidual);
	if (remove) {
		std::cout << "\timages (" << imageIndices << "): invalid match " << m_maxResidual << std::endl;
		// invalidate correspondence
		siftManager->InvalidateImageToImageCU(make_uint2(imageIndices.x, imageIndices.y));
		siftManager->CheckForInvalidFramesCU(m_solver->getVarToCorrNumEntriesPerRow(), numImages); // need to re-adjust for removed matches
		return true;
	}
	else std::cout << "\thighest residual " << m_maxResidual << " from images (" << imageIndices << ")" << std::endl;
	return false;
}

void SBA::printConvergence(const std::string& filename)
{
	if (m_recordedConvergence.empty()) return;
	std::ofstream s(filename);
	s << m_recordedConvergence.size() << " optimizations" << std::endl;
	s << std::endl;
	for (unsigned int i = 0; i < m_recordedConvergence.size(); i++) {
		s << "[ opt# " << i << " ]" << std::endl;
		for (unsigned int k = 0; k < m_recordedConvergence[i].size(); k++)
			s << "\titer " << k << ": " << m_recordedConvergence[i][k] << std::endl;
		s << std::endl;
	}
	s.close();
}
