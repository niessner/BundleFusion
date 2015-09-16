
#include "stdafx.h"
#include "SBA.h"
#include "TimingLog.h"
#include "GlobalBundlingState.h"

#define POSESIZE 6


void SBA::align(SIFTImageManager* siftManager, std::vector<ml::mat4f>& transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal)
{
	m_bVerify = false;

	std::cout << "[ align ]" << std::endl;
	m_maxResidual = -1.0f;

	ml::Timer timer;
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.start(); }
	bool removed = false;
	const unsigned int maxIts = 60;//GlobalAppState::get().s_maxNumResidualsRemoved; //!!!TODO PARAMS
	unsigned int curIt = 0;
	do {
		removed = alignCUDA(siftManager, transforms, maxNumIters, numPCGits, useVerify);
		curIt++;
	} while (removed && curIt < maxIts);

	//if (GlobalAppState::get().s_enableDetailedTimings) m_solver->evaluateTimings();
	
	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); timer.stop(); TimingLog::getFrameTiming(isLocal).timeSolve = timer.getElapsedTimeMS(); TimingLog::getFrameTiming(isLocal).numItersSolve = curIt * maxNumIters; }
	std::cout << "[ align Time:] " << timer.getElapsedTimeMS() << " ms" << std::endl;

}

bool SBA::alignCUDA(SIFTImageManager* siftManager, std::vector<ml::mat4f>& transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool useVerify)
{
	EntryJ* d_correspondences = siftManager->getGlobalCorrespondencesDEBUG();
	unsigned int numCorrespondences = siftManager->getNumGlobalCorrespondences();

	m_numCorrespondences = numCorrespondences;
	// transforms
	unsigned int numImages = (unsigned int)transforms.size();
	std::vector<Pose> poses = PoseHelper::convertToPoses(transforms);
	std::vector<ml::vec3f> rotations(numImages), translations(numImages);
	for (unsigned int i = 0; i < numImages; i++) {
		rotations[i] = ml::vec3f(poses[i][0], poses[i][1], poses[i][2]);
		translations[i] = ml::vec3f(poses[i][3], poses[i][4], poses[i][5]);
	}
	cutilSafeCall(cudaMemcpy(d_xRot, rotations.data(), sizeof(float3) * numImages, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_xTrans, translations.data(), sizeof(float3) * numImages, cudaMemcpyHostToDevice));

	m_solver->solve(d_correspondences, m_numCorrespondences, numImages, numNonLinearIterations, numLinearIterations, d_xRot, d_xTrans);

	cutilSafeCall(cudaMemcpy(rotations.data(), d_xRot, sizeof(float3) * numImages, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(translations.data(), d_xTrans, sizeof(float3) * numImages, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < numImages; i++)
		poses[i] = Pose(rotations[i].x, rotations[i].y, rotations[i].z, translations[i].x, translations[i].y, translations[i].z);
	transforms = PoseHelper::convertToMatrices(poses);

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
