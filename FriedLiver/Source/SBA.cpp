
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

	m_maxResidual = -1.0f;

#ifdef USE_GLOBAL_DENSE_EVERY_FRAME
	m_bUseGlobalDenseOpt = true;
#else
	m_bUseGlobalDenseOpt = false;
#endif
	m_globalWeightsMutex.unlock();

	m_bUseLocalDense = true;
}


bool SBA::align(SIFTImageManager* siftManager, const CUDACache* cudaCache, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal,
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
	//if (isStart) siftManager->updateGPUValidImages(); //TODO CHECK
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
	return removed;
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
		const unsigned int curFrame = (revalidateIdx == (unsigned int)-1) ? siftManager->getCurrentFrame() : revalidateIdx;
		removed = removeMaxResidualCUDA(siftManager, numImages, curFrame);
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

#include "SiftVisualization.h"
template<>
struct std::hash<vec2ui> : public std::unary_function < vec2ui, size_t > {
	size_t operator()(const vec2ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		//const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);// ^ ((size_t)v.z * p2);
		return res;
	}
};

//!!!debugging
bool SBA::removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages, unsigned int curFrame)
{
	ml::vec2ui imageIndices;
	bool remove = m_solver->getMaxResidual(curFrame, siftManager->getGlobalCorrespondencesGPU(), imageIndices, m_maxResidual);
	if (remove) {
		if (GlobalBundlingState::get().s_verbose) std::cout << "\timages (" << imageIndices << "): invalid match " << m_maxResidual << std::endl;

		/*if (false) {
			static SensorData sd; 
			if (sd.m_frames.empty()) sd.loadFromFile(GlobalAppState::get().s_binaryDumpSensorFile);
			//get residual corrs
			std::vector<EntryJ> correspondences(siftManager->getNumGlobalCorrespondences());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(correspondences.data(), siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyDeviceToHost));
			//get transforms
			float4x4* d_transforms = NULL;
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_transforms, sizeof(float4x4)*numImages))
			convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, siftManager->getValidImagesGPU());
			std::vector<mat4f> transforms(numImages);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*numImages, cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_FREE(d_transforms);
			//get residuals
			std::vector< std::pair<float, vec2ui> > residuals;
			for (unsigned int i = 0; i < correspondences.size(); i++) {
				const EntryJ& corr = correspondences[i];
				if (corr.isValid()) {
					vec3f res = transforms[corr.imgIdx_i] * vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z) - transforms[corr.imgIdx_j] * vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
					res = math::abs(res);
					float r = std::max(res.x, std::max(res.y, res.z));
					residuals.push_back(std::make_pair(r, vec2ui(corr.imgIdx_i, corr.imgIdx_j)));
				}
			} //correspondences
			std::sort(residuals.begin(), residuals.end(), [](const std::pair<float, vec2ui> &left, const std::pair<float, vec2ui> &right) {
				return fabs(left.first) > fabs(right.first);
			});
			//find image-image 
			const float thresh = 0.05f;
			std::unordered_map<vec2ui, float> imageImageResidualsSet;
			for (unsigned int i = 0; i < residuals.size(); i++) {
				if (residuals[i].first < thresh) break;
				auto it = imageImageResidualsSet.find(residuals[i].second);
				if (it == imageImageResidualsSet.end()) imageImageResidualsSet[residuals[i].second] = residuals[i].first;
				else it->second = std::max(it->second, residuals[i].first);
			}//residuals
			const unsigned int maxToPrint = 10;
			residuals.clear();
			for (const auto& a : imageImageResidualsSet) residuals.push_back(std::make_pair(a.second, a.first));
			std::sort(residuals.begin(), residuals.end(), [](const std::pair<float, vec2ui> &left, const std::pair<float, vec2ui> &right) {
				return fabs(left.first) > fabs(right.first);
			});
			if (residuals.size() > maxToPrint) residuals.resize(maxToPrint);
			std::cout << "printing " << residuals.size() << " high residual" << std::endl;
			for (const auto& impair : residuals) {
				vec3uc* im1 = sd.decompressColorAlloc(impair.second.x * 10);
				vec3uc* im2 = sd.decompressColorAlloc(impair.second.y * 10);
				ColorImageR8G8B8 image1(sd.m_colorWidth, sd.m_colorHeight, im1);
				ColorImageR8G8B8 image2(sd.m_colorWidth, sd.m_colorHeight, im2);
				std::free(im1);		std::free(im2);
				SiftVisualization::printMatch("debug/maxres/" + std::to_string((int)(100 * impair.first)) + "_" + std::to_string(impair.second.x) + "-" + std::to_string(impair.second.y) + ".png",
					impair.second, correspondences, image1, image2, sd.m_calibrationColor.m_intrinsic);
				unsigned short* d1 = sd.decompressDepthAlloc(impair.second.x * 10);
				unsigned short* d2 = sd.decompressDepthAlloc(impair.second.y * 10);
				DepthImage32 depth1(DepthImage16(sd.m_depthWidth, sd.m_depthHeight, d1));
				DepthImage32 depth2(DepthImage16(sd.m_depthWidth, sd.m_depthHeight, d2));
				std::free(d1);		std::free(d2);
				image1.resize(depth1.getWidth(), depth1.getHeight());	image2.resize(depth2.getWidth(), depth2.getHeight());
				SiftVisualization::saveKeyMatchToPointCloud("debug/maxres/" + std::to_string((int)(100 * impair.first)) + "_" + std::to_string(impair.second.x) + "-" + std::to_string(impair.second.y),
					impair.second, correspondences, depth1, image1, depth2, image2, transforms, sd.m_calibrationDepth.m_intrinsic.getInverse());
			} //print
			std::cout << "waiting..." << std::endl;
			getchar();
		}*/
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
		//	//std::ofstream s("debug/logs/" + std::to_string(siftManager->getNumImages()) + "-" + std::to_string(siftManager->getCurrentFrame()) + ".txt");
		//	//s << imageIndices << " : " << m_maxResidual << std::endl;
		//	//if (siftManager->getCurrentFrame() >= 70) {
		//	//float4x4* d_transforms = NULL;
		//	//MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_transforms, sizeof(float4x4)*numImages));
		//	//convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, siftManager->getValidImagesGPU());
		//	//std::vector<mat4f> transforms(numImages);
		//	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*numImages, cudaMemcpyDeviceToHost));
		//	//const std::vector<int>& valid = siftManager->getValidImages();
		//	//for (unsigned int i = 0; i < numImages; i++) if (valid[i] == 0) transforms[i].setZero(-std::numeric_limits<float>::infinity());
		//	//BinaryDataStreamFile ofs("debug/logs/" + std::to_string(siftManager->getCurrentFrame()) + ".trajectory", true);
		//	//ofs << transforms; ofs.closeStream();
		//	//MLIB_CUDA_SAFE_FREE(d_transforms);
		//	//siftManager->saveToFile("debug/logs/" + std::to_string(siftManager->getCurrentFrame()) + ".sift");
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