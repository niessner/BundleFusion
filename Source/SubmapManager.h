#pragma once
#ifndef SUBMAP_MANAGER_H
#define SUBMAP_MANAGER_H

#include "SiftGPU/SIFTImageManager.h"
#include "CUDAImageManager.h"
#include "CUDACache.h"
#include "SBA.h"

#include "SiftGPU/CUDATimer.h"
#include "GlobalBundlingState.h"
#include "mLibCuda.h"

//#define DEBUG_PRINT_MATCHING
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
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

class SiftGPU;
class SiftMatchGPU;

extern "C" void updateTrajectoryCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory, unsigned int numLocalTrajectories,
	int* d_imageInvalidateList);

extern "C" void initNextGlobalTransformCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, unsigned int initGlobalIdx,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory); //,unsigned int lastMatchedLocal)

class SubmapManager {
public:
	SubmapManager();
	void init(unsigned int maxNumGlobalImages, unsigned int maxNumLocalImages, unsigned int maxNumKeysPerImage,
		unsigned int submapSize, const CUDAImageManager* manager, unsigned int numTotalFrames = (unsigned int)-1);

	void setTotalNumFrames(unsigned int n) {
		m_numTotalFrames = n;
	}

	~SubmapManager();

	float4x4* getLocalTrajectoryGPU(unsigned int localIdx) const {
		return d_localTrajectories + localIdx * (m_submapSize + 1);
	}

	void invalidateImages(unsigned int startFrame, unsigned int endFrame = -1) {
		if (endFrame == -1) m_invalidImagesList[startFrame] = 0;
		else {
			for (unsigned int i = startFrame; i < endFrame; i++)
				m_invalidImagesList[i] = 0;
		}
	}
	void validateImages(unsigned int startFrame, unsigned int endFrame = -1) {
		if (endFrame == -1) m_invalidImagesList[startFrame] = 1;
		else {
			for (unsigned int i = startFrame; i < endFrame; i++)
				m_invalidImagesList[i] = 1;
		}
	}

	void switchLocal() {
		mutex_nextLocal.lock();
		std::swap(m_currentLocal, m_nextLocal);
		std::swap(m_currentLocalCache, m_nextLocalCache);
		mutex_nextLocal.unlock();
	}

	bool isLastLocalFrame(unsigned int curFrame) const { return (curFrame >= m_submapSize && (curFrame % m_submapSize) == 0); }
	unsigned int getCurrLocal(unsigned int curFrame) const {
		//const unsigned int curLocalIdx = (curFrame + 1 == m_numTotalFrames && (curFrame % m_submapSize != 0)) ? (curFrame / m_submapSize) : (curFrame / m_submapSize) - 1; // adjust for endframe
		//return curLocalIdx;
		return (std::max(curFrame, 1u) - 1) / m_submapSize;
	}

	void computeCurrentSiftTransform(unsigned int frameIdx, unsigned int localFrameIdx, unsigned int lastValidCompleteTransform) {
		const std::vector<int>& validImages = m_currentLocal->getValidImages();
		if (validImages[localFrameIdx] == 0) {
			m_currIntegrateTransform[frameIdx].setZero(-std::numeric_limits<float>::infinity());
			assert(frameIdx > 0);
			cutilSafeCall(cudaMemcpy(d_siftTrajectory + frameIdx, d_siftTrajectory + frameIdx - 1, sizeof(float4x4), cudaMemcpyDeviceToDevice)); //set invalid
		}
		else if (frameIdx > 0) {
			m_currentLocal->computeSiftTransformCU(d_completeTrajectory, lastValidCompleteTransform, d_siftTrajectory, frameIdx, localFrameIdx, d_currIntegrateTransform + frameIdx);
			cutilSafeCall(cudaMemcpy(&m_currIntegrateTransform[frameIdx], d_currIntegrateTransform + frameIdx, sizeof(float4x4), cudaMemcpyDeviceToHost));
		}
	}
	const mat4f& getCurrentIntegrateTransform(unsigned int frameIdx) const { return m_currIntegrateTransform[frameIdx]; }
	const std::vector<mat4f>& getAllIntegrateTransforms() const { return m_currIntegrateTransform; }

	void getCacheIntrinsics(float4x4& intrinsics, float4x4& intrinsicsInv);

	//! run sift for current local
	unsigned int runSIFT(unsigned int curFrame, float* d_intensitySIFT, const float* d_inputDepthFilt,
		unsigned int depthWidth, unsigned int depthHeight, const uchar4* d_inputColor,
		unsigned int colorWidth, unsigned int colorHeight, const float* d_inputDepthRaw);
	//! valid if at least frames 0, 1 valid
	bool isCurrentLocalValidChunk();
	unsigned int getNumNextLocalFrames();
	bool localMatchAndFilter(const float4x4& siftIntrinsicsInv, bool isLastLocal) {
		//!!!debugging
		//if (m_global->getNumImages() >= 63 && m_global->getNumImages() <= 66) {
		//	setPrintMatchesDEBUG(true);
		//}
		//!!!debugging
		unsigned int lastMatchedFrame = matchAndFilter(true, m_currentLocal, m_currentLocalCache, siftIntrinsicsInv);
		if (isLastLocal) {
			m_prevLastValidLocal = m_lastValidLocal;
			if (lastMatchedFrame != (unsigned int)-1) 
				m_lastValidLocal = m_currentLocal->getCurrentFrame();
			else 
				m_lastValidLocal = (unsigned int)-1;
		}
		//!!!debugging
		//if (m_currentLocal->getNumImages() == m_submapSize + 1 && m_global->getNumImages() == 66) {
		//	setPrintMatchesDEBUG(false);
		//}
		//!!!debugging
		return (lastMatchedFrame != (unsigned int)-1);
	}

	void copyToGlobalCache();
	void incrementGlobalCache() { m_globalCache->incrementCache(); }

	//! optimize local
	bool optimizeLocal(unsigned int curLocalIdx, unsigned int numNonLinIterations, unsigned int numLinIterations);
	int computeAndMatchGlobalKeys(unsigned int lastLocalSolved, const float4x4& siftIntrinsics, const float4x4& siftIntrinsicsInv);

	void tryRevalidation(unsigned int curGlobalFrame, const float4x4& siftIntrinsicsInv, bool isScanningDone = false);

	void addInvalidGlobalKey();

	//! optimize global
	bool optimizeGlobal(unsigned int numFrames, unsigned int numNonLinIterations, unsigned int numLinIterations, bool isStart, bool removeMaxResidual, bool isScanDone);

	void invalidateLastGlobalFrame();

	// update complete trajectory with new global trajectory info
	void updateTrajectory(unsigned int curFrame);

	const float4x4* getCompleteTrajectory() const { return d_completeTrajectory; }

	//debugging
	void saveGlobalSiftManagerAndCache(const std::string& prefix) const;
	void saveCompleteTrajectory(const std::string& filename, unsigned int numTransforms) const;
	void saveSiftTrajectory(const std::string& filename, unsigned int numTransforms) const;
	void printConvergence(const std::string& filename) const {
		m_SparseBundler.printConvergence(filename);
	}
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	void printSparseCorrEval() const {
		//std::cout << "========= LOCAL =========" << std::endl;
		//std::cout << "SIFT MATCH:" << std::endl;
		//std::cout << "\tprecision: (" << _siftMatch_frameFrameLocal.x << " / " << _siftMatch_frameFrameLocal.y << ") = " << ((float)_siftMatch_frameFrameLocal.x / (float)_siftMatch_frameFrameLocal.y) << std::endl;
		//std::cout << "\trecall:    (" << _siftMatch_frameFrameLocal.x << " / " << _gtFrameFrameTransformsLocal.size() << ") = " << ((float)_siftMatch_frameFrameLocal.x / (float)_gtFrameFrameTransformsLocal.size()) << std::endl;
		//std::cout << "SIFT VERIFY:" << std::endl;
		//std::cout << "\tprecision: (" << _siftVerify_frameFrameLocal.x << " / " << _siftVerify_frameFrameLocal.y << ") = " << ((float)_siftVerify_frameFrameLocal.x / (float)_siftVerify_frameFrameLocal.y) << std::endl;
		//std::cout << "\trecall:    (" << _siftVerify_frameFrameLocal.x << " / " << _gtFrameFrameTransformsLocal.size() << ") = " << ((float)_siftVerify_frameFrameLocal.x / (float)_gtFrameFrameTransformsLocal.size()) << std::endl;
		std::cout << "========= GLOBAL =========" << std::endl;
		std::cout << "SIFT RAW:" << std::endl;
		std::cout << "\tprecision: (" << _siftRaw_frameFrameGlobal.x << " / " << _siftRaw_frameFrameGlobal.y << ") = " << ((float)_siftRaw_frameFrameGlobal.x / (float)_siftRaw_frameFrameGlobal.y) << std::endl;
		std::cout << "\trecall:    (" << _siftRaw_frameFrameGlobal.x << " / " << _gtFrameFrameTransformsGlobal.size() << ") = " << ((float)_siftRaw_frameFrameGlobal.x / (float)_gtFrameFrameTransformsGlobal.size()) << std::endl;
		std::cout << "SIFT MATCH:" << std::endl;
		std::cout << "\tprecision: (" << _siftMatch_frameFrameGlobal.x << " / " << _siftMatch_frameFrameGlobal.y << ") = " << ((float)_siftMatch_frameFrameGlobal.x / (float)_siftMatch_frameFrameGlobal.y) << std::endl;
		std::cout << "\trecall:    (" << _siftMatch_frameFrameGlobal.x << " / " << _gtFrameFrameTransformsGlobal.size() << ") = " << ((float)_siftMatch_frameFrameGlobal.x / (float)_gtFrameFrameTransformsGlobal.size()) << std::endl;
		std::cout << "SIFT VERIFY:" << std::endl;
		std::cout << "\tprecision: (" << _siftVerify_frameFrameGlobal.x << " / " << _siftVerify_frameFrameGlobal.y << ") = " << ((float)_siftVerify_frameFrameGlobal.x / (float)_siftVerify_frameFrameGlobal.y) << std::endl;
		std::cout << "\trecall:    (" << _siftVerify_frameFrameGlobal.x << " / " << _gtFrameFrameTransformsGlobal.size() << ") = " << ((float)_siftVerify_frameFrameGlobal.x / (float)_gtFrameFrameTransformsGlobal.size()) << std::endl;
		std::cout << "OPT VERIFY:" << std::endl;
		std::cout << "\tprecision: (" << _opt_frameFrameGlobal.x << " / " << _opt_frameFrameGlobal.y << ") = " << ((float)_opt_frameFrameGlobal.x / (float)_opt_frameFrameGlobal.y) << std::endl;
		std::cout << "\trecall:    (" << _opt_frameFrameGlobal.x << " / " << _gtFrameFrameTransformsGlobal.size() << ") = " << ((float)_opt_frameFrameGlobal.x / (float)_gtFrameFrameTransformsGlobal.size()) << std::endl;
	}
#endif
#ifdef PRINT_MEM_STATS
	void printMemStats() const {
		std::cout << "#sift keys = " << m_global->getTotalNumKeyPoints() << std::endl;
		std::cout << "#residuals = " << m_global->getNumGlobalCorrespondences() << std::endl;
	}
#endif

	//! only debug 
	const SIFTImageManager* getCurrentLocalDEBUG() const { return m_currentLocal; }
	const SIFTImageManager* getGlobalDEBUG() const { return m_global; }
#ifdef DEBUG_PRINT_MATCHING
	void setPrintMatchesDEBUG(bool b) { _debugPrintMatches = b; }
	void saveLogImImCorrsToFile(const std::string& prefix) const {
		const std::string corrsPrefix = prefix + "_im-im-corrs";
		{
			BinaryDataStreamFile s(corrsPrefix + ".bin", true);
			s << _logFoundCorrespondences.size();
			if (!_logFoundCorrespondences.empty()) s.writeData((const BYTE*)_logFoundCorrespondences.data(), sizeof(std::pair<vec2ui, mat4f>)*_logFoundCorrespondences.size());
			s.closeStream();

			//human-readable version, just print the image indices
			std::ofstream os(corrsPrefix + ".txt");
			os << "# logged im-im correspondences = " << _logFoundCorrespondences.size() << std::endl;
			for (unsigned int i = 0; i < _logFoundCorrespondences.size(); i++)
				os << _logFoundCorrespondences[i].first << std::endl;
			os.close();
		}
		m_SparseBundler.saveLogRemovedCorrToFile(prefix + "_rm-im-im-corrs");
	}
#endif
	// to fake opt finish when no opt
	void resetDEBUG(bool initNextGlobal, int numLocalSolved, unsigned int curFrame) { //numLocalSolved == numGlobalFrames
		mutex_nextLocal.lock();
		if (initNextGlobal) {
			if (numLocalSolved >= 0) {
				float4x4 relativeTransform;
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(&relativeTransform, getLocalTrajectoryGPU(numLocalSolved) + m_submapSize, sizeof(float4x4), cudaMemcpyDeviceToHost));
				float4x4 prevTransform;
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(&prevTransform, d_globalTrajectory + numLocalSolved, sizeof(float4x4), cudaMemcpyDeviceToHost));
				float4x4 newTransform = prevTransform * relativeTransform;
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globalTrajectory + numLocalSolved + 1, &newTransform, sizeof(float4x4), cudaMemcpyHostToDevice));
			}
			if (numLocalSolved > 0) {
				// update trajectory
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_imageInvalidateList, m_invalidImagesList.data(), sizeof(int)*curFrame, cudaMemcpyHostToDevice));

				updateTrajectoryCU(d_globalTrajectory, numLocalSolved,
					d_completeTrajectory, curFrame,
					d_localTrajectories, m_submapSize + 1, numLocalSolved,
					d_imageInvalidateList);
			}
		}
		finishLocalOpt();
		mutex_nextLocal.unlock();
	}
	//TODO fix this hack
	void setEndSolveGlobalDenseWeights();

private:

	//! sift matching
	unsigned int matchAndFilter(bool isLocal, SIFTImageManager* siftManager, CUDACache* cudaCache, const float4x4& siftIntrinsicsInv); //!!!TODO FIX TIMING LOG

	void initSIFT(unsigned int widthSift, unsigned int heightSift);
	//! called when global locked
	void initializeNextGlobalTransform(unsigned int initGlobalIdx, unsigned int lastValidLocal) {
		const unsigned int numGlobalFrames = m_global->getNumImages();
		MLIB_ASSERT(numGlobalFrames >= 1);
		if (initGlobalIdx == (unsigned int)-1) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globalTrajectory + numGlobalFrames, d_globalTrajectory + numGlobalFrames - 1, sizeof(float4x4), cudaMemcpyDeviceToDevice));
		}
		else {
			initNextGlobalTransformCU(d_globalTrajectory, numGlobalFrames, initGlobalIdx, d_localTrajectories, m_submapSize + 1);
		}
	}

	//! called when nextlocal locked
	void finishLocalOpt() {
		m_nextLocal->reset();
		m_nextLocalCache->reset();
	}
	//! assumes nextlocal locked
	void saveOptToPointCloud(const std::string& filename, const CUDACache* cudaCache, const std::vector<int>& valid, const float4x4* d_transforms, unsigned int numFrames, bool saveFrameByFrame = false);
	void saveImPairToPointCloud(const std::string& prefix, const CUDACache* cudaCache, const float4x4* d_transforms, const vec2ui& imageIndices, const mat4f& transformCurToPrv = mat4f::zero()) const;

	//*********** SIFT *******************
	SiftGPU*				m_sift;
	SiftMatchGPU*			m_siftMatcher;
	//************ SUBMAPS ********************
	SBA						m_SparseBundler;

	std::mutex mutex_nextLocal;
	std::mutex m_mutexMatcher;

	CUDACache*			m_currentLocalCache;
	SIFTImageManager*	m_currentLocal;

	CUDACache*			m_nextLocalCache;
	SIFTImageManager*	m_nextLocal;

	CUDACache*			m_globalCache;
	SIFTImageManager*	m_global;

	//!!!TODO HERE
	CUDACache*			m_optLocalCache;
	SIFTImageManager*	m_optLocal;
	//*********** TRAJECTORIES ************
	float4x4* d_globalTrajectory;
	float4x4* d_completeTrajectory;
	float4x4* d_localTrajectories;
	std::vector<std::vector<int>> m_localTrajectoriesValid;

	float4x4*	 d_siftTrajectory; // frame-to-frame sift tracking for all frames in sequence
	//************************************
	unsigned int m_lastValidLocal, m_prevLastValidLocal; //TODO THIS SHOULD NOT LIVE HERE

	std::vector<unsigned int>	m_invalidImagesList;
	int*						d_imageInvalidateList; // tmp for updateTrajectory //TODO just to update trajectory on CPU

	float4x4*					d_currIntegrateTransform;
	std::vector<mat4f>			m_currIntegrateTransform;

	unsigned int m_numTotalFrames;
	unsigned int m_submapSize;

	int m_continueRetry;
	unsigned int m_revalidatedIdx;
#ifdef DEBUG_PRINT_MATCHING
	bool _debugPrintMatches;
	std::vector<std::pair<vec2ui, mat4f>> _logFoundCorrespondences; // logging the initially found (no invalidation) image-image correspondences; for global; (frame, curframe) transform from cur to frame
#endif
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	//vec2ui _siftMatch_frameFrameLocal; //#correct, #total got
	//vec2ui _siftVerify_frameFrameLocal; //#correct, #total got
	//std::unordered_map<vec2ui, mat4f> _gtFrameFrameTransformsLocal; //from x -> y
	vec2ui _siftRaw_frameFrameGlobal; //#correct, #total got
	vec2ui _siftMatch_frameFrameGlobal; //#correct, #total got
	vec2ui _siftVerify_frameFrameGlobal; //#correct, #total got
	vec2ui _opt_frameFrameGlobal; //#correct, #total got
	std::unordered_map<vec2ui, mat4f> _gtFrameFrameTransformsGlobal;
#endif
};

#endif