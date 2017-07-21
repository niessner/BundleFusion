#pragma once

#include "SBA.h"
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
#include "CorrespondenceEvaluator.h"
#endif

class SiftGPU;
class SiftMatchGPU;
class SIFTImageManager;
class CUDACache;
class CUDAImageManager;


#define USE_RETRY //TODO MOVE OUT TO PARAMS??

class Bundler
{
public:
	Bundler(unsigned int maxNumImages, unsigned int maxNumKeysPerImage,
		const mat4f& siftIntrinsicsInv, const CUDAImageManager* manager, bool isLocal);
	~Bundler();

	const float4x4* getTrajectoryGPU() const { return d_trajectory; }
	float4x4* getTrajectoryGPU() { return d_trajectory; }
	const std::vector<int>& getValidImages() const;
	void getCacheIntrinsics(float4x4& intrinsics, float4x4& intrinsicsInv);
	unsigned int getCurrFrameNumber() const { return m_siftManager->getCurrentFrame(); }
	unsigned int getNumFrames() const { return m_siftManager->getNumImages(); }

	//whether has >1 valid frame
	bool isValid() const;

	void reset();

	//TODO should this live outside?
	void detectFeatures(float* d_intensitySift, const float* d_inputDepthFilt);

	void storeCachedFrame(unsigned int depthWidth, unsigned int depthHeight, const uchar4* d_inputColor,
		unsigned int colorWidth, unsigned int colorHeight, const float* d_inputDepthRaw);
	void copyFrame(const Bundler* b, unsigned int frame);
	void addInvalidFrame();
	void invalidateLastFrame();

	const float4x4* getCurrentSiftTransformsGPU() const { return m_siftManager->getFiltTransformsToWorldGPU(); }
	const int* getNumFiltMatchesGPU() const { return m_siftManager->getNumFiltMatchesGPU(); }

	unsigned int matchAndFilter();
	bool optimize(unsigned int numNonLinIterations, unsigned int numLinIterations, bool bUseVerify, bool bRemoveMaxResidual, bool bIsScanDone, bool& bOptRemoved);
	void setSolveWeights(const std::vector<float>& sparse, const std::vector<float>& densedepth, const std::vector<float>& densecolor) {
		m_optimizer.setGlobalWeights(sparse, densedepth, densecolor, densedepth.back() > 0 || densecolor.back() > 0);
		std::cout << "set end solve global dense weights" << std::endl;
	}
	void fuseToGlobal(Bundler* glob);

	unsigned int tryRevalidation(unsigned int curGlobalFrame, bool bIsScanDone);
	unsigned int getRevalidatedIdx() const { return m_revalidatedIdx; }


	// -- various logging
#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	void initializeCorrespondenceEvaluator(const std::vector<mat4f>& trajectory, const std::string& logFilePrefix) {
		SAFE_DELETE(m_corrEvaluator);
		m_corrEvaluator = new CorrespondenceEvaluator(trajectory, logFilePrefix);
		std::cout << "[CorrespondenceEvaluator] " << logFilePrefix << std::endl;
	}
	void finishCorrespondenceEvaluatorLogging() { if (m_corrEvaluator) m_corrEvaluator->finishLoggingToFile(); }
#endif
	void saveSparseCorrsToFile(const std::string& filename) const;
	//TODO logging for residual information

private:
	void initSift(unsigned int widthSift, unsigned int heightSift, bool isLocal);

	void initializeNextTransformUnknown() {
		const unsigned int numFrames = m_siftManager->getNumImages();
		MLIB_ASSERT(numFrames >= 1);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_trajectory + numFrames, d_trajectory + numFrames - 1, sizeof(float4x4), cudaMemcpyDeviceToDevice));
	}

	//*********** SIFT *******************
	SiftGPU*				m_sift;
	SiftMatchGPU*			m_siftMatcher;
	float4x4				m_siftIntrinsics;
	float4x4				m_siftIntrinsicsInv;

	int							m_continueRetry;
	unsigned int				m_revalidatedIdx;

	//*********** OPTIMIZATION *******************
	SIFTImageManager*		m_siftManager;
	CUDACache*				m_cudaCache;
	SBA						m_optimizer;

	//*********** TRAJECTORIES *******************
	float4x4*				d_trajectory;

	bool					m_bIsLocal;
	Timer					m_timer;

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
	CorrespondenceEvaluator* m_corrEvaluator;
#endif
};


