#pragma once

#include "SBA_param.h"

#ifdef USE_GPU_SOLVE
#include "SiftGPU/cudaUtil.h"
#include "SiftGPU/SIFTImageManager.h"
#include "PoseHelper.h"

#include "Solver/CUDASolverBundling.h"
#include "GlobalBundlingState.h"



struct JacobianBlock {
	ml::vec3f data[6];
};

class SBA
{
public:
	SBA();
	void init(unsigned int maxImages, unsigned int maxNumResiduals) {
		unsigned int maxNumImages = maxImages;
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xRot, sizeof(EntryJ)*maxNumImages));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xTrans, sizeof(EntryJ)*maxNumImages));

		m_solver = new CUDASolverBundling(maxImages, maxNumResiduals);
		m_bVerify = false;

		m_bUseComprehensiveFrameInvalidation = GlobalBundlingState::get().s_useComprehensiveFrameInvalidation;
		m_bUseLocalDense = GlobalBundlingState::get().s_useLocalDense;
	}
	~SBA() {
		SAFE_DELETE(m_solver);

		MLIB_CUDA_SAFE_FREE(d_xRot);
		MLIB_CUDA_SAFE_FREE(d_xTrans);
	}

	void align(SIFTImageManager* siftManager, const CUDACache* cudaCache, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

	void evaluateSolverTimings() {
		m_solver->evaluateTimings();
	}
	void printConvergence(const std::string& filename) const;

	//void setLocalWeights(const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor) {
	//	m_localWeightsSparse = weightsSparse;
	//	m_localWeightsDenseDepth = weightsDenseDepth;
	//	m_localWeightsDenseColor = weightsDenseColor;
	//}
	void setGlobalWeights(const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool useGlobalDenseOpt) {
		m_globalWeightsMutex.lock();
		m_globalWeightsSparse = weightsSparse;
		m_globalWeightsDenseDepth = weightsDenseDepth;
		m_globalWeightsDenseColor = weightsDenseColor;
		m_bUseGlobalDenseOpt = useGlobalDenseOpt;
		m_globalWeightsMutex.unlock();
	}

private:
	void evalResidualDEBUG(SIFTImageManager* siftManager, const float4x4* d_transforms) const {
		std::vector<EntryJ> corrs(siftManager->getNumGlobalCorrespondences());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(corrs.data(), siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*corrs.size(), cudaMemcpyDeviceToHost));
		std::vector<mat4f> transforms(siftManager->getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(transforms.data(), d_transforms, sizeof(float4x4)*transforms.size(), cudaMemcpyDeviceToHost));

		float sumResidual2 = 0.0f;
		for (unsigned int i = 0; i < corrs.size(); i++) {
			const EntryJ& corr = corrs[i];
			if (corr.isValid()) {
				vec3f r = transforms[corr.imgIdx_i] * vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z) - transforms[corr.imgIdx_j] * vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
				sumResidual2 += (r | r);
			}
		}
		std::cout << "sum residual2 = " << sumResidual2 << std::endl;
	}

	bool alignCUDA(SIFTImageManager* siftManager, const CUDACache* cudaCache, bool useDensePairwise,
		const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor,
		unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd);

	bool removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages);
	
	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;

	//dense opt params
	bool m_bUseLocalDense;
	bool m_bUseGlobalDenseOpt;
	std::vector<float> m_localWeightsSparse;
	std::vector<float> m_localWeightsDenseDepth;
	std::vector<float> m_localWeightsDenseColor;
	std::vector<float> m_globalWeightsSparse;
	std::vector<float> m_globalWeightsDenseDepth;
	std::vector<float> m_globalWeightsDenseColor;
	std::mutex m_globalWeightsMutex;

	CUDASolverBundling* m_solver;

	bool m_bUseComprehensiveFrameInvalidation;

	//for gpu solver
	float m_maxResidual; //!!!todo why is this here...
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;

	static Timer s_timer;
};

#endif