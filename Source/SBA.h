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
	SBA() {
		d_xRot = NULL;
		d_xTrans = NULL;
		m_solver = NULL;

		m_bUseComprehensiveFrameInvalidation = false;

		m_weightSparse = 1.0f;
		m_weightDense = 1.0f;
	}
	void init(unsigned int maxImages, unsigned int maxNumCorrPerImage) {
		unsigned int maxNumImages = maxImages;
		cutilSafeCall(cudaMalloc(&d_xRot, sizeof(EntryJ)*maxNumImages));
		cutilSafeCall(cudaMalloc(&d_xTrans, sizeof(EntryJ)*maxNumImages));

		m_solver = new CUDASolverBundling(maxImages, maxNumCorrPerImage);
		m_bVerify = false;

		m_bUseComprehensiveFrameInvalidation = GlobalBundlingState::get().s_useComprehensiveFrameInvalidation;
	}
	~SBA() {
		SAFE_DELETE(m_solver);

		if (d_xRot) cutilSafeCall(cudaFree(d_xRot));
		if (d_xTrans) cutilSafeCall(cudaFree(d_xTrans));
	}

	void align(SIFTImageManager* siftManager, CUDACache* cudaCache, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

	void evaluateSolverTimings() {
		m_solver->evaluateTimings();
	}
	void printConvergence(const std::string& filename) const;

	void setWeights(float ws, float wd) {
		m_weightSparse = ws;
		m_weightDense = wd;
	}

private:

	bool alignCUDA(SIFTImageManager* siftManager, CUDACache* cudaCache, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd);

	bool removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages);
	
	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;

	float			m_weightSparse;
	float			m_weightDense;

	CUDASolverBundling* m_solver;

	bool m_bUseComprehensiveFrameInvalidation;

	//for gpu solver
	float m_maxResidual; //!!!todo why is this here...
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;

	static Timer s_timer;
};

#endif