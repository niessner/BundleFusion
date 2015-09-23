#pragma once


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

	void align(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

	void evaluateSolverTimings() {
		m_solver->evaluateTimings();
	}
	void printConvergence(const std::string& filename);

private:

	bool alignCUDA(SIFTImageManager* siftManager, float4x4* d_transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd);

	bool removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages);
	
	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;

	CUDASolverBundling* m_solver;

	bool m_bUseComprehensiveFrameInvalidation;

	//for gpu solver
	float m_maxResidual; //!!!todo why is this here...
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;

	static Timer s_timer;
};

