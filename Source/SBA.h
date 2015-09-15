#pragma once


#include "SiftGPU/cudaUtil.h"
#include "SiftGPU/SIFTImageManager.h"
#include "PoseHelper.h"

#include "Solver/CUDASolverBundling.h"
//#include "cuda_SimpleVectorUtil.h"

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
	}
	//!!!TODO PARAMS
	void init(unsigned int maxImages, unsigned int maxNumCorrPerImage) {
		unsigned int maxNumImages = maxImages;//GlobalAppState::get().s_maxNumImages;
		cutilSafeCall(cudaMalloc(&d_xRot, sizeof(EntryJ)*maxNumImages));
		cutilSafeCall(cudaMalloc(&d_xTrans, sizeof(EntryJ)*maxNumImages));

		m_solver = new CUDASolverBundling(maxImages, maxNumCorrPerImage);//GlobalAppState::get().s_maxNumImages, GlobalAppState::get().s_maxNumCorrPerImage);
		m_bVerify = false;
	}
	~SBA() {
		SAFE_DELETE(m_solver);

		if (d_xRot) cutilSafeCall(cudaFree(d_xRot));
		if (d_xTrans) cutilSafeCall(cudaFree(d_xTrans));
	}

	void align(SIFTImageManager* siftManager, std::vector<ml::mat4f>& transforms, unsigned int maxNumIters, unsigned int numPCGits, bool useVerify);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

private:

	bool alignCUDA(SIFTImageManager* siftManager, std::vector<ml::mat4f>& transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool useVerify);

	bool removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages);
	
	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;

	CUDASolverBundling* m_solver;

	//for gpu solver
	float m_maxResidual; //!!!todo why is this here...
	bool m_bVerify;

};

