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

	//return if removed res
	bool align(SIFTImageManager* siftManager, const CUDACache* cudaCache, float4x4* d_transforms, unsigned int maxNumIters, unsigned int numPCGits,
		bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx = (unsigned int)-1);

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

	//!!!debugging
	void saveLogRemovedCorrToFile(const std::string& prefix) const {
		BinaryDataStreamFile s(prefix + ".bin", true);
		s << _logRemovedImImCorrs.size();
		if (!_logRemovedImImCorrs.empty()) s.writeData((const BYTE*)_logRemovedImImCorrs.data(), sizeof(std::pair<vec2ui, float>)*_logRemovedImImCorrs.size());
		s.close();

		// human readable version
		std::ofstream os(prefix + ".txt");
		os << "# remove im-im correspondences = " << _logRemovedImImCorrs.size() << std::endl;
		for (unsigned int i = 0; i < _logRemovedImImCorrs.size(); i++) 
			os << _logRemovedImImCorrs[i].first << "\t\t" << _logRemovedImImCorrs[i].second << std::endl;
		os.close();
	}
	//!!!debugging

private:

	bool alignCUDA(SIFTImageManager* siftManager, const CUDACache* cudaCache, bool useDensePairwise,
		const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor,
		unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool isStart, bool isEnd,
		unsigned int revalidateIdx);

	bool removeMaxResidualCUDA(SIFTImageManager* siftManager, unsigned int numImages, unsigned int curFrame);
	
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

	//record residual removal
	float m_maxResidual;
	//for gpu solver
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;

	static Timer s_timer;


	//!!!debugging
	std::vector<std::pair<vec2ui, float>> _logRemovedImImCorrs; 
	//!!!debugging
};

#endif