#pragma once

#include "SBA_param.h"
#ifdef USE_CPU_SOLVE

#include "cudaUtil.h"
//#include "alglib.h"
#include "GlobalAppState.h"
#include "PoseHelper.h"
#include "Derivatives.h"


struct JacobianBlock {
	vec3f data[6];
};

class SBAEnergyTermHelper {
public:
	void init(ImageBundle* bundle, unsigned int numFrames)
	{
#ifdef USE_LIE_SPACE
		std::cout << "SBA: using lie" << std::endl;
		throw MLIB_EXCEPTION("UNPACK RESIDUALS - OTHERWISE DOESN'T WORK");
#else
		std::cout << "SBA: using euler" << std::endl;
#endif
		m_bundle = bundle;
		m_numFrames = (numFrames == 0) ? bundle->getNumImages() : numFrames;
		MLIB_ASSERT(m_numFrames >= 2);

		m_numResiduals = 0;
		unsigned int idx = 0;
		const Matches& matches = m_bundle->getMatches();
		for (unsigned int t = 0; t < matches.size(); t++) {
			const ImagePairMatchSet& set = matches.getImagePairMatchSet(t);
			const vec2i& imageIndices = set.getImageIndices();
			if (imageIndices.x < (int)m_numFrames && imageIndices.y < (int)m_numFrames) {
				for (unsigned int m = 0; m < set.getNumMatches(); m++) {
					matchIdxToResIdx[vec2ui(t, m)] = idx++;
					resIdxToMatchIdx.push_back(vec2ui(t, m));
					m_numResiduals++;
				}
			} // check if is frame to optimize
		}
		m_numResiduals *= 3;
		m_lastEnergy = 0.0f;

		m_numVars = (m_numFrames - 1) * 6; // don't optimize for the first frame
		m_JTJ.resize(m_numVars, m_numVars);
		m_JTr.resize(m_numVars);

		m_residuals.resize(m_numResiduals);

#ifdef DEBUG_JAC_RES
		d_residuals.resize(m_numResiduals);
		d_jacobian.resize(m_numResiduals, m_numVars);
#endif
	}

	float computeEnergySum(const std::vector<mat4f>& transforms) {
		computeEnergy(transforms);
		return m_lastEnergy;
	}

	void computeEnergy(const std::vector<mat4f>& transforms);

	void computeEnergyAndJacobian(const std::vector<mat4f>& transforms, const std::vector<Pose>& poses);

	unsigned int getNumUnknowns() const { return m_numVars; }
	unsigned int getNumResiduals() const { return m_numResiduals; }

	float getLastEnergy() const { return m_lastEnergy; }

	const Eigen::MatrixXf& getJTJ() const { return m_JTJ; }
	const Eigen::VectorXf& getJTr() const { return m_JTr; }

	ImageBundle* getImageBundle() const { return m_bundle; }

	void printResiduals(const std::string& filename) {
		std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
			return left.second > right.second;
		});
		std::ofstream s(filename);
		s << m_residuals.size() << " residuals, not squared" << std::endl;
		for (unsigned int i = 0; i < m_residuals.size(); i++) {
			const vec2ui& id = resIdxToMatchIdx[m_residuals[i].first / 3]; // (im-im id, match id)
			const vec2i& imageIndices = m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices();
			s << m_residuals[i].second << "(" << imageIndices << ")" << std::endl;
		}
		s.close();
	}
	bool useVerification();
	unsigned int getNumFrames() const { return m_numFrames; }

	const std::vector< std::pair<unsigned int, float> >& getResiduals() const { return m_residuals; }

#ifdef DEBUG_JAC_RES
	const Eigen::MatrixXf& getJacobian() const { return d_jacobian; };
	const Eigen::VectorXf& getResidualVector() const { return d_residuals; }
	void computeNumericJacobian(const std::vector<Pose>& poses, float diffStep = 0.0f001) {
		const float diff2 = diffStep / 2.0f;

		// set to zero
		unsigned int numVars = (unsigned int)((poses.size() - 1) * 6);
		for (unsigned int i = 0; i < m_numResiduals; i++) {
			for (unsigned int j = 0; j < numVars; j++)
				d_jacobian(i, j) = 0.0f;
		}

		std::vector< std::pair<unsigned int, float> > resForward, resBackward;
		for (unsigned int p = 1; p < poses.size(); p++) { // for each variable
			for (unsigned int i = 0; i < 6; i++) {
				std::vector<Pose> stepPoses = poses;
				stepPoses[p][i] += diff2;
				std::vector<mat4f> stepMatrices = PoseHelper::convertToMatrices(stepPoses);
				computeEnergy(stepMatrices); resForward = m_residuals;

				stepPoses[p][i] = poses[p][i] - diff2;
				stepMatrices[p] = PoseHelper::PoseToMatrix(stepPoses[p]);
				computeEnergy(stepMatrices); resBackward = m_residuals;

				unsigned int varIdx = (p - 1) * 6 + i;
				for (unsigned int r = 0; r < m_numResiduals; r++)
					d_jacobian(r, varIdx) = (resForward[r].second - resBackward[r].second) / diffStep;
			}
		}
	}
#endif
	bool findAndRemoveOutliers();

private:

#ifdef USE_LIE_SPACE
	//! part of jacobian row for res corresponding to wpos i
	void computeJacobianBlockRow_i(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_i, const mat4f& transform_i, JacobianBlock& jacPart_i);
	//! part of jacobian row for res corresponding to wpos j							 
	void computeJacobianBlockRow_j(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_j, const mat4f& transform_j, JacobianBlock& jacPart_j);
#else
	//! part of jacobian row for res corresponding to wpos i
	void computeJacobianBlockRow_i(const vec3f& cpos_i, const vec3f& dist, MatrixDerivatives& deriv_i, JacobianBlock& jacPart_i);
	//! part of jacobian row for res corresponding to wpos j																		
	void computeJacobianBlockRow_j(const vec3f& cpos_j, const vec3f& dist, MatrixDerivatives& deriv_j, JacobianBlock& jacPart_j);

	void computeJTJStatistics(const std::string& filename) const;
#endif

	void addToLocalSystem(const JacobianBlock& jacobianBlockRow_i, const JacobianBlock& jacobianBlockRow_j, int varIdxBase_i, int varIdxBase_j, const vec3f& residual);

	ImageBundle* m_bundle; // edit-able for residual filtering

	float m_lastEnergy;
	unsigned int m_numVars;
	unsigned int m_numResiduals;
	unsigned int m_numFrames;

	Eigen::MatrixXf m_JTJ;
	Eigen::VectorXf m_JTr;

	//! residual debugging info
	std::vector<vec2ui> resIdxToMatchIdx;
	std::vector< std::pair<unsigned int, float> > m_residuals; // not squared
	std::unordered_map<vec2ui, unsigned int, std::hash<vec2ui>> matchIdxToResIdx;

#ifdef DEBUG_JAC_RES
	Eigen::MatrixXf d_jacobian;
	Eigen::VectorXf d_residuals;
#endif
};


class SBA
{
public:
	SBA() {
		unsigned int maxNumImages = GlobalAppState::get().s_maxNumImages;
		cutilSafeCall(cudaMalloc(&d_correspondences, sizeof(Correspondence)*maxNumImages*GlobalAppState::get().s_maxNumCorrPerImage));
		cutilSafeCall(cudaMalloc(&d_xRot, sizeof(Correspondence)*maxNumImages));
		cutilSafeCall(cudaMalloc(&d_xTrans, sizeof(Correspondence)*maxNumImages));

		m_solver = new CUDASolverBundling(GlobalAppState::get().s_maxNumImages, GlobalAppState::get().s_maxNumCorrPerImage);
		m_bVerify = false;
	}
	~SBA() {
		SAFE_DELETE(m_energyTermHelper);
		SAFE_DELETE(m_solver);

		if (d_correspondences) cutilSafeCall(cudaFree(d_correspondences));
		if (d_xRot) cutilSafeCall(cudaFree(d_xRot));
		if (d_xTrans) cutilSafeCall(cudaFree(d_xTrans));
	}

	// numFrames = 0 -> use all frames, otherwise only use frames < numFrames
	void init(ImageBundle* bundle, unsigned int numFrames = 0) {
		m_energyTermHelper = new SBAEnergyTermHelper;
		m_energyTermHelper->init(bundle, numFrames);
		setEnergyHelperGlobal(m_energyTermHelper);
	}

	static void setEnergyHelperGlobal(SBAEnergyTermHelper* es) {
		s_energyTermHelperGlobal = es;
	}

	static SBAEnergyTermHelper* getEnergyTermHelper() {
		return s_energyTermHelperGlobal;
	}

	void align(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useGPUSolve, unsigned int numPCGits, bool useVerify);


	void alignGradientDescent(std::vector<mat4f>& transforms, unsigned int maxNumIters);
#ifdef DEBUG_JAC_RES
	void alignNumeric(std::vector<mat4f>& transforms, unsigned int maxNumIters);
#endif

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

private:

	void alignInternal(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useCPUPCG);
	bool alignCUDA(std::vector<mat4f>& transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool useVerify);

	bool removeMaxResidualCUDA(Correspondence* d_correspondences);

	static SBAEnergyTermHelper* s_energyTermHelperGlobal;
	SBAEnergyTermHelper*		m_energyTermHelper;

	static Timer m_timer;

	Correspondence* d_correspondences;
	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;

	CUDASolverBundling* m_solver;

	//for gpu solver
	float m_maxResidual; //!!!todo why is this here...
	bool m_bVerify;

};

#endif