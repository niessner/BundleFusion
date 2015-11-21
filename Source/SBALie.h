#pragma once

#include "SBA_param.h"

#ifdef USE_LIE_SPACE
//WARNING broken: needs residual unpacking


#include "GlobalAppState.h"
#include "PoseHelper.h"
#include "SIFTImageManager.h"
#include "mLibEigen.h"

struct JacobianBlock {
	vec3f data[6];
};

class LieEnergyTermHelper {
public:
	void init(const EntryJ* d_correspondences, unsigned int numCorr, unsigned int numFrames)
	{
		MLIB_ASSERT(numCorr > 0 && numFrames > 1);

		std::cout << "SBA: using lie" << std::endl;
		throw MLIB_EXCEPTION("UNPACK RESIDUALS - OTHERWISE DOESN'T WORK");

		m_numVars = numFrames;
		m_numResiduals = numCorr * 3;

		m_correspondences.resize(numCorr);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_correspondences.data(), d_correspondences, sizeof(EntryJ)*numCorr, cudaMemcpyDeviceToHost));

		m_varToCorr.resize(numFrames);
		for (unsigned int i = 0; i < m_correspondences.size(); i++) {
			const EntryJ& corr = m_correspondences[i];
			if (corr.isValid()) {
				m_varToCorr[corr.imgIdx_i].push_back(i);
				m_varToCorr[corr.imgIdx_j].push_back(i);
			}
		}
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

	//void printResiduals(const std::string& filename) {
	//	std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
	//		return left.second > right.second;
	//	});
	//	std::ofstream s(filename);
	//	s << m_residuals.size() << " residuals, not squared" << std::endl;
	//	for (unsigned int i = 0; i < m_residuals.size(); i++) {
	//		const vec2ui& id = resIdxToMatchIdx[m_residuals[i].first / 3]; // (im-im id, match id)
	//		const vec2i& imageIndices = m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices();
	//		s << m_residuals[i].second << "(" << imageIndices << ")" << std::endl;
	//	}
	//	s.close();
	//}
	bool useVerification();
	unsigned int getNumFrames() const { return m_numFrames; }

	const std::vector< std::pair<unsigned int, float> >& getResiduals() const { return m_residuals; }

	bool findAndRemoveOutliers();

private:

	//! part of jacobian row for res corresponding to wpos i
	void computeJacobianBlockRow_i(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_i, const mat4f& transform_i, JacobianBlock& jacPart_i);
	//! part of jacobian row for res corresponding to wpos j							 
	void computeJacobianBlockRow_j(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_j, const mat4f& transform_j, JacobianBlock& jacPart_j);

	void addToLocalSystem(const JacobianBlock& jacobianBlockRow_i, const JacobianBlock& jacobianBlockRow_j, int varIdxBase_i, int varIdxBase_j, const vec3f& residual);

	float m_lastEnergy;
	unsigned int m_numVars;
	unsigned int m_numResiduals;
	unsigned int m_numFrames;

	Eigen::MatrixXf m_JTJ;
	Eigen::VectorXf m_JTr;

	std::vector<EntryJ> m_correspondences;
	std::vector< std::vector<unsigned int> > m_varToCorr;

	std::vector< std::pair<unsigned int, float> > m_residuals; // not squared
};


class SBA
{
public:
	SBA() {
		m_bVerify = false;
	}
	~SBA() {
		SAFE_DELETE(m_energyTermHelper);
	}

	void init() {
		m_energyTermHelper = new LieEnergyTermHelper;
		setEnergyHelperGlobal(m_energyTermHelper);
	}

	static void setEnergyHelperGlobal(LieEnergyTermHelper* es) {
		s_energyTermHelperGlobal = es;
	}

	static LieEnergyTermHelper* getEnergyTermHelper() {
		return s_energyTermHelperGlobal;
	}

	void align(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useVerify);

	float getMaxResidual() const { return m_maxResidual; }
	bool useVerification() const { return m_bVerify; }

private:

	void alignInternal(std::vector<mat4f>& transforms, unsigned int maxNumIters);

	static LieEnergyTermHelper* s_energyTermHelperGlobal;
	LieEnergyTermHelper*		m_energyTermHelper;

	bool m_bVerify;
	float m_maxResidual;
};


#endif