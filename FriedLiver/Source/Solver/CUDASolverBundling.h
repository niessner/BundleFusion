#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "../SiftGPU/cudaUtil.h"
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"

#include "../SiftGPU/cuda_SimpleMatrixUtil.h"
#include "../SiftGPU/CUDATimer.h"

#include <conio.h>

class CUDACache;
//#define NEW_GUIDED_REMOVE 


class CUDASolverBundling
{
public:

	CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxNumResiduals);
	~CUDASolverBundling();

	//weightSparse*Esparse + (#iters*weightDenseLinFactor + weightDense)*Edense
	void solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,
		const int* d_validImages, unsigned int numberOfImages,
		unsigned int nNonLinearIterations, unsigned int nLinearIterations, const CUDACache* cudaCache,
		const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool usePairwiseDense,
		float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns,
		bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx);
	const std::vector<float>& getConvergenceAnalysis() const { return m_convergence; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_linConvergence; }

	void getMaxResidual(float& max, int& index) const {
		max = m_solverExtra.h_maxResidual[0];
		index = m_solverExtra.h_maxResidualIndex[0];
	};
	bool getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, ml::vec2ui& imageIndices, float& maxRes);
	bool useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);

	const int* getVariablesToCorrespondences() const { return d_variablesToCorrespondences; }
	const int* getVarToCorrNumEntriesPerRow() const { return d_numEntriesPerRow; }

	void evaluateTimings() {
		if (m_timer) {
			//std::cout << "********* SOLVER TIMINGS *********" << std::endl;
			m_timer->evaluate(true);
			std::cout << std::endl << std::endl;
		}
	}

	void resetTimer() {
		if (m_timer) m_timer->reset();
	}

#ifdef NEW_GUIDED_REMOVE
	const std::vector<vec2ui>& getGuidedMaxResImagesToRemove() const { return m_maxResImPairs; }
#endif
private:

	//!helper
	static bool isSimilarImagePair(const vec2ui& pair0, const vec2ui& pair1) {
		if ((std::abs((int)pair0.x - (int)pair1.x) < 10 && std::abs((int)pair0.y - (int)pair1.y) < 10) ||
			(std::abs((int)pair0.x - (int)pair1.y) < 10 && std::abs((int)pair0.y - (int)pair1.x) < 10))
			return true;
		return false;
	}
	//static bool isLookingAtSame(const mat4f& trans_i, const mat4f& trans_j,
	//	const mat4f& intrinsics, const mat4f& intrinsicsInv, unsigned int width, unsigned int height)
	//{
	//	{//check if looking at opposites
	//		vec3f x(1.0f, 1.0f, 1.0f); x.normalize();
	//		vec3f v = (trans_i.getInverse()*trans_j).getMatrix3x3() * x;
	//		float angle = std::acos(math::clamp(x | v, -1.0f, 1.0f));
	//		if (fabs(angle) > 2.5f) return false; //~143 degrees
	//	}
	//	const float minDepth = 1.0f;
	//	const float maxDepth = 5.0f;
	//	vec2f imgMin(0.0f);
	//	vec2f imgMax((float)width, (float)height);
	//	std::vector<vec3f> corners(4);
	//	corners[0] = vec3f(imgMin.x, imgMin.y, 1.0f);		corners[1] = vec3f(imgMin.x, imgMax.y, 1.0f);
	//	corners[2] = vec3f(imgMax.x, imgMax.y, 1.0f);		corners[3] = vec3f(imgMax.x, imgMin.y, 1.0f);
	//	mat4f invTrans_i = trans_i.getInverse();
	//	mat4f invTrans_j = trans_j.getInverse();
	//	BoundingBox2f i_in_j, j_in_i;
	//	for (unsigned int c = 0; c < corners.size(); c++) {
	//		vec3f ij = intrinsics * (invTrans_j * trans_i * intrinsicsInv * (minDepth * corners[c]));
	//		vec3f ji = intrinsics * (invTrans_i * trans_j * intrinsicsInv * (minDepth * corners[c]));
	//		i_in_j.include(ij.getVec2() / ij.z);
	//		j_in_i.include(ji.getVec2() / ji.z);
	//		ij = intrinsics * (invTrans_j * trans_i * intrinsicsInv * (maxDepth * corners[c]));
	//		ji = intrinsics * (invTrans_i * trans_j * intrinsicsInv * (maxDepth * corners[c]));
	//		i_in_j.include(ij.getVec2() / ij.z);
	//		j_in_i.include(ji.getVec2() / ji.z);
	//	}
	//	BoundingBox2f bbImg(imgMin, imgMax);
	//	if ((bbImg.intersects(i_in_j) || bbImg.intersects(j_in_i)))
	//		return true;
	//	return false;
	//}

	void buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);
	void computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx);

	SolverState	m_solverState;
	SolverStateAnalysis m_solverExtra;
	const unsigned int THREADS_PER_BLOCK;

	unsigned int m_maxNumberOfImages;
	unsigned int m_maxCorrPerImage;

	unsigned int m_maxNumDenseImPairs;

	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	std::vector<float> m_convergence; // convergence analysis (energy per non-linear iteration)
	std::vector<float> m_linConvergence; // linear residual per linear iteration, concatenates for nonlinear its

	float m_verifyOptDistThresh;
	float m_verifyOptPercentThresh;

	bool		m_bRecordConvergence;
	CUDATimer *m_timer;

	SolverParameters m_defaultParams;
	float			 m_maxResidualThresh;

#ifdef NEW_GUIDED_REMOVE
	//for more than one im-pair removal
	std::vector<vec2ui> m_maxResImPairs;

	//!!!debugging
	float4x4*	d_transforms;
	//!!!debugging
#endif
};
