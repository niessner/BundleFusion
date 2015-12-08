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

class CUDASolverBundling
{
	public:

		CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxCorrPerImage);
		~CUDASolverBundling();

		void solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int numberOfImages, 
			unsigned int nNonLinearIterations, unsigned int nLinearIterations,
			CUDACache* cudaCache, float sparseWeight, float denseWeight,
			float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns,
			bool rebuildJT, bool findMaxResidual);
		const std::vector<float>& getConvergenceAnalysis() const { return m_convergence; }
		const std::vector<float>& getLinearConvergenceAnalysis() const { return m_linConvergence; }

		void getMaxResidual(float& max, int& index) const {
			max = m_solverState.h_maxResidual[0];
			index = m_solverState.h_maxResidualIndex[0];
		};
		bool getMaxResidual(EntryJ* d_correspondences, ml::vec2ui& imageIndices, float& maxRes);
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

	private:

		void buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);
		void computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters);

		SolverState	m_solverState;
		const unsigned int THREADS_PER_BLOCK;

		unsigned int m_maxNumberOfImages;
		unsigned int m_maxCorrPerImage;

		int* d_variablesToCorrespondences;
		int* d_numEntriesPerRow;

		std::vector<float> m_convergence; // convergence analysis (energy per non-linear iteration)
		std::vector<float> m_linConvergence; // linear residual per linear iteration, concatenates for nonlinear its

		float m_verifyOptDistThresh;
		float m_verifyOptPercentThresh;

		bool		m_bRecordConvergence;
		CUDATimer *m_timer;
};
