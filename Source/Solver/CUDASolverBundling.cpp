
#include "stdafx.h"
#include "CUDASolverBundling.h"
#include "../GlobalBundlingState.h"

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);
extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer);
extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, float* convergenceAnalysis, CUDATimer* timer);

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);

CUDASolverBundling::CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxCorrPerImage) : m_maxNumberOfImages(maxNumberOfImages), m_maxCorrPerImage(maxCorrPerImage)
, THREADS_PER_BLOCK(512) // keep consistent with the GPU
{
	m_timer = NULL;
	if (GlobalBundlingState::get().s_enableDetailedTimings) m_timer = new CUDATimer();
	m_bRecordConvergence = GlobalBundlingState::get().s_recordSolverConvergence;

	//!!!TODO PARAMS
	m_verifyOptDistThresh = 0.02f;//GlobalAppState::get().s_verifyOptDistThresh;
	m_verifyOptPercentThresh = 0.05f;//GlobalAppState::get().s_verifyOptPercentThresh;

	const unsigned int numberOfVariables = maxNumberOfImages;
	const unsigned int numberOfResiduums = maxNumberOfImages*maxCorrPerImage;

	// State
	cutilSafeCall(cudaMalloc(&m_solverState.d_deltaRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_deltaTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_zRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_zTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_pRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_pTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Jp, sizeof(float3)*numberOfResiduums));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_XRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_Ap_XTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float) * 2));
	//cutilSafeCall(cudaMalloc(&m_solverState.d_scanBeta, sizeof(float)));
	cutilSafeCall(cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float) *numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondionerRot, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_precondionerTrans, sizeof(float3)*numberOfVariables));
	cutilSafeCall(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));
	unsigned int n = (numberOfResiduums*3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cutilSafeCall(cudaMalloc(&m_solverState.d_maxResidual, sizeof(float) * n));
	cutilSafeCall(cudaMalloc(&m_solverState.d_maxResidualIndex, sizeof(int) * n));
	m_solverState.h_maxResidual = new float[n];
	m_solverState.h_maxResidualIndex = new int[n];

	cutilSafeCall(cudaMalloc(&d_variablesToCorrespondences, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage));
	cutilSafeCall(cudaMalloc(&d_numEntriesPerRow, sizeof(int)*m_maxNumberOfImages));

	cutilSafeCall(cudaMalloc(&m_solverState.d_countHighResidual, sizeof(int)));
}

CUDASolverBundling::~CUDASolverBundling()
{
	if (m_timer) delete m_timer;

	// State
	cutilSafeCall(cudaFree(m_solverState.d_deltaRot));
	cutilSafeCall(cudaFree(m_solverState.d_deltaTrans));
	cutilSafeCall(cudaFree(m_solverState.d_rRot));
	cutilSafeCall(cudaFree(m_solverState.d_rTrans));
	cutilSafeCall(cudaFree(m_solverState.d_zRot));
	cutilSafeCall(cudaFree(m_solverState.d_zTrans));
	cutilSafeCall(cudaFree(m_solverState.d_pRot));
	cutilSafeCall(cudaFree(m_solverState.d_pTrans));
	cutilSafeCall(cudaFree(m_solverState.d_Jp));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_XRot));
	cutilSafeCall(cudaFree(m_solverState.d_Ap_XTrans));
	cutilSafeCall(cudaFree(m_solverState.d_scanAlpha));
	//cutilSafeCall(cudaFree(m_solverState.d_scanBeta));
	cutilSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerRot));
	cutilSafeCall(cudaFree(m_solverState.d_precondionerTrans));
	cutilSafeCall(cudaFree(m_solverState.d_sumResidual));
	cutilSafeCall(cudaFree(m_solverState.d_maxResidual));
	cutilSafeCall(cudaFree(m_solverState.d_maxResidualIndex));
	SAFE_DELETE_ARRAY(m_solverState.h_maxResidual);
	SAFE_DELETE_ARRAY(m_solverState.h_maxResidualIndex);

	cutilSafeCall(cudaFree(d_variablesToCorrespondences));
	cutilSafeCall(cudaFree(d_numEntriesPerRow));

	cutilSafeCall(cudaFree(m_solverState.d_countHighResidual));
}

void CUDASolverBundling::solve(EntryJ* d_targetCorrespondences, unsigned int numberOfCorrespondences, unsigned int numberOfImages, unsigned int nNonLinearIterations, unsigned int nLinearIterations, float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns)
{
	float* convergence = NULL;
	if (m_bRecordConvergence) {
		m_convergence.resize(nNonLinearIterations + 1, -1.0f);
		convergence = m_convergence.data();
	}

	m_solverState.d_xRot = d_rotationAnglesUnknowns;
	m_solverState.d_xTrans = d_translationUnknowns;

	SolverParameters parameters;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	SolverInput solverInput;
	solverInput.d_correspondences = d_targetCorrespondences;
	solverInput.d_variablesToCorrespondences = d_variablesToCorrespondences;
	solverInput.d_numEntriesPerRow = d_numEntriesPerRow;
	solverInput.numberOfImages = numberOfImages;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	buildVariablesToCorrespondencesTable(d_targetCorrespondences, numberOfCorrespondences);
	solveBundlingStub(solverInput, m_solverState, parameters, convergence, m_timer);

	computeMaxResidual(solverInput, parameters);
}

void CUDASolverBundling::buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	cutilSafeCall(cudaMemset(d_numEntriesPerRow, 0, sizeof(int)*m_maxNumberOfImages));

	buildVariablesToCorrespondencesTableCUDA(d_correspondences, numberOfCorrespondences, m_maxCorrPerImage, d_variablesToCorrespondences, d_numEntriesPerRow, m_timer);
}

void CUDASolverBundling::computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters)
{
	evalMaxResidual(solverInput, m_solverState, parameters, m_timer);
	// copy to cpu
	unsigned int n = (solverInput.numberOfCorrespondences*3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	cutilSafeCall(cudaMemcpy(m_solverState.h_maxResidual, m_solverState.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(m_solverState.h_maxResidualIndex, m_solverState.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));
	// compute max
	float maxResidual = 0.0f; int maxResidualIndex = 0;
	for (unsigned int i = 0; i < n; i++) {
		if (maxResidual < m_solverState.h_maxResidual[i]) {
			maxResidual = m_solverState.h_maxResidual[i];
			maxResidualIndex = m_solverState.h_maxResidualIndex[i];
		}
	}
	m_solverState.h_maxResidual[0] = maxResidual;
	m_solverState.h_maxResidualIndex[0] = maxResidualIndex;
}

bool CUDASolverBundling::getMaxResidual(EntryJ* d_correspondences, ml::vec2ui& imageIndices, float& maxRes)
{
	const float MAX_RESIDUAL = 0.05f; // nonsquared residual
	if (m_timer) m_timer->startEvent(__FUNCTION__);

	maxRes = m_solverState.h_maxResidual[0];

	// for debugging get image indices regardless
	EntryJ h_corr;
	unsigned int imIdx = m_solverState.h_maxResidualIndex[0] / 3;
	cutilSafeCall(cudaMemcpy(&h_corr, d_correspondences + imIdx, sizeof(EntryJ), cudaMemcpyDeviceToHost));
	imageIndices = ml::vec2ui(h_corr.imgIdx_i, h_corr.imgIdx_j);

	if (m_timer) m_timer->endEvent();

	if (m_solverState.h_maxResidual[0] > MAX_RESIDUAL) { // remove!
		
		return true;
	}

	return false;
}

bool CUDASolverBundling::useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	SolverParameters parameters;
	parameters.nNonLinearIterations = 0;
	parameters.nLinIterations = 0;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = NULL;
	solverInput.d_numEntriesPerRow = NULL;
	solverInput.numberOfImages = 0;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	unsigned int numHighResiduals = countHighResiduals(solverInput, m_solverState, parameters, m_timer);
	unsigned int total = solverInput.numberOfCorrespondences * 3;
	//std::cout << "\t[ useVerification ] " << numHighResiduals << " / " << total << " = " << (float)numHighResiduals / total << " vs " << parameters.verifyOptPercentThresh << std::endl;
	if ((float)numHighResiduals / total >= parameters.verifyOptPercentThresh) return true;
	return false;
}
