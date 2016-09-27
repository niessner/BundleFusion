#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 
#include "../SiftGPU/SIFTImageManager.h"
#include "../CUDACacheUtil.h"

struct SolverInput
{	
	EntryJ* d_correspondences;
	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	unsigned int numberOfCorrespondences;
	unsigned int numberOfImages;

	unsigned int maxNumberOfImages;
	unsigned int maxCorrPerImage;

	const int* d_validImages;
	const CUDACachedFrame* d_cacheFrames;
	unsigned int denseDepthWidth;
	unsigned int denseDepthHeight;
	float4 intrinsics;				//TODO constant buffer for this + siftimagemanger stuff?
	unsigned int maxNumDenseImPairs;
	float2 colorFocalLength; //color camera params (actually same as depthIntrinsics...)

	const float* weightsSparse;
	const float* weightsDenseDepth;
	const float* weightsDenseColor;
};

// State of the GN Solver
struct SolverState
{
	float3*	d_deltaRot;					// Current linear update to be computed
	float3*	d_deltaTrans;				// Current linear update to be computed
	
	float3* d_xRot;						// Current state
	float3* d_xTrans;					// Current state

	float3*	d_rRot;						// Residuum // jtf
	float3*	d_rTrans;					// Residuum // jtf
	
	float3*	d_zRot;						// Preconditioned residuum
	float3*	d_zTrans;					// Preconditioned residuum
	
	float3*	d_pRot;						// Decent direction
	float3*	d_pTrans;					// Decent direction
	
	float3*	d_Jp;						// Cache values after J

	float3*	d_Ap_XRot;					// Cache values for next kernel call after A = J^T x J x p
	float3*	d_Ap_XTrans;				// Cache values for next kernel call after A = J^T x J x p

	float*	d_scanAlpha;				// Tmp memory for alpha scan

	float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)
	
	float3*	d_precondionerRot;			// Preconditioner for linear system
	float3*	d_precondionerTrans;		// Preconditioner for linear system

	float*	d_sumResidual;				// sum of the squared residuals //debug

	//float* d_residuals; // debugging
	//float* d_sumLinResidual; // debugging // helper to compute linear residual

	int* d_countHighResidual;

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}

	// for dense depth term
	float* d_denseJtJ;
	float* d_denseJtr;
	float* d_denseCorrCounts;

	float4x4* d_xTransforms;
	float4x4* d_xTransformInverses;

	uint2* d_denseOverlappingImages;
	int* d_numDenseOverlappingImages;

	//!!!DEBUGGING
	int* d_corrCount;
	int* d_corrCountColor;
	float* d_sumResidualColor;
};

struct SolverStateAnalysis
{
	// residual pruning
	int*	d_maxResidualIndex;
	float*	d_maxResidual;

	int*	h_maxResidualIndex;
	float*	h_maxResidual;
};

#endif
