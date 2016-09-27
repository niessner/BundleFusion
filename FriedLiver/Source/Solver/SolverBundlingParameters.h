#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
	unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
	unsigned int nLinIterations;			// Steps of the linear solver

	float verifyOptDistThresh; // for verifying local 
	float verifyOptPercentThresh;

	float highResidualThresh;

	// dense depth corr
	float denseDistThresh;
	float denseNormalThresh;
	float denseColorThresh;
	float denseColorGradientMin;
	float denseDepthMin;
	float denseDepthMax;

	bool useDenseDepthAllPairwise; // instead of frame-to-frame
	unsigned int denseOverlapCheckSubsampleFactor;

	float weightSparse;		
	float weightDenseDepth;	
	float weightDenseColor;
	bool useDense;
};

#endif
