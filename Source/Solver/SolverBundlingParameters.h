#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
	unsigned int nNonLinearIterations;		// Steps of the non-linear solver	
	unsigned int nLinIterations;			// Steps of the linear solver

	float verifyOptDistThresh; // for verifying local 
	float verifyOptPercentThresh;

	// dense depth corr
	float denseDepthDistThresh;
	float denseDepthNormalThresh;
	float denseDepthColorThresh;
	float denseDepthMin;
	float denseDepthMax;

	bool useDenseDepthAllPairwise; // instead of frame-to-frame

	float weightSparse;
	float weightDenseDepth;
	float weightDenseDepthInit;
	float weightDenseDepthLinFactor;
};

#endif
