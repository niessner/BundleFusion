#include <iostream>

#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "SolverBundlingUtil.h"
#include "SolverBundlingEquations.h"
#include "../../SiftGPU/CUDATimer.h"

#include <conio.h>

/////////////////////////////////////////////////////////////////////////
// Eval Max Residual
/////////////////////////////////////////////////////////////////////////

__global__ void EvalMaxResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	__shared__ int maxResIndex[THREADS_PER_BLOCK];
	__shared__ float maxRes[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfCorrespondences * 3; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	maxResIndex[threadIdx.x] = 0;
	maxRes[threadIdx.x] = 0.0f;

	if (x < N) {
		const unsigned int corrIdx = x / 3;
		const unsigned int componentIdx = x - corrIdx * 3;
		float residual = evalResidualDeviceFloat3(corrIdx, componentIdx, input, state, parameters);

		maxRes[threadIdx.x] = residual;
		maxResIndex[threadIdx.x] = x;

		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {

			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				if (maxRes[first] < maxRes[second]) {
					maxRes[first] = maxRes[second];
					maxResIndex[first] = maxResIndex[second];
				}
			}

			__syncthreads();
		}

		if (threadIdx.x == 0) {
			//printf("d_maxResidual[%d] = %f (index %d)\n", blockIdx.x, maxRes[0], maxResIndex[0]);
			state.d_maxResidual[blockIdx.x] = maxRes[0];
			state.d_maxResidualIndex[blockIdx.x] = maxResIndex[0];
		}
	}
}

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences * 3; // Number of correspondences (*3 per xyz)
	EvalMaxResidualDevice<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, state, parameters);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// Eval Cost
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float residual = 0.0f;
	if (x < N) {
		residual = evalFDevice(x, input, state, parameters);
		//float out = warpReduce(residual);
		//unsigned int laneid;
		////This command gets the lane ID within the current warp
		//asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
		//if (laneid == 0) {
		//	atomicAdd(&state.d_sumResidual[0], out);
		//}
		atomicAdd(&state.d_sumResidual[0], residual);
	}
}

float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	float residual = 0.0f;

	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	residual = state.getSumResidual();

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();

	return residual;
}

/////////////////////////////////////////////////////////////////////////
// Eval Linear Residual
/////////////////////////////////////////////////////////////////////////

//__global__ void SumLinearResDevice(SolverInput input, SolverState state, SolverParameters parameters)
//{
//	const unsigned int N = input.numberOfImages; // Number of block variables
//	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//
//	float residual = 0.0f;
//	if (x > 0 && x < N) {
//		residual = dot(state.d_rRot[x], state.d_rRot[x]) + dot(state.d_rTrans[x], state.d_rTrans[x]);
//		atomicAdd(state.d_sumLinResidual, residual);
//	}
//}
//float EvalLinearRes(SolverInput& input, SolverState& state, SolverParameters& parameters)
//{
//	float residual = 0.0f;
//
//	const unsigned int N = input.numberOfImages;	// Number of block variables
//
//	// Do PCG step
//	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//
//	float init = 0.0f;
//	cutilSafeCall(cudaMemcpy(state.d_sumLinResidual, &init, sizeof(float), cudaMemcpyHostToDevice));
//
//	SumLinearResDevice << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
//#ifdef _DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//
//	cutilSafeCall(cudaMemcpy(&residual, state.d_sumLinResidual, sizeof(float), cudaMemcpyDeviceToHost));
//	return residual;
//}

/////////////////////////////////////////////////////////////////////////
// Count High Residuals
/////////////////////////////////////////////////////////////////////////

__global__ void CountHighResidualsDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences * 3; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		const unsigned int corrIdx = x / 3;
		const unsigned int componentIdx = x - corrIdx * 3;
		float residual = evalResidualDeviceFloat3(corrIdx, componentIdx, input, state, parameters);

		if (residual > parameters.verifyOptDistThresh) 
			atomicAdd(state.d_countHighResidual, 1);
	}
}

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences * 3; // Number of correspondences (*3 per xyz)
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));
	CountHighResidualsDevice<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(input, state, parameters);
	
	int count;
	cutilSafeCall(cudaMemcpy(&count, state.d_countHighResidual, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();
	return count;
}




// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		float3 resRot, resTrans;
		evalMinusJTFDevice(x, input, state, parameters, resRot, resTrans);  // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		state.d_rRot[x] = resRot;											// store for next iteration
		state.d_rTrans[x] = resTrans;										// store for next iteration

		const float3 pRot = state.d_precondionerRot[x] * resRot;			// apply preconditioner M^-1
		state.d_pRot[x] = pRot;

		const float3 pTrans = state.d_precondionerTrans[x] * resTrans;		// apply preconditioner M^-1
		state.d_pTrans[x] = pTrans;

		d = dot(resRot, pRot) + dot(resTrans, pTrans);						// x-th term of nomimator for computing alpha and denominator for computing beta
	
		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
	}

	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];				// store result for next kernel call
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	if (timer) timer->startEvent("Init1");

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif		

	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif		
	if (timer) timer->endEvent();

	if (timer) timer->startEvent("Init2");
	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(N, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
		
	if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel0(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences;					// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		const float3 tmp = applyJDevice(x, input, state, parameters);		// A x p_k  => J^T x J x p_k 
		state.d_Jp[x] = tmp;												// store for next kernel call
	}
}

//__global__ void PCGStep_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
//
//	const unsigned int N = input.numberOfImages;							// Number of block variables
//	const unsigned int xHat = blockIdx.x * blockDim.x + threadIdx.x;
//
//	const unsigned int x = xHat / 32;
//	const unsigned int lane = threadIdx.x % WARP_SIZE;
//
//	float d = 0.0f;
//	if (x > 0 && x < N)
//	{
//		float3 rot, trans;
//		applyJTDevice(x, input, state, parameters, rot, trans, lane);			// A x p_k  => J^T x J x p_k 
//
//		if (lane == 0)
//		{
//			state.d_Ap_XRot[x] = rot;											// store for next kernel call
//			state.d_Ap_XTrans[x] = trans;										// store for next kernel call
//
//			d = dot(state.d_pRot[x], rot) + dot(state.d_pTrans[x], trans);		// x-th term of denominator of alpha
//		}
//	}
//	//d = warpReduce(d);
//	if (threadIdx.x % WARP_SIZE == 0)
//	{
//		atomicAdd(state.d_scanAlpha, d);
//	}
//

__global__ void PCGStep_Kernel1a(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N	= input.numberOfImages;							// Number of block variables
	const unsigned int x	= blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	//float d = 0.0f;
	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTDevice(x, input, state, parameters, rot, trans, threadIdx.x, lane);			// A x p_k  => J^T x J x p_k 

		if (lane == 0)
		{
			atomicAdd(&state.d_Ap_XRot[x].x, rot.x);
			atomicAdd(&state.d_Ap_XRot[x].y, rot.y);
			atomicAdd(&state.d_Ap_XRot[x].z, rot.z);

			atomicAdd(&state.d_Ap_XTrans[x].x, trans.x);
			atomicAdd(&state.d_Ap_XTrans[x].y, trans.y);
			atomicAdd(&state.d_Ap_XTrans[x].z, trans.z);
		}
	}
}

__global__ void PCGStep_Kernel1b(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;								// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		d = dot(state.d_pRot[x], state.d_Ap_XRot[x]) + dot(state.d_pTrans[x], state.d_Ap_XTrans[x]);		// x-th term of denominator of alpha
	}
	
	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x > 0 && x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x]/dotProduct;		// update step size alpha

		state.d_deltaRot[x] = state.d_deltaRot[x] + alpha*state.d_pRot[x];			// do a decent step
		state.d_deltaTrans[x] = state.d_deltaTrans[x] + alpha*state.d_pTrans[x];	// do a decent step

		float3 rRot = state.d_rRot[x] - alpha*state.d_Ap_XRot[x];					// update residuum
		state.d_rRot[x] = rRot;														// store for next kernel call

		float3 rTrans = state.d_rTrans[x] - alpha*state.d_Ap_XTrans[x];				// update residuum
		state.d_rTrans[x] = rTrans;													// store for next kernel call

		float3 zRot = state.d_precondionerRot[x] * rRot;							// apply preconditioner M^-1
		state.d_zRot[x] = zRot;														// save for next kernel call

		float3 zTrans = state.d_precondionerTrans[x] * rTrans;						// apply preconditioner M^-1
		state.d_zTrans[x] = zTrans;													// save for next kernel call

		b = dot(zRot, rRot) + dot(zTrans, rTrans);									// compute x-th term of the nominator of beta
	}
	b = warpReduce(b);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(&state.d_scanAlpha[1], b);
	}
}

template<bool lastIteration>
__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N)
	{
		const float rDotzNew = state.d_scanAlpha[1];								// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];								// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;				// update step size beta

		state.d_rDotzOld[x] = rDotzNew;											// save new rDotz for next iteration
		state.d_pRot[x]		= state.d_zRot  [x]	+ beta*state.d_pRot  [x];		// update decent direction
		state.d_pTrans[x]	= state.d_zTrans[x] + beta*state.d_pTrans[x];		// update decent direction


		state.d_Ap_XRot[x]   = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{
			state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
			state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
		}
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, bool lastIteration, CUDATimer *timer)
{
	const unsigned int N = input.numberOfImages;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	//if (timer) timer->startEvent("PCGIteration::applyJ");
	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)*2));
	//cutilSafeCall(cudaMemset(state.d_scanBeta, 0, sizeof(float)));

	const unsigned int Ncorr = input.numberOfCorrespondences;
	const int blocksPerGridCorr = (Ncorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	PCGStep_Kernel0 << <blocksPerGridCorr, THREADS_PER_BLOCK >> >(input, state, parameters);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
	//if (timer) timer->endEvent();

	//if (timer) timer->startEvent("PCGIteration::applyJTa");
	PCGStep_Kernel1a << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
	#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
	#endif
	//if (timer) timer->endEvent();

	//if (timer) timer->startEvent("PCGIteration::applyJTb");
	PCGStep_Kernel1b << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
	#ifdef _DEBUG
				cutilSafeCall(cudaDeviceSynchronize());
				cutilCheckMsg(__FUNCTION__);
	#endif
	//if (timer) timer->endEvent();

	//if (timer) timer->startEvent("PCGIteration::2");
	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
	//if (timer) timer->endEvent();
	
	//if (timer) timer->startEvent("PCGIteration::3");
	if (lastIteration) {
		PCGStep_Kernel3<true> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}
	else {
		PCGStep_Kernel3<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}
	
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	//if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N) {
		state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
		state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
	}
}

void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.numberOfImages; // Number of block variables
	ApplyLinearUpdateDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, float* convergenceAnalysis, CUDATimer *timer)
{
	if (convergenceAnalysis) {
		float initialResidual = EvalResidual(input, state, parameters, timer);
		//printf("initial = %f\n", initialResidual);
		convergenceAnalysis[0] = initialResidual; // initial residual
	}
	//unsigned int idx = 0;

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		Initialization(input, state, parameters, timer);
		
		//float linearResidual = EvalLinearRes(input, state, parameters);
		//linConvergenceAnalysis[idx++] = linearResidual;

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
		{
			PCGIteration(input, state, parameters, linIter == parameters.nLinIterations - 1, timer);

			//linearResidual = EvalLinearRes(input, state, parameters);
			//linConvergenceAnalysis[idx++] = linearResidual;
		}

		//ApplyLinearUpdate(input, state, parameters);	//this should be also done in the last PCGIteration

		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
			//printf("[niter %d] %f\n", nIter, residual);
		}
	}
}

////////////////////////////////////////////////////////////////////
// build variables to correspondences lookup
////////////////////////////////////////////////////////////////////

__global__ void BuildVariablesToCorrespondencesTableDevice(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow)
{
	const unsigned int N = numberOfCorrespondences; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		const EntryJ& corr = d_correspondences[x];
		if (corr.isValid()) {
			int offset = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_i], 1); // may overflow - need to check when read
			if (offset < maxNumCorrespondencesPerImage)	d_variablesToCorrespondences[corr.imgIdx_i * maxNumCorrespondencesPerImage + offset] = x;

			offset = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_j], 1); // may overflow - need to check when read
			if (offset < maxNumCorrespondencesPerImage)	d_variablesToCorrespondences[corr.imgIdx_j * maxNumCorrespondencesPerImage + offset] = x;
		}
	}
}

extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer)
{
	const unsigned int N = numberOfCorrespondences;

	if (timer) timer->startEvent(__FUNCTION__);

	BuildVariablesToCorrespondencesTableDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_correspondences, numberOfCorrespondences, maxNumCorrespondencesPerImage, d_variablesToCorrespondences, d_numEntriesPerRow);
	
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	if (timer) timer->endEvent();
}
