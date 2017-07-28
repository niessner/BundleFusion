#include <iostream>

////for debug purposes
//#define PRINT_RESIDUALS_SPARSE
//#define PRINT_RESIDUALS_DENSE

#define ENABLE_EARLY_OUT

#include "GlobalDefines.h"
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "SolverBundlingUtil.h"
#include "SolverBundlingEquations.h"
#include "SolverBundlingEquationsLie.h"
#include "SolverBundlingDenseUtil.h"
#include "../../SiftGPU/CUDATimer.h"

#include <conio.h>

#define THREADS_PER_BLOCK_DENSE_DEPTH 128
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64

#define THREADS_PER_BLOCK_DENSE_OVERLAP 512


/////////////////////////////////////////////////////////////////////////
// Dense Depth Term
/////////////////////////////////////////////////////////////////////////
template<bool usePairwise>
__global__ void FindImageImageCorr_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	// image indices
	unsigned int i, j; // project from j to i
	if (usePairwise) {
		i = blockIdx.x; j = blockIdx.y; // all pairwise
		if (i >= j) return;
	}
	else {
		i = blockIdx.x; j = i + 1; // frame-to-frame
	}
	if (input.d_validImages[i] == 0 || input.d_validImages[j] == 0) return;

	const unsigned int tidx = threadIdx.x;
	const unsigned int subWidth = input.denseDepthWidth / parameters.denseOverlapCheckSubsampleFactor;
	const unsigned int x = (tidx % subWidth) * parameters.denseOverlapCheckSubsampleFactor;
	const unsigned int y = (tidx / subWidth) * parameters.denseOverlapCheckSubsampleFactor;
	const unsigned int idx = y * input.denseDepthWidth + x;

	if (idx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform = state.d_xTransformInverses[i] * state.d_xTransforms[j];
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse();						//TODO PRECOMPUTE?
		float4x4 transform = invTransform_i * transform_j;
#endif
		//if (!computeAngleDiff(transform, 1.0f)) return; //~60 degrees //TODO HERE ANGIE
		//if (!computeAngleDiff(transform, 0.8f)) return; //~45 degrees
		if (!computeAngleDiff(transform, 0.52f)) return; //~30 degrees

		// find correspondence
		__shared__ int foundCorr[1]; foundCorr[0] = 0;
		__syncthreads();
		if (findDenseCorr(idx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[j].d_depthDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax)) { //i tgt, j src		//TODO PARAMS
			atomicAdd(foundCorr, 1);
		} // found correspondence
		__syncthreads();
		if (tidx == 0) {
			if (foundCorr[0] > 10) { //TODO PARAMS
				int addr = atomicAdd(state.d_numDenseOverlappingImages, 1);
				state.d_denseOverlappingImages[addr] = make_uint2(i, j);
			}
		}
	} // valid image pixel
}

__global__ void FlipJtJ_Kernel(unsigned int total, unsigned int dim, float* d_JtJ)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		const unsigned int x = idx % dim;
		const unsigned int y = idx / dim;
		if (x > y) {
			d_JtJ[y * dim + x] = d_JtJ[x * dim + y];
		}
	}
}
__global__ void FindDenseCorrespondences_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	const int imPairIdx = blockIdx.x; //should not go out of bounds, no need to check
	uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
	unsigned int i = imageIndices.x;	unsigned int j = imageIndices.y;

	const unsigned int tidx = threadIdx.x;
	const unsigned int gidx = tidx * gridDim.y + blockIdx.y;

	if (gidx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform = state.d_xTransformInverses[i] * state.d_xTransforms[j]; //invTransform_i * transform_j
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse();						
		float4x4 transform = invTransform_i * transform_j;
#endif
		// find correspondence
		const int numWarps = THREADS_PER_BLOCK_DENSE_DEPTH / WARP_SIZE;
		__shared__ int s_count[numWarps];
		s_count[0] = 0;
		int count = 0.0f;
		//TODO HERE ANGIE
#ifdef CUDACACHE_UCHAR_NORMALS
		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampledUCHAR4,
			input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampledUCHAR4,
			parameters.denseDepthMin, parameters.denseDepthMax)) { //i tgt, j src
#elif defined(CUDACACHE_FLOAT_NORMALS)
		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
			input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax)) { //i tgt, j src
#endif
//#ifdef CUDACACHE_UCHAR_NORMALS
//		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
//			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
//			input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampledUCHAR4,
//			input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampledUCHAR4,
//			parameters.denseDepthMin, parameters.denseDepthMax)) { //i tgt, j src
//#else
//		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
//			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
//			input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
//			input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
//			parameters.denseDepthMin, parameters.denseDepthMax)) { //i tgt, j src
//#endif
			//atomicAdd(&state.d_denseCorrCounts[imPairIdx], 1.0f);
			count++;
		} // found correspondence
		count = warpReduce(count);
		__syncthreads();
		if (tidx % WARP_SIZE == 0) {
			s_count[tidx / WARP_SIZE] = count;
			//atomicAdd(&state.d_denseCorrCounts[imPairIdx], count);
		}
		__syncthreads();
		for (unsigned int stride = numWarps / 2; stride > 0; stride /= 2) {
			if (tidx < stride) s_count[tidx] = s_count[tidx] + s_count[tidx + stride];
			__syncthreads();
		}
		if (tidx == 0) {
			atomicAdd(&state.d_denseCorrCounts[imPairIdx], s_count[0]);
		}
	} // valid image pixel
}

__global__ void WeightDenseCorrespondences_Kernel(unsigned int N, SolverState state)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		// apply ln to weights
		float x = state.d_denseCorrCounts[idx];
		if (x > 0) {
			//if (x < 3200) state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr //TODO PARAMS
			if (x < 800) state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr //TODO PARAMS
			//if (x < 400) state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr //TODO PARAMS
			//if (x < 200) state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr //TODO PARAMS //TODO EVAL DEBUG
			else {
				state.d_denseCorrCounts[idx] = 1.0f / min(logf(x), 9.0f); // natural log //TODO PARAMS
			}

			//state.d_denseCorrCounts[idx] = 1.0f / clamp(logf(x), 2.0f, 9.0f); // natural log //TODO PARAMS
		}
	}
}

template<bool useDepth, bool useColor>
__global__ void BuildDenseSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	const int imPairIdx = blockIdx.x;
	uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
	unsigned int i = imageIndices.x;	unsigned int j = imageIndices.y;

	float imPairWeight = state.d_denseCorrCounts[imPairIdx];
	if (imPairWeight == 0.0f) return;

	const unsigned int idx = threadIdx.x;
	const unsigned int srcIdx = idx * gridDim.y + blockIdx.y;

	if (srcIdx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform_i = state.d_xTransforms[i];
		float4x4 transform_j = state.d_xTransforms[j];
		float4x4 invTransform_i = state.d_xTransformInverses[i];
		float4x4 invTransform_j = state.d_xTransformInverses[j];
		float4x4 transform = invTransform_i * transform_j;
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse();						//TODO PRECOMPUTE?
		float4x4 transform = invTransform_i * transform_j;
#endif
		// point-to-plane term
		matNxM<1, 6> depthJacBlockRow_i, depthJacBlockRow_j; depthJacBlockRow_i.setZero(); depthJacBlockRow_j.setZero();
		float depthRes = 0.0f; float depthWeight = 0.0f;
		// color term
		matNxM<1, 6> colorJacBlockRow_i, colorJacBlockRow_j; colorJacBlockRow_i.setZero(); colorJacBlockRow_j.setZero();
		float colorRes = 0.0f; float colorWeight = 0.0f;

		// find correspondence
		float3 camPosSrc; float3 camPosSrcToTgt; float3 camPosTgt; float3 normalTgt; float2 tgtScreenPos;
		//TODO HERE ANGIE
#ifdef CUDACACHE_FLOAT_NORMALS
		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt); //i tgt, j src
#elif defined(CUDACACHE_UCHAR_NORMALS)
		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampledUCHAR4,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampledUCHAR4,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt); //i tgt, j src
#endif
//#ifdef CUDACACHE_UCHAR_NORMALS
//		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
//			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
//			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampledUCHAR4,
//			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampledUCHAR4,
//			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt); //i tgt, j src
//#else
//		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
//			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
//			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
//			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
//			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt); //i tgt, j src
//#endif
		if (useDepth) {
			if (foundCorr) {
				// point-to-plane residual
				float3 diff = camPosTgt - camPosSrcToTgt;
				depthRes = dot(diff, normalTgt);
				//depthWeight = parameters.weightDenseDepth * imPairWeight * max(0.0f, 0.5f*((1.0f - length(diff) / parameters.denseDistThresh) + (1.0f - camPosTgt.z / parameters.denseDepthMax)));
				//depthWeight = parameters.weightDenseDepth * imPairWeight * max(0.0f, (1.0f - camPosTgt.z / 2.0f)); //fr1_desk
				//depthWeight = parameters.weightDenseDepth * imPairWeight * max(0.0f, (1.0f - camPosTgt.z / 2.5f)); //fr3_office, fr2_xyz_half // livingroom1
				//depthWeight = parameters.weightDenseDepth * imPairWeight * max(0.0f, (1.0f - camPosTgt.z / 3.0f)); //fr3_nstn
				//depthWeight = parameters.weightDenseDepth * imPairWeight * max(0.0f, (1.0f - camPosTgt.z / 1.8f));
				//depthWeight = parameters.weightDenseDepth * imPairWeight * (pow(max(0.0f, 1.0f - camPosTgt.z / 2.5f), 1.8f));
				//depthWeight = parameters.weightDenseDepth * imPairWeight * (pow(max(0.0f, 1.0f - camPosTgt.z / 2.0f), 1.8f)); //fr3_office, fr1_desk_f20
				depthWeight = parameters.weightDenseDepth * imPairWeight * (pow(max(0.0f, 1.0f - camPosTgt.z / 2.0f), 2.5f)); //fr2_xyz_half
				//depthWeight = parameters.weightDenseDepth * imPairWeight * (pow(max(0.0f, 1.0f - camPosTgt.z / 3.5f), 1.8f)); //fr3_nstn

				//depthWeight = parameters.weightDenseDepth * imPairWeight * (pow(max(0.0f, 1.0f - camPosTgt.z / parameters.denseDepthMax), 1.8f)); //TODO EVAL DEBUGGING

				//float wtgt = (pow(max(0.0f, 1.0f - camPosTgt.z / 2.5f), 1.8f));
				//float wsrc = (pow(max(0.0f, 1.0f - camPosSrc.z / 2.5f), 1.8f));
				//depthWeight = parameters.weightDenseDepth * imPairWeight * wtgt * wsrc;
#ifdef USE_LIE_SPACE
				if (i > 0) computeJacobianBlockRow_i(depthJacBlockRow_i, transform_i, invTransform_j, camPosSrc, normalTgt);
				if (j > 0) computeJacobianBlockRow_j(depthJacBlockRow_j, invTransform_i, transform_j, camPosSrc, normalTgt);
#else
				if (i > 0) computeJacobianBlockRow_i(depthJacBlockRow_i, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, normalTgt);
				if (j > 0) computeJacobianBlockRow_j(depthJacBlockRow_j, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, normalTgt);
#endif
			}
			addToLocalSystem(foundCorr, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
				depthJacBlockRow_i, depthJacBlockRow_j, i, j, depthRes, depthWeight, idx
				, state.d_sumResidual, state.d_corrCount);
			//addToLocalSystemBrute(foundCorr, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
			//	depthJacBlockRow_i, depthJacBlockRow_j, i, j, depthRes, depthWeight, idx);
		}
		if (useColor) {
			bool foundCorrColor = false;
			if (foundCorr) {
				const float2 intensityDerivTgt = bilinearInterpolationFloat2(tgtScreenPos.x, tgtScreenPos.y, input.d_cacheFrames[i].d_intensityDerivsDownsampled, input.denseDepthWidth, input.denseDepthHeight);
				const float intensityTgt = bilinearInterpolationFloat(tgtScreenPos.x, tgtScreenPos.y, input.d_cacheFrames[i].d_intensityDownsampled, input.denseDepthWidth, input.denseDepthHeight);
				colorRes = intensityTgt - input.d_cacheFrames[j].d_intensityDownsampled[srcIdx];
				foundCorrColor = (intensityDerivTgt.x != MINF && abs(colorRes) < parameters.denseColorThresh && length(intensityDerivTgt) > parameters.denseColorGradientMin);
				if (foundCorrColor) {
					const float2 focalLength = make_float2(input.intrinsics.x, input.intrinsics.y);
#ifdef USE_LIE_SPACE
					if (i > 0) computeJacobianBlockIntensityRow_i(colorJacBlockRow_i, focalLength, transform_i, invTransform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (j > 0) computeJacobianBlockIntensityRow_j(colorJacBlockRow_j, focalLength, invTransform_i, transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
#else
					if (i > 0) computeJacobianBlockIntensityRow_i(colorJacBlockRow_i, focalLength, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (j > 0) computeJacobianBlockIntensityRow_j(colorJacBlockRow_j, focalLength, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
#endif
					colorWeight = parameters.weightDenseColor * imPairWeight * max(0.0f, 1.0f - abs(colorRes) / (1.15f*parameters.denseColorThresh));
					//colorWeight = parameters.weightDenseColor * imPairWeight * max(0.0f, 1.0f - abs(colorRes) / parameters.denseColorThresh) * max(0.0f, (1.0f - camPosTgt.z / 1.0f));
					//colorWeight = parameters.weightDenseColor * imPairWeight * max(0.0f, 0.5f*(1.0f - abs(colorRes) / parameters.denseColorThresh) + 0.5f*max(0.0f, (1.0f - camPosTgt.z / 1.0f)));
				}
			}
			addToLocalSystem(foundCorrColor, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
				colorJacBlockRow_i, colorJacBlockRow_j, i, j, colorRes, colorWeight, idx
				, state.d_sumResidualColor, state.d_corrCountColor);
			//addToLocalSystemBrute(foundCorrColor, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
			//	colorJacBlockRow_i, colorJacBlockRow_j, i, j, colorRes, colorWeight, idx);
		}
	} // valid image pixel
}

bool BuildDenseSystem(const SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;
	const int sizeJtr = 6 * N;
	const int sizeJtJ = sizeJtr * sizeJtr;

#ifdef PRINT_RESIDUALS_DENSE
	cutilSafeCall(cudaMemset(state.d_corrCount, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidual, 0, sizeof(float)));
	cutilSafeCall(cudaMemset(state.d_corrCountColor, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidualColor, 0, sizeof(float)));
#endif

	const unsigned int maxDenseImPairs = input.numberOfImages * (input.numberOfImages - 1) / 2;
	cutilSafeCall(cudaMemset(state.d_denseCorrCounts, 0, sizeof(float) * maxDenseImPairs));
	cutilSafeCall(cudaMemset(state.d_denseJtJ, 0, sizeof(float) * sizeJtJ));
	cutilSafeCall(cudaMemset(state.d_denseJtr, 0, sizeof(float) * sizeJtr));
	cutilSafeCall(cudaMemset(state.d_numDenseOverlappingImages, 0, sizeof(int)));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	dim3 gridImImOverlap;
	if (parameters.useDenseDepthAllPairwise) gridImImOverlap = dim3(N, N, 1); // pairwise
	else gridImImOverlap = dim3(N - 1, 1, 1); // for frame-to-frame

	if (timer) timer->startEvent("BuildDenseDepthSystem - find image corr");
	if (parameters.useDenseDepthAllPairwise) FindImageImageCorr_Kernel<true> << < gridImImOverlap, THREADS_PER_BLOCK_DENSE_OVERLAP >> >(input, state, parameters);
	else									 FindImageImageCorr_Kernel<false> << < gridImImOverlap, THREADS_PER_BLOCK_DENSE_OVERLAP >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();

	int numOverlapImagePairs;
	cutilSafeCall(cudaMemcpy(&numOverlapImagePairs, state.d_numDenseOverlappingImages, sizeof(int), cudaMemcpyDeviceToHost));
	if (numOverlapImagePairs == 0) {
		printf("warning: no overlapping images for dense solve\n");
		return false;
	}
	const int reductionGlobal = (input.denseDepthWidth*input.denseDepthHeight + THREADS_PER_BLOCK_DENSE_DEPTH - 1) / THREADS_PER_BLOCK_DENSE_DEPTH;
	dim3 grid(numOverlapImagePairs, reductionGlobal);
	//if (N > 11) printf("num overlap image pairs = %d\n", numOverlapImagePairs); //debugging only

	if (timer) timer->startEvent("BuildDenseDepthSystem - compute im-im weights");

	FindDenseCorrespondences_Kernel << <grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	////debugging //remember the delete!
	//float* denseCorrCounts = new float[numOverlapImagePairs];
	//cutilSafeCall(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts, sizeof(float)*numOverlapImagePairs, cudaMemcpyDeviceToHost));
	//unsigned int totalCount = 0;
	//for (unsigned int i = 0; i < numOverlapImagePairs; i++) { totalCount += (unsigned int)denseCorrCounts[i]; }
	//printf("total count = %d\n", totalCount);

	//uint2* imageIndices = new uint2[numOverlapImagePairs];
	//cutilSafeCall(cudaMemcpy(imageIndices, state.d_denseOverlappingImages, sizeof(uint2)*numOverlapImagePairs, cudaMemcpyDeviceToHost));
	//if (imageIndices) delete[] imageIndices;
	////debugging

	//debugging - compute some overlap stats
	//if (true || input.numberOfImages > 11) {
	//	float4x4* transforms = new float4x4[input.numberOfImages];
	//	float* denseCorrCounts = new float[numOverlapImagePairs];
	//	uint2* imageIndices = new uint2[numOverlapImagePairs];
	//	cutilSafeCall(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts, sizeof(float)*numOverlapImagePairs, cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(imageIndices, state.d_denseOverlappingImages, sizeof(uint2)*numOverlapImagePairs, cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(transforms, state.d_xTransforms, sizeof(float4x4)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//	FILE* fp = fopen("debug/overlaps.csv", "w");
	//	char buffer[128];
	//	for (int i = 0; i < numOverlapImagePairs; i++) {
	//		if (denseCorrCounts[i] > 0) {
	//			float3 d = transforms[imageIndices[i].x].getTranslation() - transforms[imageIndices[i].y].getTranslation();
	//			sprintf(buffer, "%d,%d,%d,%f\n", imageIndices[i].x, imageIndices[i].y, (int)denseCorrCounts[i], length(d));
	//			fwrite(buffer, sizeof(char), strlen(buffer), fp);
	//		}
	//	}
	//	fclose(fp);
	//	if (transforms) delete[] transforms;
	//	if (denseCorrCounts) delete[] denseCorrCounts;
	//	if (imageIndices) delete[] imageIndices;
	//	int a = 5;
	//}

	int wgrid = (numOverlapImagePairs + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
	WeightDenseCorrespondences_Kernel << < wgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >(maxDenseImPairs, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	////debugging
	//cutilSafeCall(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts, sizeof(float)*maxDenseImPairs, cudaMemcpyDeviceToHost));
	//totalCount = 0;
	//for (unsigned int i = 0; i < maxDenseImPairs; i++) { if (denseCorrCounts[i] > 0.0f) totalCount++; }
	//printf("total count = %d\n", totalCount);
	//if (denseCorrCounts) delete[] denseCorrCounts;
	////debugging
	if (timer) timer->endEvent();
	if (timer) timer->startEvent("BuildDenseDepthSystem - build jtj/jtr");

	if (parameters.weightDenseDepth > 0.0f) {
		if (parameters.weightDenseColor > 0.0f) BuildDenseSystem_Kernel<true, true> << <grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >(input, state, parameters);
		else									BuildDenseSystem_Kernel<true, false> << <grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >(input, state, parameters);
	}
	else {
		BuildDenseSystem_Kernel<false, true> << <grid, THREADS_PER_BLOCK_DENSE_DEPTH >> >(input, state, parameters);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	
	////debugging
	//bool debugPrint = true;
	//float* h_JtJ = NULL;
	//float* h_Jtr = NULL;
	//if (debugPrint) {
	//	h_JtJ = new float[sizeJtJ];
	//	h_Jtr = new float[sizeJtr];
	//	cutilSafeCall(cudaMemcpy(h_JtJ, state.d_denseJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(h_Jtr, state.d_denseJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
	//	printf("JtJ:\n");
	//	//for (unsigned int i = 0; i < 6 * N; i++) {
	//	//	for (unsigned int j = 0; j < 6 * N; j++)
	//	for (unsigned int i = 6 * 1; i < 6 * 2; i++) {
	//		for (unsigned int j = 6 * 1; j < 6 * 2; j++)
	//			printf(" %f,", h_JtJ[j * 6 * N + i]);
	//		printf("\n");
	//	}
	//	printf("Jtr:\n");
	//	for (unsigned int i = 0; i < 6 * N; i++) {
	//		printf(" %f,", h_Jtr[i]);
	//	}
	//	printf("\n");
	//}
	////debugging
#ifdef PRINT_RESIDUALS_DENSE
	if (parameters.weightDenseDepth > 0) {
		float sumResidual; int corrCount;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(&corrCount, state.d_corrCount, sizeof(int), cudaMemcpyDeviceToHost));
		printf("\tdense depth: weights * residual = %f * %f = %f\t[#corr = %d]\n", parameters.weightDenseDepth, sumResidual / parameters.weightDenseDepth, sumResidual, corrCount);
	}
	if (parameters.weightDenseColor > 0) {
		float sumResidual; int corrCount;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidualColor, sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(&corrCount, state.d_corrCountColor, sizeof(int), cudaMemcpyDeviceToHost));
		printf("\tdense color: weights * residual = %f * %f = %f\t[#corr = %d]\n", parameters.weightDenseColor, sumResidual / parameters.weightDenseColor, sumResidual, corrCount);
	}
#endif
	const unsigned int flipgrid = (sizeJtJ + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
	FlipJtJ_Kernel << <flipgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >(sizeJtJ, sizeJtr, state.d_denseJtJ);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();
	return true;
}

//todo more efficient?? (there are multiple per image-image...)
//get high residuals
__global__ void collectHighResidualsDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters, unsigned int maxNumHighResiduals)
{
	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);
		if (residual > parameters.highResidualThresh) {
			int idx = atomicAdd(state.d_countHighResidual, 1);
			if (idx < maxNumHighResiduals) {
				analysis.d_maxResidual[idx] = residual;
				analysis.d_maxResidualIndex[idx] = corrIdx;
			}
		}
	}
}
extern "C" void collectHighResiduals(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));

	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences 
	unsigned int maxNumHighResiduals = (input.maxCorrPerImage*input.maxNumberOfImages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	collectHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters, maxNumHighResiduals);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// Eval Max Residual
/////////////////////////////////////////////////////////////////////////

__global__ void EvalMaxResidualDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters)
{
	__shared__ int maxResIndex[THREADS_PER_BLOCK];
	__shared__ float maxRes[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	maxResIndex[threadIdx.x] = 0;
	maxRes[threadIdx.x] = 0.0f;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		maxRes[threadIdx.x] = residual;
		maxResIndex[threadIdx.x] = corrIdx;

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
			analysis.d_maxResidual[blockIdx.x] = maxRes[0];
			analysis.d_maxResidualIndex[blockIdx.x] = maxResIndex[0];
		}
	}
}

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences 
	EvalMaxResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters);

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

extern "C" float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
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
	const unsigned int N = input.numberOfCorrespondences; // Number of block variables
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		if (residual > parameters.verifyOptDistThresh)
			atomicAdd(state.d_countHighResidual, 1);
	}
}

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences; // Number of correspondences
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));
	CountHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	int count;
	cutilSafeCall(cudaMemcpy(&count, state.d_countHighResidual, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();
	return count;
}

/////////////////////////////////////////////////////////////////////////
// Convergence Analysis
/////////////////////////////////////////////////////////////////////////

//uses same data store as max residual
__global__ void EvalGNConvergenceDevice(SolverInput input, SolverStateAnalysis analysis, SolverState state) //compute max of delta
{
	__shared__ float maxVal[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	maxVal[threadIdx.x] = 0.0f;

	if (x < N)
	{
		if (x == 0 || input.d_validImages[x] == 0)
			maxVal[threadIdx.x] = 0.0f;
		else {
			float3 r3 = fmaxf(fabs(state.d_deltaRot[x]), fabs(state.d_deltaTrans[x]));
			float r = fmaxf(r3.x, fmaxf(r3.y, r3.z));
			maxVal[threadIdx.x] = r;
		}
		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				maxVal[first] = fmaxf(maxVal[first], maxVal[second]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			analysis.d_maxResidual[blockIdx.x] = maxVal[0];
		}
	}
}
float EvalGNConvergence(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfImages;
	const unsigned int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	EvalGNConvergenceDevice << < blocksPerGrid, THREADS_PER_BLOCK >> >(input, analysis, state);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	//copy to host and compute max
	cutilSafeCall(cudaMemcpy(analysis.h_maxResidual, analysis.d_maxResidual, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(analysis.h_maxResidualIndex, analysis.d_maxResidualIndex, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost));
	float maxVal = 0.0f;
	for (unsigned int i = 0; i < blocksPerGrid; i++) {
		if (maxVal < analysis.h_maxResidual[i]) maxVal = analysis.h_maxResidual[i];
	}
	if (timer) timer->endEvent();

	return maxVal;
}

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

template<bool useDense>
__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		float3 resRot, resTrans;
		evalMinusJTFDevice<useDense>(x, input, state, parameters, resRot, resTrans);  // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

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

	if (timer) timer->startEvent("Initialization");

	//!!!DEBUGGING //remember to uncomment the delete...
	//float3* rRot = new float3[input.numberOfImages]; // -jtf
	//float3* rTrans = new float3[input.numberOfImages];
	//!!!DEBUGGING

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif		

	if (parameters.useDense) PCGInit_Kernel1<true> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
	else PCGInit_Kernel1<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif		

	//cutilSafeCall(cudaMemcpy(rRot, state.d_rRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(rTrans, state.d_rTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rRot[i].x)) { printf("NaN in jtr rRot %d\n", i); getchar(); } }
	//for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rTrans[i].x)) { printf("NaN in jtr rTrans %d\n", i); getchar(); } }
	//cutilSafeCall(cudaMemcpy(rRot, state.d_pRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(rTrans, state.d_pTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rRot[i].x)) { printf("NaN in jtr pRot %d\n", i); getchar(); } }
	//for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rTrans[i].x)) { printf("NaN in jtr pTrans %d\n", i); getchar(); } }

	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(N, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();

	//float scanAlpha;
	//cutilSafeCall(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	//if (rRot) delete[] rRot;
	//if (rTrans) delete[] rTrans;
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

//inefficient
__global__ void PCGStep_Kernel_Dense_Brute(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDenseBruteDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans); // A x p_k  => J^T x J x p_k 

		state.d_Ap_XRot[x] += rot;
		state.d_Ap_XTrans[x] += trans;
	}
}
__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDenseDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans, threadIdx.x);			// A x p_k  => J^T x J x p_k 

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

__global__ void PCGStep_Kernel1a(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

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
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;		// update step size alpha

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
		state.d_pRot[x] = state.d_zRot[x] + beta*state.d_pRot[x];		// update decent direction
		state.d_pTrans[x] = state.d_zTrans[x] + beta*state.d_pTrans[x];		// update decent direction


		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{
			//if (input.d_validImages[x]) { //not really necessary
#ifdef USE_LIE_SPACE //TODO just keep that matrix transforms around
			float3 rot, trans;
			computeLieUpdate(state.d_deltaRot[x], state.d_deltaTrans[x], state.d_xRot[x], state.d_xTrans[x], rot, trans);
			state.d_xRot[x] = rot;
			state.d_xTrans[x] = trans;
#else
			state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
			state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
#endif
			//}
		}
	}
}

template<bool useSparse, bool useDense>
bool PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, bool lastIteration, CUDATimer *timer)
{
	const unsigned int N = input.numberOfImages;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}
	if (timer) timer->startEvent("PCGIteration");

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	// sparse part
	if (useSparse) {
		const unsigned int Ncorr = input.numberOfCorrespondences;
		const int blocksPerGridCorr = (Ncorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		PCGStep_Kernel0 << <blocksPerGridCorr, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		PCGStep_Kernel1a << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
	if (useDense) {
		//if (timer) timer->startEvent("apply JTJ dense");
		PCGStep_Kernel_Dense << < N, THREADS_PER_BLOCK_JT_DENSE >> >(input, state, parameters);
		//PCGStep_Kernel_Dense_Brute << < N, 1 >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		//if (timer) timer->endEvent();
	}
	//!!!debugging
	//float3* Ap_Rot = new float3[input.numberOfImages];
	//float3* Ap_Trans = new float3[input.numberOfImages];
	//cutilSafeCall(cudaMemcpy(Ap_Rot, state.d_Ap_XRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(Ap_Trans, state.d_Ap_XTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//for (unsigned int i = 1; i < input.maxNumberOfImages; i++) { if (isnan(Ap_Rot[i].x)) { printf("NaN at Ap rot %d\n", i); getchar(); } }
	//for (unsigned int i = 1; i < input.maxNumberOfImages; i++) { if (isnan(Ap_Trans[i].x)) { printf("NaN at Ap trans %d\n", i); getchar(); } }
	//if (Ap_Rot) delete[] Ap_Rot;
	//if (Ap_Trans) delete[] Ap_Trans;
	//!!!debugging

	PCGStep_Kernel1b << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
#ifdef ENABLE_EARLY_OUT //for convergence
	float scanAlpha; cutilSafeCall(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	//if (fabs(scanAlpha) < 0.00005f) lastIteration = true;  //todo check this part
	//if (fabs(scanAlpha) < 1e-6) lastIteration = true;  //todo check this part
	if (fabs(scanAlpha) < 5e-7) { lastIteration = true; }  //todo check this part
#endif
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
	if (timer) timer->endEvent();

	return lastIteration;
}

#ifdef USE_LIE_SPACE //TODO
////////////////////////////////////////////////////////////////////
// matrix <-> pose
////////////////////////////////////////////////////////////////////
__global__ void convertLiePosesToMatricesCU_Kernel(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numTransforms) {
		poseToMatrix(d_rot[idx], d_trans[idx], d_transforms[idx]);
		d_transformInvs[idx] = d_transforms[idx].getInverse();
	}
}
extern "C"
void convertLiePosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs)
{
	convertLiePosesToMatricesCU_Kernel << <(numTransforms + 8 - 1) / 8, 8 >> >(d_rot, d_trans, numTransforms, d_transforms, d_transformInvs);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
#endif

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer *timer)
{
	if (convergenceAnalysis) {
		float initialResidual = EvalResidual(input, state, parameters, timer);
		convergenceAnalysis[0] = initialResidual; // initial residual
	}

	//!!!DEBUGGING
#ifdef PRINT_RESIDUALS_SPARSE
	if (parameters.weightSparse > 0) {
		if (input.numberOfCorrespondences == 0) { printf("ERROR: %d correspondences\n", input.numberOfCorrespondences); getchar(); }
		float initialResidual = EvalResidual(input, state, parameters, timer);
		printf("initial sparse = %f*%f = %f\n", parameters.weightSparse, initialResidual / parameters.weightSparse, initialResidual);
	}
#endif
	//float3* xRot = new float3[input.numberOfImages];	//remember the delete!
	//float3* xTrans = new float3[input.numberOfImages];
	//timer = new CUDATimer();
	//static unsigned int totalLinIters = 0, numLin = 0, totalNonLinIters = 0, numNonLin = 0;
	//!!!DEBUGGING

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		parameters.weightSparse = input.weightsSparse[nIter];
		parameters.weightDenseDepth = input.weightsDenseDepth[nIter];
		parameters.weightDenseColor = input.weightsDenseColor[nIter];
		parameters.useDense = (parameters.weightDenseDepth > 0 || parameters.weightDenseColor > 0);
#ifdef USE_LIE_SPACE
		convertLiePosesToMatricesCU(state.d_xRot, state.d_xTrans, input.numberOfImages, state.d_xTransforms, state.d_xTransformInverses);
#endif
		if (parameters.useDense) parameters.useDense = BuildDenseSystem(input, state, parameters, timer); //don't solve dense if no overlapping frames found
		Initialization(input, state, parameters, timer);

		if (parameters.weightSparse > 0.0f) {
			if (parameters.useDense) {
				for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
					if (PCGIteration<true, true>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) { break; }
			}
			else {
				for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
					if (PCGIteration<true, false>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) {
						//totalLinIters += (linIter+1); numLin++; 
						break;
					}
			}
		}
		else {
			for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
				if (PCGIteration<false, true>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) break;
		}
		//!!!debugging
		//cutilSafeCall(cudaMemcpy(xRot, state.d_xRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//cutilSafeCall(cudaMemcpy(xTrans, state.d_xTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//!!!debugging
#ifdef PRINT_RESIDUALS_SPARSE
		if (parameters.weightSparse > 0) {
			float residual = EvalResidual(input, state, parameters, timer);
			printf("[niter %d] weight * sparse = %f*%f = %f\t[#corr = %d]\n", nIter, parameters.weightSparse, residual / parameters.weightSparse, residual, input.numberOfCorrespondences);
		}
#endif
		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
		}

		//if (timer) timer->evaluate(true);

#ifdef ENABLE_EARLY_OUT //convergence
		//if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.01f) { //!!! TODO CHECK HOW THESE GENERALIZE
		if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.005f) { //0.001?
		//if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.001f) { 
			//if (!parameters.useDense) { totalNonLinIters += (nIter+1); numNonLin++; }
			break;
		}
		//else if (!parameters.useDense && nIter == parameters.nNonLinearIterations - 1) { totalNonLinIters += (nIter+1); numNonLin++; }
#endif
		}
	//!!!debugging
	//if (xRot) delete[] xRot;
	//if (xTrans) delete[] xTrans;
	//if (timer) { timer->evaluate(true, false); delete timer; }
	//if (!parameters.useDense) { printf("mean #pcg its = %f\tmean #gn its = %f\n", (float)totalLinIters / (float)numLin, (float)totalNonLinIters / (float)numNonLin); } //just stats for global solve
	//!!!debugging
	}

////////////////////////////////////////////////////////////////////
// build variables to correspondences lookup
////////////////////////////////////////////////////////////////////

__global__ void BuildVariablesToCorrespondencesTableDevice(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,
	unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow)
{
	const unsigned int N = numberOfCorrespondences; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		EntryJ& corr = d_correspondences[x];
		if (corr.isValid()) {
			int offset0 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_i], 1); // may overflow - need to check when read
			int offset1 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_j], 1); // may overflow - need to check when read
			if (offset0 < maxNumCorrespondencesPerImage && offset1 < maxNumCorrespondencesPerImage)	{
				d_variablesToCorrespondences[corr.imgIdx_i * maxNumCorrespondencesPerImage + offset0] = x;
				d_variablesToCorrespondences[corr.imgIdx_j * maxNumCorrespondencesPerImage + offset1] = x;
			}
			else { //invalidate
				printf("EXCEEDED MAX NUM CORR PER IMAGE IN SOLVER, INVALIDATING %d(%d,%d) [%d,%d | %d]\n",
					x, corr.imgIdx_i, corr.imgIdx_j, offset0, offset1, maxNumCorrespondencesPerImage); //debugging
				corr.setInvalid(); //make sure j corresponds to jt
			}
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
