#include <iostream>

#include "GlobalDefines.h"
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "SolverBundlingUtil.h"
#include "SolverBundlingEquations.h"
#include "SolverBundlingEquationsLie.h"
#include "../../SiftGPU/CUDATimer.h"

#include <conio.h>

//!!!DEBUGGING
#define PRINT_RESIDUALS_SPARSE
#define PRINT_RESIDUALS_DENSE
//!!!DEBUGGING
#define THREADS_PER_BLOCK_DENSE_DEPTH_X 32
#define THREADS_PER_BLOCK_DENSE_DEPTH_Y 4 
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64

/////////////////////////////////////////////////////////////////////////
// Dense Depth Term
/////////////////////////////////////////////////////////////////////////
__global__ void FlipJtJ_Kernel(unsigned int total, unsigned int dim, float* d_JtJ)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		const unsigned int x = idx % dim;
		const unsigned int y = idx / dim;
		if (x > y) {
			//!!!debugging
			if (y * dim + x >= total || x * dim + y >= total) {
				printf("ERROR FlipJtJ: (%d,%d) [%d,%d] out of bounds!\n", x, y, total, dim);
			}
			//!!!debugging
			d_JtJ[y * dim + x] = d_JtJ[x * dim + y];
		}
	}
}
__device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4x4& intrinsics,
	const float4* tgtCamPos, const float4* tgtNormals, const float4* srcCamPos, const float4* srcNormals,
	float depthMin, float depthMax, float4& camPosSrcToTgt, unsigned int& tgtIdx)
{
	const float4 cposj = srcCamPos[idx];
	//if (debugPrint) printf("cam pos j = %f %f %f\n", cposj.x, cposj.y, cposj.z);
	if (cposj.z > depthMin && cposj.z < depthMax) {
		float4 nrmj = srcNormals[idx];
		//if (debugPrint) printf("normal j = %f %f %f\n", nrmj.x, nrmj.y, nrmj.z);
		if (nrmj.x != MINF) {
			nrmj = transform * nrmj;
			camPosSrcToTgt = transform * cposj;
			float3 proj = intrinsics * make_float3(camPosSrcToTgt.x, camPosSrcToTgt.y, camPosSrcToTgt.z);
			int2 tgtScreenPos = make_int2((int)roundf(proj.x / proj.z), (int)roundf(proj.y / proj.z));
			//if (debugPrint) {
			//	printf("cam pos j2i = %f %f %f\n", camPosSrcToTgt.x, camPosSrcToTgt.y, camPosSrcToTgt.z);
			//	printf("proj %f %f %f -> %f %f\n", proj.x, proj.y, proj.z, proj.x / proj.z, proj.y / proj.z);
			//	printf("screen pos = %d %d\n", tgtScreenPos.x, tgtScreenPos.y);
			//}
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				tgtIdx = tgtScreenPos.y * imageWidth + tgtScreenPos.x;
				const float4 camPosTgt = tgtCamPos[tgtIdx];
				//if (debugPrint) printf("cam pos i = %f %f %f\n", camPosTgt.x, camPosTgt.y, camPosTgt.z);
				if (camPosTgt.z > depthMin && camPosTgt.z < depthMax) {
					const float4 normalTgt = tgtNormals[tgtIdx];
					//if (debugPrint) printf("normal i = %f %f %f\n", normalTgt.x, normalTgt.y, normalTgt.z);
					if (normalTgt.x != MINF) {
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt);
						//if (debugPrint) printf("dist = %f, dnormal = %f\n", dist, dNormal);
						if (dNormal >= normalThresh && dist <= distThresh) {
							return true;
						}
					}
				}
			} // valid projection
		} // valid src normal
	} // valid src camera position
	return false;
}
__global__ void FindDenseCorrespondences_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	// image indices
	unsigned int i, j; // project from j to i
	unsigned int imPairIdx;
	if (parameters.useDenseDepthAllPairwise) {
		i = blockIdx.x; j = blockIdx.y; // all pairwise
		if (i >= j) return;
		imPairIdx = i * gridDim.x + j;
	}
	else {
		i = blockIdx.x; j = i + 1; // frame-to-frame
		imPairIdx = i;
	}
	if (input.d_validImages[i] == 0 || input.d_validImages[j] == 0) return;

	const unsigned int idx = threadIdx.y * THREADS_PER_BLOCK_DENSE_DEPTH_X + threadIdx.x;
	const unsigned int gidx = idx * gridDim.z + blockIdx.z;

	if (gidx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform_i = poseToMatrix(state.d_xRot[i], state.d_xTrans[i]);		//TODO HERE
		float4x4 transform_j = poseToMatrix(state.d_xRot[j], state.d_xTrans[j]);
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
#endif
		float4x4 invTransform_i = transform_i.getInverse(); //TODO pre-compute the inverts

		float4x4 transform = invTransform_i * transform_j;

		//!!!debugging
		//if (!computeAngleDiff(transform, 0.5f)) return; //~30 degrees
		//!!!debugging

		// find correspondence
		float4 camPosSrcToTgt; unsigned int tgtIdx;
		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.depthIntrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrcToTgt, tgtIdx)) { //i tgt, j src
			atomicAdd(&state.d_denseCorrCounts[imPairIdx], 1.0f);
		} // found correspondence
	} // valid image pixel
}

__global__ void WeightDenseCorrespondences_Kernel(unsigned int N, SolverState state)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		// apply ln to weights
		float x = state.d_denseCorrCounts[idx];
		if (x > 0) {
			if (x < 400) state.d_denseCorrCounts[idx] = 0; //don't consider too small #corr //TODO PARAMS
			else {
				state.d_denseCorrCounts[idx] = 1.0f / min(logf(x), 9.0f); // natural log //TODO PARAMS
			}
		}
	}
}

template<bool useColor>
__global__ void BuildDenseSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	// image indices
	unsigned int i, j; // project from j to i
	unsigned int imPairIdx;
	if (parameters.useDenseDepthAllPairwise) {
		i = blockIdx.x; j = blockIdx.y; // all pairwise
		if (i >= j) return;
		imPairIdx = i * gridDim.x + j;
	}
	else {
		i = blockIdx.x; j = i + 1; // frame-to-frame
		imPairIdx = i;
	}
	if (input.d_validImages[i] == 0 || input.d_validImages[j] == 0) return;
	float imPairWeight = state.d_denseCorrCounts[imPairIdx];
	if (imPairWeight == 0.0f) return;

	const unsigned int idx = threadIdx.y * THREADS_PER_BLOCK_DENSE_DEPTH_X + threadIdx.x;
	const unsigned int srcIdx = idx * gridDim.z + blockIdx.z;

	if (srcIdx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform_i = poseToMatrix(state.d_xRot[i], state.d_xTrans[i]);		//TODO HERE
		float4x4 transform_j = poseToMatrix(state.d_xRot[j], state.d_xTrans[j]);
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
#endif
		float4x4 invTransform_i = transform_i.getInverse(); //TODO precompute this crap
		float4x4 invTransform_j = transform_j.getInverse(); //TODO precompute this crap

		float4x4 transform = invTransform_i * transform_j;

		// find correspondence
		float4 camPosSrcToTgt; unsigned int tgtIdx;
		if (findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.depthIntrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrcToTgt, tgtIdx)) { //i tgt, j src
			const float4 camPosTgt = input.d_cacheFrames[i].d_cameraposDownsampled[tgtIdx];
			const float4 normalTgt = input.d_cacheFrames[i].d_normalsDownsampled[tgtIdx];
			// point-to-plane residual
			float4 diff = camPosTgt - camPosSrcToTgt;
			float res = dot(diff, normalTgt);
			float weight = max(0.0f, 0.5f*((1.0f - length(diff) / parameters.denseDistThresh) + (1.0f - camPosTgt.z / parameters.denseDepthMax)));

			// point-to-plane jacobian
			const float4 camPosSrc = input.d_cacheFrames[j].d_cameraposDownsampled[srcIdx];
			matNxM<1, 6> jacobianBlockRow_i, jacobianBlockRow_j;
#ifdef USE_LIE_SPACE
			//if (i > 0) computeJacobianBlockRow_i(jacobianBlockRow_i, transform_i, transform_j, camPosSrc, normalTgt);
			if (i > 0) computeJacobianBlockRow_i(jacobianBlockRow_i, transform_i, invTransform_j, camPosSrc, normalTgt);
			if (j > 0) computeJacobianBlockRow_j(jacobianBlockRow_j, invTransform_i, transform_j, camPosSrc, normalTgt);
#else
			if (i > 0) computeJacobianBlockRow_i(jacobianBlockRow_i, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, normalTgt);
			if (j > 0) computeJacobianBlockRow_j(jacobianBlockRow_j, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, normalTgt);
#endif

			//!!!debugging
			const unsigned int x = srcIdx % input.denseDepthWidth; const unsigned int y = srcIdx / input.denseDepthWidth;
			if (i > 0 && (isnan(jacobianBlockRow_i(0)) || isnan(jacobianBlockRow_i(1)) || isnan(jacobianBlockRow_i(2)) || isnan(jacobianBlockRow_i(3)) || isnan(jacobianBlockRow_i(4)) || isnan(jacobianBlockRow_i(5))) ||
				j > 0 && (isnan(jacobianBlockRow_j(0)) || isnan(jacobianBlockRow_j(1)) || isnan(jacobianBlockRow_j(2)) || isnan(jacobianBlockRow_j(3)) || isnan(jacobianBlockRow_j(4)) || isnan(jacobianBlockRow_j(5))) ||
				isnan(res) || isnan(weight)) {
				printf("ERROR NaN (%d,%d,%d,%d) %f %f | (%f %f %f %f %f %f) (%f %f %f %f %f %f)\n", i, j, x, y, res, weight,
					jacobianBlockRow_i(0), jacobianBlockRow_i(1), jacobianBlockRow_i(2),
					jacobianBlockRow_i(3), jacobianBlockRow_i(4), jacobianBlockRow_i(5),
					jacobianBlockRow_j(0), jacobianBlockRow_j(1), jacobianBlockRow_j(2),
					jacobianBlockRow_j(3), jacobianBlockRow_j(4), jacobianBlockRow_j(5));
			}
			//!!!debugging

			addToLocalSystem(state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
				jacobianBlockRow_i, jacobianBlockRow_j, i, j, res, parameters.weightDenseDepth * weight * imPairWeight);

			// color term
			if (useColor) {
				const float2 intensityDerivTgt = input.d_cacheFrames[i].d_intensityDerivsDownsampled[tgtIdx];
				float diffIntensity = input.d_cacheFrames[i].d_intensityDownsampled[tgtIdx] - input.d_cacheFrames[j].d_intensityDownsampled[srcIdx];
				if (intensityDerivTgt.x != MINF && abs(diffIntensity) < parameters.denseColorThresh && length(intensityDerivTgt) > parameters.denseColorGradientMin) {
					//if (i > 0) computeJacobianBlockIntensityRow_i(jacobianBlockRow_i, input.colorFocalLength, transform_i, transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (i > 0) computeJacobianBlockIntensityRow_i(jacobianBlockRow_i, input.colorFocalLength, transform_i, invTransform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (j > 0) computeJacobianBlockIntensityRow_j(jacobianBlockRow_j, input.colorFocalLength, invTransform_i, transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					//weight = max(0.0f, 1.0f - abs(diffIntensity) / parameters.denseColorThresh);
					weight = 1.0f;

					addToLocalSystem(state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6,
						jacobianBlockRow_i, jacobianBlockRow_j, i, j, diffIntensity, parameters.weightDenseColor * weight * imPairWeight);

					if (i > 0 && (isnan(jacobianBlockRow_i(0)) || isnan(jacobianBlockRow_i(1)) || isnan(jacobianBlockRow_i(2)) || isnan(jacobianBlockRow_i(3)) || isnan(jacobianBlockRow_i(4)) || isnan(jacobianBlockRow_i(5))) ||
						j > 0 && (isnan(jacobianBlockRow_j(0)) || isnan(jacobianBlockRow_j(1)) || isnan(jacobianBlockRow_j(2)) || isnan(jacobianBlockRow_j(3)) || isnan(jacobianBlockRow_j(4)) || isnan(jacobianBlockRow_j(5))) ||
						isnan(diffIntensity) || isnan(weight)) {
						printf("ERROR NaN color (%d,%d,%d,%d) %f %f | (%f %f %f %f %f %f) (%f %f %f %f %f %f)\n", i, j, x, y, diffIntensity, weight,
							jacobianBlockRow_i(0), jacobianBlockRow_i(1), jacobianBlockRow_i(2),
							jacobianBlockRow_i(3), jacobianBlockRow_i(4), jacobianBlockRow_i(5),
							jacobianBlockRow_j(0), jacobianBlockRow_j(1), jacobianBlockRow_j(2),
							jacobianBlockRow_j(3), jacobianBlockRow_j(4), jacobianBlockRow_j(5));
					}
#ifdef PRINT_RESIDUALS_DENSE
					atomicAdd(state.d_sumResidualColor, parameters.weightDenseColor * weight * imPairWeight * res * res);
					atomicAdd(state.d_corrCountColor, 1);
#endif
				}
			}
#ifdef PRINT_RESIDUALS_DENSE
			atomicAdd(state.d_sumResidual, parameters.weightDenseDepth * weight * imPairWeight * res * res);
			atomicAdd(state.d_corrCount, 1);
#endif
		} // found correspondence
	} // valid image pixel
}

extern "C"
void BuildDenseSystem(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;

	const int threadsPerBlock = THREADS_PER_BLOCK_DENSE_DEPTH_X * THREADS_PER_BLOCK_DENSE_DEPTH_Y;
	const int reductionGlobal = (input.denseDepthWidth*input.denseDepthHeight + threadsPerBlock - 1) / threadsPerBlock;
	const int sizeJtr = 6 * N;
	const int sizeJtJ = sizeJtr * sizeJtr;

	dim3 grid;
	if (parameters.useDenseDepthAllPairwise) grid = dim3(N, N, reductionGlobal); // pairwise
	else grid = dim3(N - 1, 1, reductionGlobal); // for frame-to-frame
	dim3 block(THREADS_PER_BLOCK_DENSE_DEPTH_X, THREADS_PER_BLOCK_DENSE_DEPTH_Y);

	if (timer) timer->startEvent("BuildDenseDepthSystem");

	//!!!debugging
#ifdef PRINT_RESIDUALS_DENSE
	cutilSafeCall(cudaMemset(state.d_corrCount, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidual, 0, sizeof(float)));
	cutilSafeCall(cudaMemset(state.d_corrCountColor, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidualColor, 0, sizeof(float)));
#endif
	//!!!debugging

	//TODO move this call into parameters.useDense and template the JTF computation with sparse/dense
	const unsigned int maxDenseImPairs = input.numberOfImages * input.numberOfImages;
	cutilSafeCall(cudaMemset(state.d_denseCorrCounts, 0, sizeof(float) * maxDenseImPairs));
	cutilSafeCall(cudaMemset(state.d_denseJtJ, 0, sizeof(float) * sizeJtJ)); //TODO check if necessary
	cutilSafeCall(cudaMemset(state.d_denseJtr, 0, sizeof(float) * sizeJtr));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif		

	if (parameters.useDense) {
		FindDenseCorrespondences_Kernel << <grid, block >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		//!!!DEBUGGING //remember the delete!
		//float* denseCorrCounts = new float[maxDenseImPairs];
		//cutilSafeCall(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts, sizeof(float)*maxDenseImPairs, cudaMemcpyDeviceToHost));
		//unsigned int totalCount = 0;
		//for (unsigned int i = 0; i < maxDenseImPairs; i++) { totalCount += (unsigned int)denseCorrCounts[i]; }
		//printf("total count = %d\n", totalCount);
		//!!!DEBUGGING
		int wgrid = (maxDenseImPairs + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
		WeightDenseCorrespondences_Kernel << < wgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >(maxDenseImPairs, state);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		//!!!DEBUGGING
		//cutilSafeCall(cudaMemcpy(denseCorrCounts, state.d_denseCorrCounts, sizeof(float)*maxDenseImPairs, cudaMemcpyDeviceToHost));
		//if (denseCorrCounts) delete[] denseCorrCounts;
		//!!!DEBUGGING

		if (parameters.weightDenseColor > 0.0f)
			BuildDenseSystem_Kernel<true> << <grid, block >> >(input, state, parameters);
		else
			BuildDenseSystem_Kernel<false> << <grid, block >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif

		//!!!debugging
		bool debugPrint = false;
		float* h_JtJ = NULL;
		float* h_Jtr = NULL;
		if (debugPrint) {
			h_JtJ = new float[sizeJtJ];
			h_Jtr = new float[sizeJtr];
			cutilSafeCall(cudaMemcpy(h_JtJ, state.d_denseJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_Jtr, state.d_denseJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
			//printf("JtJ:\n");
			//for (unsigned int i = 0; i < 6 * N; i++) {
			//	for (unsigned int j = 0; j < 6 * N; j++)
			//		printf(" %f,", h_JtJ[j * 6 * N + i]);
			//	printf("\n");
			//}
			printf("Jtr:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				printf(" %f,", h_Jtr[i]);
			}
			printf("\n");
		}
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
		//if (debugPrint) {
		//	cutilSafeCall(cudaMemcpy(h_JtJ, state.d_denseJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
		//	cutilSafeCall(cudaMemcpy(h_Jtr, state.d_denseJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
		//	printf("JtJ:\n");
		//	for (unsigned int i = 0; i < 6 * N; i++) {
		//		for (unsigned int j = 0; j < 6 * N; j++)
		//			printf(" %f,", h_JtJ[j * 6 * N + i]);
		//		printf("\n");
		//	}
		//	printf("Jtr:\n");
		//	for (unsigned int i = 0; i < 6 * N; i++) {
		//		printf(" %f,", h_Jtr[i]);
		//	}
		//	printf("\n\n");
		//	if (h_JtJ) delete[] h_JtJ;
		//	if (h_Jtr) delete[] h_Jtr;
		//}
		//!!!debugging
	}
	if (timer) timer->endEvent();
}

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
		float residual = evalAbsResidualDeviceFloat3(corrIdx, componentIdx, input, state, parameters);

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
	EvalMaxResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

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
		residual = evalFDevice(x, input, state, parameters, x == 0);
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
		float residual = evalAbsResidualDeviceFloat3(corrIdx, componentIdx, input, state, parameters);

		if (residual > parameters.verifyOptDistThresh)
			atomicAdd(state.d_countHighResidual, 1);
	}
}

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences * 3; // Number of correspondences (*3 per xyz)
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

	//!!!DEBUGGING //remember to uncomment the delete...
	//float3* rRot = new float3[input.numberOfImages]; // -jtf
	//float3* rTrans = new float3[input.numberOfImages];
	//!!!DEBUGGING

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

	//cutilSafeCall(cudaMemcpy(rRot, state.d_rRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(rTrans, state.d_rTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	////for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rRot[i].x)) { printf("NaN in jtr rRot %d\n", i); getchar(); } }
	////for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rTrans[i].x)) { printf("NaN in jtr rTrans %d\n", i); getchar(); } }
	//cutilSafeCall(cudaMemcpy(rRot, state.d_pRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(rTrans, state.d_pTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	////for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rRot[i].x)) { printf("NaN in jtr pRot %d\n", i); getchar(); } }
	////for (unsigned int i = 1; i < input.numberOfImages; i++) { if (isnan(rTrans[i].x)) { printf("NaN in jtr pTrans %d\n", i); getchar(); } }

	if (timer) timer->startEvent("Init2");
	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(N, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();

	//if (rRot) delete[] rRot;
	//if (rTrans) delete[] rTrans;
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

//TODO MAKE EFFICIENT
__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans); // A x p_k  => J^T x J x p_k 

		state.d_Ap_XRot[x] += rot;
		state.d_Ap_XTrans[x] += trans;
	}
}
//__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters)
//{
//	const unsigned int N = input.numberOfImages;							// Number of block variables
//	const unsigned int x = blockIdx.x;
//
//	//float d = 0.0f;
//	if (x > 0 && x < N)
//	{
//		const unsigned int lane = threadIdx.x % WARP_SIZE;
//
//		float3 rot, trans;
//		applyJTJDenseDevice(x, input, state, parameters, rot, trans, threadIdx.x);			// A x p_k  => J^T x J x p_k 
//
//		if (lane == 0)
//		{
//			atomicAdd(&state.d_Ap_XRot[x].x, rot.x);
//			atomicAdd(&state.d_Ap_XRot[x].y, rot.y);
//			atomicAdd(&state.d_Ap_XRot[x].z, rot.z);
//
//			atomicAdd(&state.d_Ap_XTrans[x].x, trans.x);
//			atomicAdd(&state.d_Ap_XTrans[x].y, trans.y);
//			atomicAdd(&state.d_Ap_XTrans[x].z, trans.z);
//		}
//	}
//}

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
#ifdef USE_LIE_SPACE //TODO just keep that matrix transforms around
			float3 rot, trans;
			computeLieUpdate(state.d_deltaRot[x], state.d_deltaTrans[x], state.d_xRot[x], state.d_xTrans[x], rot, trans);
			state.d_xRot[x] = rot;
			state.d_xTrans[x] = trans;
#else
			state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
			state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
#endif
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

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	// sparse part
	if (parameters.weightSparse > 0.0f) {
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
	if (parameters.useDense) {
		//PCGStep_Kernel_Dense << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
		PCGStep_Kernel_Dense << < N, 1 >> >(input, state, parameters); //TODO fix this part
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
	//!!!debugging
	//float3* Ap_Rot = new float3[input.numberOfImages];
	//float3* Ap_Trans = new float3[input.numberOfImages];
	//cutilSafeCall(cudaMemcpy(Ap_Rot, state.d_Ap_XRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(Ap_Trans, state.d_Ap_XTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//if (isnan(Ap_Rot[1].x)) { printf("NaN in Ap rot\n"); getchar(); }
	//if (isnan(Ap_Trans[1].x)) { printf("NaN in Ap trans\n"); getchar(); }
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

	//!!!DEBUGGING
#ifdef PRINT_RESIDUALS_SPARSE
	if (parameters.weightSparse > 0) {
		float initialResidual = EvalResidual(input, state, parameters, timer);
		printf("initial sparse = %f*%f = %f\n", parameters.weightSparse, initialResidual / parameters.weightSparse, initialResidual);
	}
#endif
	//float3* xRot = new float3[input.numberOfImages];	//remember the delete!
	//float3* xTrans = new float3[input.numberOfImages];
	//!!!DEBUGGING

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		parameters.weightSparse = input.weightsSparse[nIter];
		parameters.weightDenseDepth = input.weightsDenseDepth[nIter];
		parameters.weightDenseColor = input.weightsDenseColor[nIter];
		parameters.useDense = (parameters.weightDenseDepth > 0 || parameters.weightDenseColor > 0);

		BuildDenseSystem(input, state, parameters, timer);
		Initialization(input, state, parameters, timer);

		//float linearResidual = EvalLinearRes(input, state, parameters);
		//linConvergenceAnalysis[idx++] = linearResidual;

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
		{
			//!!!debugging
			//if (linIter == parameters.nLinIterations - 1) {
			//	cutilSafeCall(cudaMemcpy(xRot, state.d_deltaRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
			//	cutilSafeCall(cudaMemcpy(xTrans, state.d_deltaTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
			//	//char buffer [50];
			//	//sprintf (buffer, "debug/delta-%d.txt", nIter);
			//	//FILE* pf = fopen(buffer, "w");
			//	//if (pf == NULL) { printf("failed to open %s for writing\n", buffer); getchar(); }
			//	//fprintf(pf, "delta rot:\n");
			//	//for (unsigned int i = 1; i < input.numberOfImages; i++) { 
			//	//	if (isnan(xRot[i].x) || isnan(xRot[i].y) || isnan(xRot[i].z)) { printf("NaN in input delta rot %d\n", i); getchar(); }
			//	//	fprintf(pf, "%d: %f %f %f\n", i, xRot[i].x, xRot[i].y, xRot[i].z);
			//	//}
			//	//fprintf(pf, "delta trans:\n");
			//	//for (unsigned int i = 1; i < input.numberOfImages; i++) { 
			//	//	if (isnan(xTrans[i].x) || isnan(xTrans[i].y) || isnan(xTrans[i].z)) { printf("NaN in input delta rot %d\n", i); getchar(); }
			//	//	fprintf(pf, "%d: %f %f %f\n", i, xTrans[i].x, xTrans[i].y, xTrans[i].z);
			//	//}
			//	//fclose(pf);
			//	int a = 5;
			//}
			//!!!debugging
			PCGIteration(input, state, parameters, linIter == parameters.nLinIterations - 1, timer);

			//linearResidual = EvalLinearRes(input, state, parameters);
			//linConvergenceAnalysis[idx++] = linearResidual;
		}
		//!!!debugging
		//cutilSafeCall(cudaMemcpy(xRot, state.d_xRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//cutilSafeCall(cudaMemcpy(xTrans, state.d_xTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//!!!debugging

		//!!!DEBUGGING
#ifdef PRINT_RESIDUALS_SPARSE
		if (parameters.weightSparse > 0) {
			float residual = EvalResidual(input, state, parameters, timer);
			printf("[niter %d] weight * sparse = %f*%f = %f\t[#corr = %d]\n", nIter, parameters.weightSparse, residual / parameters.weightSparse, residual, input.numberOfCorrespondences);
		}
#endif
		//!!!DEBUGGING

		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
			//printf("[niter %d] %f\n", nIter, residual);
		}
	}
	//!!!debugging
	//if (xRot) delete[] xRot;
	//if (xTrans) delete[] xTrans;
	//!!!debugging
}

////////////////////////////////////////////////////////////////////
// build variables to correspondences lookup
////////////////////////////////////////////////////////////////////

__global__ void BuildVariablesToCorrespondencesTableDevice(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow)
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
