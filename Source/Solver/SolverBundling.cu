#include <iostream>

#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "SolverBundlingUtil.h"
#include "SolverBundlingEquations.h"
#include "../../SiftGPU/CUDATimer.h"

#include <conio.h>


#define THREADS_PER_BLOCK_DENSE_DEPTH_X 32
#define THREADS_PER_BLOCK_DENSE_DEPTH_Y 4 
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64



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

/////////////////////////////////////////////////////////////////////////
// Sparse Term (materialize jtj/jtr)
/////////////////////////////////////////////////////////////////////////
__global__ void BuildSparseSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < input.numberOfCorrespondences) {
		const EntryJ corr = input.d_correspondences[idx];
		if (corr.isValid()) {
			const float3x3 TI = evalRMat(state.d_xRot[corr.imgIdx_i]);
			const float3x3 TJ = evalRMat(state.d_xRot[corr.imgIdx_j]);

			const float3 angles_i = state.d_xRot[corr.imgIdx_i]; // get angles
			const float3 angles_j = state.d_xRot[corr.imgIdx_j]; // get angles

			const float3 dist = (TI * corr.pos_i + state.d_xTrans[corr.imgIdx_i]) - (TJ * corr.pos_j + state.d_xTrans[corr.imgIdx_j]);

			float3 jacobianBlockRow_i[6];
			float3 jacobianBlockRow_j[6];
			if (corr.imgIdx_i > 0) {
				jacobianBlockRow_i[0] = evalR_dAlpha(angles_i) * corr.pos_i; // alpha
				jacobianBlockRow_i[1] = evalR_dBeta(angles_i) * corr.pos_i; // beta
				jacobianBlockRow_i[2] = evalR_dGamma(angles_i) * corr.pos_i; // gamma
				jacobianBlockRow_i[3] = make_float3(1.0f, 0.0f, 0.0f); // x
				jacobianBlockRow_i[4] = make_float3(0.0f, 1.0f, 0.0f); // y
				jacobianBlockRow_i[5] = make_float3(0.0f, 0.0f, 1.0f); // z
			}
			if (corr.imgIdx_j > 0) {
				jacobianBlockRow_j[0] = -(evalR_dAlpha(angles_j) * corr.pos_j); // alpha
				jacobianBlockRow_j[1] = -(evalR_dBeta(angles_j) * corr.pos_j); // beta
				jacobianBlockRow_j[2] = -(evalR_dGamma(angles_j) * corr.pos_j); // gamma
				jacobianBlockRow_j[3] = make_float3(1.0f, 0.0f, 0.0f); // x
				jacobianBlockRow_j[4] = make_float3(0.0f, 1.0f, 0.0f); // y
				jacobianBlockRow_j[5] = make_float3(0.0f, 0.0f, 1.0f); // z
			}
			
			addToLocalSystemSparse(state.d_sparseJtJ, state.d_sparseJtr, 6*input.numberOfImages,
				jacobianBlockRow_i, jacobianBlockRow_j, corr.imgIdx_i, corr.imgIdx_j, dist, parameters.weightSparse);

			//!!!DEBUGGING
			atomicAdd(state.d_sumResidual, parameters.weightSparse * dot(dist, dist));
			//!!!DEBUGGING
		} //valid correspondence
	}
}

extern "C"
void BuildSparseSystem(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int Ncorr = input.numberOfCorrespondences;

	const int grid =(Ncorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int sizeJtr = 6 * input.numberOfImages;
	const int sizeJtJ = sizeJtr * sizeJtr;

	if (timer) timer->startEvent("BuildSparseSystem");

	//!!!debugging
	cutilSafeCall(cudaMemset(state.d_sumResidual, 0, sizeof(float)));
	cutilSafeCall(cudaMemset(state.d_sparseJtJ, 0, sizeof(float) * sizeJtJ)); //TODO only need one for sparse and dense
	cutilSafeCall(cudaMemset(state.d_sparseJtr, 0, sizeof(float) * sizeJtr));
	//!!!debugging
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif		

	if (parameters.weightSparse > 0.0f) {
		BuildSparseSystem_Kernel << <grid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		//!!!debugging
		bool debugPrint = false;
		float* h_JtJ = NULL;
		float* h_Jtr = NULL;
		if (debugPrint) {
			const unsigned int N = input.numberOfImages;
			h_JtJ = new float[sizeJtJ];
			h_Jtr = new float[sizeJtr];
			cutilSafeCall(cudaMemcpy(h_JtJ, state.d_sparseJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_Jtr, state.d_sparseJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
			printf("sparse JtJ:\n");
			for (unsigned int i = 0; i < 6 * N; i++) { //row
				for (unsigned int j = 0; j < 6 * N; j++) //col
					printf(" %f,", h_JtJ[i * 6 * N + j]);
				printf("\n");
			}
			printf("sparse Jtr:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				printf(" %f,", h_Jtr[i]);
			}
			printf("\n");
		}
		float sumResidual;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost));
		printf("sparse res = %f\n", sumResidual);

		const unsigned int flipgrid = (sizeJtJ + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
		FlipJtJ_Kernel << <flipgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >(sizeJtJ, sizeJtr, state.d_sparseJtJ);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif	
		if (debugPrint) {
			const unsigned int N = input.numberOfImages;
			cutilSafeCall(cudaMemcpy(h_JtJ, state.d_sparseJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_Jtr, state.d_sparseJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
			printf("sparse JtJ:\n");
			for (unsigned int i = 0; i < 6 * N; i++) { //row
				for (unsigned int j = 0; j < 6 * N; j++) //col
					printf(" %f,", h_JtJ[i * 6 * N + j]);
				printf("\n");
			}
			printf("sparse Jtr:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				printf(" %f,", h_Jtr[i]);
			}
			printf("\n\n");
			if (h_JtJ) delete[] h_JtJ;
			if (h_Jtr) delete[] h_Jtr;
		}
		//!!!debugging
	}
	if (timer) timer->endEvent();
}

/////////////////////////////////////////////////////////////////////////
// Dense Depth Term
/////////////////////////////////////////////////////////////////////////

__device__ bool findDenseDepthCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, /*float colorThresh,*/ const float4x4& transform, const float4x4& intrinsics,
	const float4* tgtCamPos, const float4* tgtNormals, //const uchar4* tgtColor,
	const float4* srcCamPos, const float4* srcNormals, //const uchar4* srcColor,
	float depthMin, float depthMax,
	float4& camPosSrcToTgt, float4& camPosTgt, float4& normalTgt
	, bool debugPrint)
{
	//!!!DEBUGGING
	//bool debugPrint = false;
	//!!!DEBUGGING
	const float4& cposj = srcCamPos[idx];
	if (debugPrint) printf("cam pos j = %f %f %f\n", cposj.x, cposj.y, cposj.z);
	if (cposj.z > depthMin && cposj.z < depthMax) {
		float4 nrmj = srcNormals[idx];
		if (debugPrint) printf("normal j = %f %f %f\n", nrmj.x, nrmj.y, nrmj.z);
		if (nrmj.x != MINF) {
			nrmj = transform * nrmj;
			camPosSrcToTgt = transform * cposj;
			float3 proj = intrinsics * make_float3(camPosSrcToTgt.x, camPosSrcToTgt.y, camPosSrcToTgt.z);
			int2 screenPos = make_int2((int)roundf(proj.x / proj.z), (int)roundf(proj.y / proj.z));
			//const uchar4& colorj = srcColor[idx];
			if (debugPrint) {
				printf("cam pos j2i = %f %f %f\n", camPosSrcToTgt.x, camPosSrcToTgt.y, camPosSrcToTgt.z);
				printf("proj %f %f %f -> %f %f\n", proj.x, proj.y, proj.z, proj.x / proj.z, proj.y / proj.z);
				printf("screen pos = %d %d\n", screenPos.x, screenPos.y);
			}
			if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < (int)imageWidth && screenPos.y < (int)imageHeight) {
				camPosTgt = tgtCamPos[screenPos.y * imageWidth + screenPos.x];
				if (debugPrint) printf("cam pos i = %f %f %f\n", camPosTgt.x, camPosTgt.y, camPosTgt.z);
				if (camPosTgt.z > depthMin && camPosTgt.z < depthMax) {
					normalTgt = tgtNormals[screenPos.y * imageWidth + screenPos.x];
					if (debugPrint) printf("normal i = %f %f %f\n", normalTgt.x, normalTgt.y, normalTgt.z);
					if (normalTgt.x != MINF) {
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt);
						//const uchar4& colori = tgtColor[idx];
						//float dColor = length(make_float3(colorj.x - colori.x, colorj.y - colori.y, colorj.z - colori.z) / 255.0f);
						//if (debugPrint) printf("dist = %f, dnormal = %f, colorDist = %f\n", dist, dNormal, dColor);
						if (debugPrint) printf("dist = %f, dnormal = %f\n", dist, dNormal);
						if (dNormal >= normalThresh && dist <= distThresh /*&& dColor <= colorThresh*/) {
							return true;
						}
					}
				}
			} // valid projection
		} // valid src normal
	} // valid src camera position
	return false;
}
__global__ void BuildDenseDepthSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	// image indices
	// all pairwise
	const unsigned int i = blockIdx.x; const unsigned int j = blockIdx.y; // project from j to i
	if (i >= j) return;
	//// frame-to-frame
	//const unsigned int i = blockIdx.x; const unsigned int j = i + 1; // project from j to i

	const unsigned int idx = threadIdx.y * THREADS_PER_BLOCK_DENSE_DEPTH_X + threadIdx.x;
	const unsigned int gidx = idx * gridDim.z + blockIdx.z;


	if (gidx < (input.denseDepthWidth * input.denseDepthHeight)) {
		//!!!DEBUGGING
		const float3& anglesi = state.d_xRot[i]; const float3& anglesj = state.d_xRot[j];
		const float3& transi = state.d_xTrans[i]; const float3& transj = state.d_xTrans[j];
		if (isnan(anglesi.x) || isnan(anglesi.y) || isnan(anglesi.z) ||
			isnan(transi.x) || isnan(transi.y) || isnan(transi.z) ||
			isnan(anglesj.x) || isnan(anglesj.y) || isnan(anglesj.z) ||
			isnan(transj.x) || isnan(transj.y) || isnan(transj.z)) {
			printf("ERROR NaN transforms %d %d: (%f %f %f %f %f %f) (%f %f %f %f %f %f)\n", i, j,
				anglesi.x, anglesi.y, anglesi.z, transi.x, transi.y, transi.z,
				anglesj.x, anglesj.y, anglesj.z, transj.x, transj.y, transj.z);
		}
		//!!!DEBUGGING

		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse(); //TODO unncessary invert for pairwise?

		float4x4 transform = invTransform_i * transform_j;

		//!!!debugging
		const unsigned int x = gidx % input.denseDepthWidth; const unsigned int y = gidx / input.denseDepthWidth;
		float4 camPosSrcToTgt = make_float4(MINF, MINF, MINF, MINF), camPosTgt = make_float4(MINF, MINF, MINF, MINF), normalTgt = make_float4(MINF, MINF, MINF, MINF);
		//!!!debugging

		// find correspondence
		//float4 camPosSrcToTgt, camPosTgt, normalTgt;
		if (findDenseDepthCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDepthDistThresh, parameters.denseDepthNormalThresh, /*parameters.denseDepthColorThresh,*/ transform, input.depthIntrinsics,
			input.d_depthFrames[i].d_cameraposDownsampled, input.d_depthFrames[i].d_normalsDownsampled, //input.d_depthFrames[i].d_colorDownsampled,//target
			input.d_depthFrames[j].d_cameraposDownsampled, input.d_depthFrames[j].d_normalsDownsampled, //input.d_depthFrames[j].d_colorDownsampled,//source
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrcToTgt, camPosTgt, normalTgt
			//,(i == 0 && j == 1 && x == 47 && y == 53)
			, false
			)) { //i tgt, j src
			// residual
			float4 diff = camPosTgt - camPosSrcToTgt;
			float res = dot(diff, normalTgt);

			// jacobian
			const float4& camPosSrc = input.d_depthFrames[j].d_cameraposDownsampled[gidx];
			matNxM<1, 6> jacobianBlockRow_i, jacobianBlockRow_j;
			if (i > 0) computeJacobianBlockRow_i(jacobianBlockRow_i, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, normalTgt);
			if (j > 0) computeJacobianBlockRow_j(jacobianBlockRow_j, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, normalTgt);
			float weight = max(0.0f, 0.5f*((1.0f - length(diff) / parameters.denseDepthDistThresh) + (1.0f - camPosTgt.z / parameters.denseDepthMax)));

			addToLocalSystem(state.d_depthJtJ, state.d_depthJtr, input.numberOfImages * 6,
				jacobianBlockRow_i, jacobianBlockRow_j, i, j, res, parameters.weightDenseDepth * weight
				, state.d_sumResidual);
			atomicAdd(state.d_corrCount, 1);

			//!!!debugging
			//if (i == 0 && j == 1 && x == 47 && y == 53) {
			if (i > 0 && (isnan(jacobianBlockRow_i(0)) || isnan(jacobianBlockRow_i(1)) || isnan(jacobianBlockRow_i(2)) || isnan(jacobianBlockRow_i(3)) || isnan(jacobianBlockRow_i(4)) || isnan(jacobianBlockRow_i(5))) ||
				j > 0 && (isnan(jacobianBlockRow_j(0)) || isnan(jacobianBlockRow_j(1)) || isnan(jacobianBlockRow_j(2)) || isnan(jacobianBlockRow_j(3)) || isnan(jacobianBlockRow_j(4)) || isnan(jacobianBlockRow_j(5))) ||
				isnan(res) || isnan(weight)) {
				printf("ERROR NaN (%d,%d,%d,%d) %f %f | (%f %f %f %f %f %f) (%f %f %f %f %f %f)\n", i, j, x, y, res, weight,
					jacobianBlockRow_i(0), jacobianBlockRow_i(1), jacobianBlockRow_i(2),
					jacobianBlockRow_i(3), jacobianBlockRow_i(4), jacobianBlockRow_i(5),
					jacobianBlockRow_j(0), jacobianBlockRow_j(1), jacobianBlockRow_j(2),
					jacobianBlockRow_j(3), jacobianBlockRow_j(4), jacobianBlockRow_j(5));
			//	printf("-----------\n");
			//	printf("images (%d, %d) at (%d, %d)\n", i, j, x, y);
			//	//printf("transform i:\n"); transform_i.print();
			//	//printf("inv transform i:\n"); invTransform_i.print();
			//	//printf("transform j:\n"); transform_j.print();
			//	//printf("transform:\n"); transform.print();
			//	printf("cam pos src: %f %f %f\n", camPosSrc.x, camPosSrc.y, camPosSrc.z);
			//	printf("cam pos src to tgt: %f %f %f\n", camPosSrcToTgt.x, camPosSrcToTgt.y, camPosSrcToTgt.z);
			//	printf("cam pos tgt: %f %f %f\n", camPosTgt.x, camPosTgt.y, camPosTgt.z);
			//	printf("normal tgt: %f %f %f\n", normalTgt.x, normalTgt.y, normalTgt.z);
			//	printf("diff = %f %f %f %f\n", diff.x, diff.y, diff.z, diff.w);
			//	printf("res = %f\n", res);
			//	printf("weight = %f\n", parameters.weightDenseDepth * weight);
			//	printf("jac i %f %f %f %f %f %f\n", jacobianBlockRow_i(0), jacobianBlockRow_i(1), jacobianBlockRow_i(2),
			//		jacobianBlockRow_i(3), jacobianBlockRow_i(4), jacobianBlockRow_i(5));
			//	printf("jac j %f %f %f %f %f %f\n", jacobianBlockRow_j(0), jacobianBlockRow_j(1), jacobianBlockRow_j(2),
			//		jacobianBlockRow_j(3), jacobianBlockRow_j(4), jacobianBlockRow_j(5));
			}
			//!!!debugging
		} // found correspondence
	} // valid image pixel
}

void BuildDenseDepthSystem(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;

	const int threadsPerBlock = THREADS_PER_BLOCK_DENSE_DEPTH_X * THREADS_PER_BLOCK_DENSE_DEPTH_Y;
	const int reductionGlobal = (input.denseDepthWidth*input.denseDepthHeight + threadsPerBlock - 1) / threadsPerBlock;
	const int sizeJtr = 6 * N;
	const int sizeJtJ = sizeJtr * sizeJtr;

	dim3 grid(N, N, reductionGlobal); // for all pairwise
	//dim3 grid(N - 1, 1, reductionGlobal); // for frame-to-frame
	dim3 block(THREADS_PER_BLOCK_DENSE_DEPTH_X, THREADS_PER_BLOCK_DENSE_DEPTH_Y);

	if (timer) timer->startEvent("BuildDenseDepthSystem");

	//!!!debugging
	cutilSafeCall(cudaMemset(state.d_sumResidual, 0, sizeof(float)));
	cutilSafeCall(cudaMemset(state.d_corrCount, 0, sizeof(int)));
	//!!!debugging

	cutilSafeCall(cudaMemset(state.d_depthJtJ, 0, sizeof(float) * sizeJtJ)); //TODO check if necessary
	cutilSafeCall(cudaMemset(state.d_depthJtr, 0, sizeof(float) * sizeJtr));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif		

	if (parameters.weightDenseDepth > 0.0f) {
		BuildDenseDepthSystem_Kernel << <grid, block >> >(input, state, parameters);
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
			cutilSafeCall(cudaMemcpy(h_JtJ, state.d_depthJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_Jtr, state.d_depthJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
			printf("JtJ:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				for (unsigned int j = 0; j < 6 * N; j++)
					printf(" %f,", h_JtJ[j * 6 * N + i]);
				printf("\n");
			}
			printf("Jtr:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				printf(" %f,", h_Jtr[i]);
			}
			printf("\n");
		}
		int corrCount;
		cutilSafeCall(cudaMemcpy(&corrCount, state.d_corrCount, sizeof(int), cudaMemcpyDeviceToHost));
		float sumResidual;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost));
		printf("dense res = %f (#corr = %d)\n", sumResidual, corrCount);

		const unsigned int flipgrid = (sizeJtJ + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
		FlipJtJ_Kernel << <flipgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP >> >(sizeJtJ, sizeJtr, state.d_depthJtJ);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif	
		if (debugPrint) {
			cutilSafeCall(cudaMemcpy(h_JtJ, state.d_depthJtJ, sizeof(float) * sizeJtJ, cudaMemcpyDeviceToHost));
			cutilSafeCall(cudaMemcpy(h_Jtr, state.d_depthJtr, sizeof(float) * sizeJtr, cudaMemcpyDeviceToHost));
			printf("JtJ:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				for (unsigned int j = 0; j < 6 * N; j++)
					printf(" %f,", h_JtJ[j * 6 * N + i]);
				printf("\n");
			}
			printf("Jtr:\n");
			for (unsigned int i = 0; i < 6 * N; i++) {
				printf(" %f,", h_Jtr[i]);
			}
			printf("\n\n");
			if (h_JtJ) delete[] h_JtJ;
			if (h_Jtr) delete[] h_Jtr;
		}
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
	//!!!DEBUGGING
	if (x == 0) {
		state.d_rRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_rTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_pRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_pTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
	}
	else if (x > 0 && x < N)
	//!!!DEBUGGING
	//if (x > 0 && x < N)
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

	//TODO uncomment
	//if (x > 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];				// store result for next kernel call

	//!!!DEBUGGING
	if (x == 0) {
		state.d_rDotzOld[x] = 0.0f;
	}
	else if (x > 0 && x < N) {
		state.d_rDotzOld[x] = state.d_scanAlpha[0];
	}
	//!!!DEBUGGING
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

	//!!!DEBUGGING
	float3* rRot = new float3[input.numberOfImages]; // -jtf
	float3* rTrans = new float3[input.numberOfImages];
	//float scanAlpha;
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

	cutilSafeCall(cudaMemcpy(rRot, state.d_rRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(rTrans, state.d_rTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	// print jtr
	printf("Jtr:\n");
	for (unsigned int i = 0; i < input.numberOfImages; i++) {
		const float3& rotPart = rRot[i]; const float3& transPart = rTrans[i];
		printf(" %f, %f, %f, %f, %f, %f,", rotPart.x, rotPart.y, rotPart.z, transPart.x, transPart.y, transPart.z);
	}
	printf("\n\n");
	//cutilSafeCall(cudaMemcpy(rRot, state.d_pRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(rTrans, state.d_pTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));

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

//TODO MAKE THESE EFFICIENT
__global__ void PCGStep_Kernel_Sparse(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDevice(x, state, state.d_sparseJtJ, input.numberOfImages, rot, trans);			// A x p_k  => J^T x J x p_k 

		state.d_Ap_XRot[x] += rot;
		state.d_Ap_XTrans[x] += trans;
	}
}
__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;							// Number of block variables
	const unsigned int x = blockIdx.x;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDevice(x, state, state.d_depthJtJ, input.numberOfImages, rot, trans); // A x p_k  => J^T x J x p_k 

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

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	// sparse part
	if (parameters.weightSparse > 0.0f) {
#ifdef PRECOMPUTE_SPARSE_JTJ
		//PCGStep_Kernel_Sparse << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
		PCGStep_Kernel_Sparse << < N, 1 >> >(input, state, parameters); //TODO fix this part
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
#else // applying JtJ on-the-fly
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
#endif
	}
	if (parameters.weightDenseDepth > 0.0f) {
		//PCGStep_Kernel_Dense << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
		PCGStep_Kernel_Dense << < N, 1 >> >(input, state, parameters); //TODO fix this part
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
	//float3* Ap_Rot = new float3[input.numberOfImages];
	//float3* Ap_Trans = new float3[input.numberOfImages];
	//cutilSafeCall(cudaMemcpy(Ap_Rot, state.d_Ap_XRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(Ap_Trans, state.d_Ap_XTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//if (Ap_Rot) delete[] Ap_Rot;
	//if (Ap_Trans) delete[] Ap_Trans;


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

	//!!!DEBUGGING
	//float3* xRot = new float3[input.numberOfImages];
	//float3* xTrans = new float3[input.numberOfImages];
	//cutilSafeCall(cudaMemcpy(xRot, state.d_xRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(xTrans, state.d_xTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
	if (parameters.weightSparse > 0) {
		float initialResidual = EvalResidual(input, state, parameters, timer);
		printf("initial sparse = %f\n", initialResidual);
	}
	//!!!DEBUGGING

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		parameters.weightDenseDepth = parameters.weightDenseDepthInit + nIter * parameters.weightDenseDepthLinFactor;
		BuildDenseDepthSystem(input, state, parameters, timer);
#ifdef PRECOMPUTE_SPARSE_JTJ
		BuildSparseSystem(input, state, parameters, timer);
#endif
		Initialization(input, state, parameters, timer);

		//float linearResidual = EvalLinearRes(input, state, parameters);
		//linConvergenceAnalysis[idx++] = linearResidual;

		//cutilSafeCall(cudaMemcpy(xRot, state.d_pRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//cutilSafeCall(cudaMemcpy(xTrans, state.d_pTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));

		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
		{
			//if (linIter == parameters.nLinIterations - 1) { //delta
			//	cutilSafeCall(cudaMemcpy(xRot, state.d_deltaRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
			//	cutilSafeCall(cudaMemcpy(xTrans, state.d_deltaTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
			//	int a = 5;
			//}

			PCGIteration(input, state, parameters, linIter == parameters.nLinIterations - 1, timer);

			//linearResidual = EvalLinearRes(input, state, parameters);
			//linConvergenceAnalysis[idx++] = linearResidual;
		}

		//ApplyLinearUpdate(input, state, parameters);	//this should be also done in the last PCGIteration

		//!!!DEBUGGING
		//cutilSafeCall(cudaMemcpy(xRot, state.d_xRot, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		//cutilSafeCall(cudaMemcpy(xTrans, state.d_xTrans, sizeof(float3)*input.numberOfImages, cudaMemcpyDeviceToHost));
		if (parameters.weightSparse > 0) {
			float residual = EvalResidual(input, state, parameters, timer);
			printf("[niter %d] sparse %f\n", nIter, residual);
		}
		//!!!DEBUGGING

		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
			//printf("[niter %d] %f\n", nIter, residual);
		}
	}

	//if (xRot) delete[] xRot;
	//if (xTrans) delete[] xTrans;
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
