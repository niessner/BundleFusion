#pragma once

#ifndef _SOLVER_DENSE_UTIL_
#define _SOLVER_DENSE_UTIL_

#include "GlobalDefines.h"
#include <cutil_inline.h>
#include <cutil_math.h>
#include "../SiftGPU/cuda_SimpleMatrixUtil.h"
#include "ICPUtil.h" //for the bilinear...
#include "../CUDACameraUtil.h"

//#include "SolverBundlingUtil.h"
//#include "SolverBundlingState.h"
//#include "SolverBundlingParameters.h"

////////////////////////////////////////
// build jtj/jtr
////////////////////////////////////////

//for pre-filter, no need for normal threshold
__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, const float4x4& transform, const float4& intrinsics,
	const float* tgtDepth, const float* srcDepth, float depthMin, float depthMax)
{
	unsigned int x = idx % imageWidth;		unsigned int y = idx / imageWidth;
	const float3 cposj = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, make_int2(x, y), srcDepth[idx]);
	if (cposj.z > depthMin && cposj.z < depthMax) {
		float3 camPosSrcToTgt = transform * cposj;
		float2 tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
		int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
		if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
			float3 camPosTgt = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtScreenPos, tgtDepth[tgtScreenPos.y * imageWidth + tgtScreenPos.x]);
			if (camPosTgt.z > depthMin && camPosTgt.z < depthMax) {
				if (length(camPosSrcToTgt - camPosTgt) <= distThresh) {
					return true;
				}
			}
		} // valid projection
	} // valid src camera position
	return false;
}

__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4& intrinsics,
	const float* tgtDepth, const float4* tgtNormals, const float* srcDepth, const float4* srcNormals,
	float depthMin, float depthMax)
{
	unsigned int x = idx % imageWidth;		unsigned int y = idx / imageWidth;
	const float3 cposj = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, make_int2(x, y), srcDepth[idx]);
	if (cposj.z > depthMin && cposj.z < depthMax) {
		float3 nrmj = make_float3(srcNormals[idx]);
		if (nrmj.x != MINF) {
			nrmj = transform.getFloat3x3() * nrmj;
			float3 camPosSrcToTgt = transform * cposj;
			float2 tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
			int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				float3 camPosTgt = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtScreenPos, tgtDepth[tgtScreenPos.y * imageWidth + tgtScreenPos.x]);
				if (camPosTgt.z > depthMin && camPosTgt.z < depthMax) {
					float3 normalTgt = make_float3(tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x]);
					//float3 normalTgt = make_float3(bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtNormals, imageWidth, imageHeight));
					if (normalTgt.x != MINF) {
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt);
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


//using camera positions
__device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4& intrinsics,
	const float4* tgtCamPos, const float4* tgtNormals, const float4* srcCamPos, const float4* srcNormals,
	float depthMin, float depthMax, float3& camPosSrc, float3& camPosSrcToTgt, float2& tgtScreenPosf, float3& camPosTgt, float3& normalTgt)
{
	const float4 cposj = srcCamPos[idx];
	if (cposj.z > depthMin && cposj.z < depthMax) {
		camPosSrc = make_float3(cposj.x, cposj.y, cposj.z);
		float4 nrmj = srcNormals[idx];
		if (nrmj.x != MINF) {
			nrmj = transform * nrmj;
			camPosSrcToTgt = transform * camPosSrc;
			tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
			int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				//camPosTgt = tgtCamPos[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
				float4 cposi = bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtCamPos, imageWidth, imageHeight);
				if (cposi.z > depthMin && cposi.z < depthMax) {
					camPosTgt = make_float3(cposi.x, cposi.y, cposi.z);
					//normalTgt = tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
					float4 nrmi = bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtNormals, imageWidth, imageHeight);
					if (nrmi.x != MINF) {
						normalTgt = make_float3(nrmi.x, nrmi.y, nrmi.z);
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, nrmi);
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
//using depth
__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4& intrinsics,
	const float* tgtDepth, const float4* tgtNormals, const float* srcDepth, const float4* srcNormals,
	float depthMin, float depthMax, float3& camPosSrc, float3& camPosSrcToTgt, float2& tgtScreenPosf, float3& camPosTgt, float3& normalTgt)
{
	unsigned int x = idx % imageWidth;		unsigned int y = idx / imageWidth;
	camPosSrc = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, make_int2(x, y), srcDepth[idx]);
	if (camPosSrc.z > depthMin && camPosSrc.z < depthMax) {
		float4 nrmj = srcNormals[idx];
		if (nrmj.x != MINF) {
			nrmj = transform * nrmj;
			camPosSrcToTgt = transform * camPosSrc;
			tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
			int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				//camPosTgt = tgtCamPos[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
				float depthTgt = bilinearInterpolationFloat(tgtScreenPosf.x, tgtScreenPosf.y, tgtDepth, imageWidth, imageHeight);
				if (depthTgt > depthMin && depthTgt < depthMax) {
					camPosTgt = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtScreenPos, depthTgt);
					//normalTgt = tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
					float4 normalTgt4 = bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtNormals, imageWidth, imageHeight);
					if (normalTgt4.x != MINF) {
						normalTgt = make_float3(normalTgt4.x, normalTgt4.y, normalTgt4.z);
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt4);
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

//-- uchar4 normals
__inline__ __device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4& intrinsics,
	const float* tgtDepth, const uchar4* tgtNormals, const float* srcDepth, const uchar4* srcNormals,
	float depthMin, float depthMax)
{
	unsigned int x = idx % imageWidth;		unsigned int y = idx / imageWidth;
	const float3 cposj = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, make_int2(x, y), srcDepth[idx]);
	if (cposj.z > depthMin && cposj.z < depthMax) {
		uchar4 nrmjUCHAR4 = srcNormals[idx];
		if (*(int*)(&nrmjUCHAR4) != 0) {
			float3 nrmj = make_float3(nrmjUCHAR4.x, nrmjUCHAR4.y, nrmjUCHAR4.z) / 255.0f * 2.0f - 1.0f;
			nrmj = transform.getFloat3x3() * nrmj;
			float3 camPosSrcToTgt = transform * cposj;
			float2 tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
			int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				float3 camPosTgt = depthToCamera(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, tgtScreenPos, tgtDepth[tgtScreenPos.y * imageWidth + tgtScreenPos.x]);
				if (camPosTgt.z > depthMin && camPosTgt.z < depthMax) {
					uchar4 nrmTgtUCHAR4 = tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
					if (*(int*)(&nrmTgtUCHAR4) != 0) {
						float3 normalTgt = make_float3(nrmTgtUCHAR4.x, nrmTgtUCHAR4.y, nrmTgtUCHAR4.z) / 255.0f * 2.0f - 1.0f;
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt);
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

//using camera positions
__device__ bool findDenseCorr(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, const float4x4& transform, const float4& intrinsics,
	const float4* tgtCamPos, const uchar4* tgtNormals, const float4* srcCamPos, const uchar4* srcNormals,
	float depthMin, float depthMax, float3& camPosSrc, float3& camPosSrcToTgt, float2& tgtScreenPosf, float3& camPosTgt, float3& normalTgt)
{
	const float4 cposj = srcCamPos[idx];
	if (cposj.z > depthMin && cposj.z < depthMax) {
		camPosSrc = make_float3(cposj.x, cposj.y, cposj.z);
		uchar4 nrmjUCHAR4 = srcNormals[idx];
		if (*(int*)(&nrmjUCHAR4) != 0) {
			float3 nrmj = make_float3(nrmjUCHAR4.x, nrmjUCHAR4.y, nrmjUCHAR4.z) / 255.0f * 2.0f - 1.0f;
			nrmj = transform * nrmj;
			camPosSrcToTgt = transform * camPosSrc;
			tgtScreenPosf = cameraToDepth(intrinsics.x, intrinsics.y, intrinsics.z, intrinsics.w, camPosSrcToTgt);
			int2 tgtScreenPos = make_int2((int)roundf(tgtScreenPosf.x), (int)roundf(tgtScreenPosf.y));
			if (tgtScreenPos.x >= 0 && tgtScreenPos.y >= 0 && tgtScreenPos.x < (int)imageWidth && tgtScreenPos.y < (int)imageHeight) {
				//camPosTgt = tgtCamPos[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
				float4 cposi = bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtCamPos, imageWidth, imageHeight);
				if (cposi.z > depthMin && cposi.z < depthMax) {
					camPosTgt = make_float3(cposi.x, cposi.y, cposi.z);
					uchar4 nrmTgtUCHAR4 = tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
					//normalTgt = tgtNormals[tgtScreenPos.y * imageWidth + tgtScreenPos.x];
					//float4 nrmi = bilinearInterpolationFloat4(tgtScreenPosf.x, tgtScreenPosf.y, tgtNormals, imageWidth, imageHeight);
					if (*(int*)(&nrmTgtUCHAR4) != 0) {
						float3 normalTgt = make_float3(nrmTgtUCHAR4.x, nrmTgtUCHAR4.y, nrmTgtUCHAR4.z) / 255.0f * 2.0f - 1.0f;
						float dist = length(camPosSrcToTgt - camPosTgt);
						float dNormal = dot(nrmj, normalTgt);
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

////////////////////////////////////////
// build jtj/jtr
////////////////////////////////////////

__inline__ __device__ void addToLocalSystem(bool isValidCorr, float* d_JtJ, float* d_Jtr, unsigned int dim, const matNxM<1, 6>& jacobianBlockRow_i, const matNxM<1, 6>& jacobianBlockRow_j,
	unsigned int vi, unsigned int vj, float residual, float weight, unsigned int tidx
	, float* d_sumResidualDEBUG, int* d_numCorrDEBUG)
{
	//fill in bottom half (vi < vj) -> x < y
	for (unsigned int i = 0; i < 6; i++) {
		for (unsigned int j = i; j < 6; j++) {
			float dii = 0.0f;	float djj = 0.0f;	float dij = 0.0f;	float dji = 0.0f;
			__shared__ float s_partJtJ[4];
			if (tidx == 0) { for (unsigned int i = 0; i < 4; i++) s_partJtJ[i] = 0; } //TODO try with first 4 threads for all tidx == 0

			if (isValidCorr) {
				if (vi > 0) {
					dii = jacobianBlockRow_i(i) * jacobianBlockRow_i(j) * weight;
				}
				if (vj > 0) {
					djj = jacobianBlockRow_j(i) * jacobianBlockRow_j(j) * weight;
				}
				if (vi > 0 && vj > 0) {
					dij = jacobianBlockRow_i(i) * jacobianBlockRow_j(j) * weight;
					if (i != j)	{
						dji = jacobianBlockRow_i(j) * jacobianBlockRow_j(i) * weight;
					}
				}
			}
			dii = warpReduce(dii);	djj = warpReduce(djj);	dij = warpReduce(dij);	dji = warpReduce(dji);
			__syncthreads();
			if (tidx % WARP_SIZE == 0) {
				atomicAdd(&s_partJtJ[0], dii);
				atomicAdd(&s_partJtJ[1], djj);
				atomicAdd(&s_partJtJ[2], dij);
				atomicAdd(&s_partJtJ[3], dji);
			}
			__syncthreads();
			if (tidx == 0) {
				atomicAdd(&d_JtJ[(vi * 6 + j)*dim + (vi * 6 + i)], s_partJtJ[0]);
				atomicAdd(&d_JtJ[(vj * 6 + j)*dim + (vj * 6 + i)], s_partJtJ[1]);
				atomicAdd(&d_JtJ[(vj * 6 + j)*dim + (vi * 6 + i)], s_partJtJ[2]);
				atomicAdd(&d_JtJ[(vj * 6 + i)*dim + (vi * 6 + j)], s_partJtJ[3]);
			}
		}
		float jtri = 0.0f;	float jtrj = 0.0f;
		__shared__ float s_partJtr[2];
		if (tidx == 0) { for (unsigned int i = 0; i < 2; i++) s_partJtr[i] = 0; }
		if (isValidCorr) {
			if (vi > 0) jtri = jacobianBlockRow_i(i) * residual * weight;
			if (vj > 0) jtrj = jacobianBlockRow_j(i) * residual * weight;
		}
		jtri = warpReduce(jtri);	jtrj = warpReduce(jtrj);
		__syncthreads();
		if (tidx % WARP_SIZE == 0) {
			atomicAdd(&s_partJtr[0], jtri);
			atomicAdd(&s_partJtr[1], jtrj);
		}
		__syncthreads();
		if (tidx == 0) {
			atomicAdd(&d_Jtr[vi * 6 + i], s_partJtr[0]);
			atomicAdd(&d_Jtr[vj * 6 + i], s_partJtr[1]);
		}
	}
#ifdef PRINT_RESIDUALS_DENSE
	float res = 0.0f;		int num = 0;
	if (isValidCorr) { res = weight * residual * residual;     num = 1; }
	res = warpReduce(res);						num = warpReduce(num);
	if (tidx % WARP_SIZE == 0) {
		atomicAdd(d_sumResidualDEBUG, res);
		atomicAdd(d_numCorrDEBUG, num);
	}
#endif
}
__inline__ __device__ void addToLocalSystemBrute(bool foundCorr, float* d_JtJ, float* d_Jtr, unsigned int dim, const matNxM<1, 6>& jacobianBlockRow_i, const matNxM<1, 6>& jacobianBlockRow_j,
	unsigned int vi, unsigned int vj, float residual, float weight, unsigned int threadIdx)
{
	if (foundCorr) {
		//fill in bottom half (vi < vj) -> x < y
		for (unsigned int i = 0; i < 6; i++) {
			for (unsigned int j = i; j < 6; j++) {
				if (vi > 0) {
					float dii = jacobianBlockRow_i(i) * jacobianBlockRow_i(j) * weight;
					atomicAdd(&d_JtJ[(vi * 6 + j)*dim + (vi * 6 + i)], dii);
				}
				if (vj > 0) {
					float djj = jacobianBlockRow_j(i) * jacobianBlockRow_j(j) * weight;
					atomicAdd(&d_JtJ[(vj * 6 + j)*dim + (vj * 6 + i)], djj);
				}
				if (vi > 0 && vj > 0) {
					float dij = jacobianBlockRow_i(i) * jacobianBlockRow_j(j) * weight;
					atomicAdd(&d_JtJ[(vj * 6 + j)*dim + (vi * 6 + i)], dij);
					if (i != j)	{
						float dji = jacobianBlockRow_i(j) * jacobianBlockRow_j(i) * weight;
						atomicAdd(&d_JtJ[(vj * 6 + i)*dim + (vi * 6 + j)], dji);
					}
				}
			}
			if (vi > 0) atomicAdd(&d_Jtr[vi * 6 + i], jacobianBlockRow_i(i) * residual * weight);
			if (vj > 0) atomicAdd(&d_Jtr[vj * 6 + i], jacobianBlockRow_j(i) * residual * weight);
		}
	}
}

////////////////////////////////////////
// multiply jtj
////////////////////////////////////////

//inefficient version
__inline__ __device__ void applyJTJDenseBruteDevice(unsigned int variableIdx, SolverState& state, float* d_JtJ, unsigned int N, float3& outRot, float3& outTrans)
{
	// Compute J^T*d_Jp here
	outRot = make_float3(0.0f, 0.0f, 0.0f);
	outTrans = make_float3(0.0f, 0.0f, 0.0f);

	const unsigned int dim = 6 * N;

	unsigned int baseVarIdx = variableIdx * 6;
	for (unsigned int i = 1; i < N; i++) // iterate through (6) row(s) of JtJ and all of p (i = 0 not a variable)
	{
		// (row, col) = vars, i
		unsigned int baseIdx = 6 * i;

		float3x3 block00(
			d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 2]);
		float3x3 block01(
			d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 5]);
		float3x3 block10(
			d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 2]);
		float3x3 block11(
			d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 5]);

		//outRot += block00 * state.d_pRot[i] + block01 * state.d_pTrans[i];
		//outTrans += block10 * state.d_pRot[i] + block11 * state.d_pTrans[i];
		outTrans += block00 * state.d_pTrans[i] + block01 * state.d_pRot[i];
		outRot += block10 * state.d_pTrans[i] + block11 * state.d_pRot[i];
	}
}
__inline__ __device__ void applyJTJDenseDevice(unsigned int variableIdx, SolverState& state, float* d_JtJ, unsigned int N, float3& outRot, float3& outTrans, unsigned int threadIdx)
{
	// Compute J^T*d_Jp here
	outRot = make_float3(0.0f, 0.0f, 0.0f);
	outTrans = make_float3(0.0f, 0.0f, 0.0f);

	const unsigned int dim = 6 * N;

	unsigned int baseVarIdx = variableIdx * 6;
	unsigned int i = (threadIdx > 0) ? threadIdx : THREADS_PER_BLOCK_JT_DENSE;
	for (; i < N; i += THREADS_PER_BLOCK_JT_DENSE) // iterate through (6) row(s) of JtJ
	{
		// (row, col) = vars, i
		unsigned int baseIdx = 6 * i;

		float3x3 block00(
			d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 2]);
		float3x3 block01(
			d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 0)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 1)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 2)* dim + baseIdx + 5]);
		float3x3 block10(
			d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 2],
			d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 0], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 1], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 2]);
		float3x3 block11(
			d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 3)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 4)* dim + baseIdx + 5],
			d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 3], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 4], d_JtJ[(baseVarIdx + 5)* dim + baseIdx + 5]);

		//outRot += block00 * state.d_pRot[i] + block01 * state.d_pTrans[i];
		//outTrans += block10 * state.d_pRot[i] + block11 * state.d_pTrans[i];
		outTrans += block00 * state.d_pTrans[i] + block01 * state.d_pRot[i];
		outRot += block10 * state.d_pTrans[i] + block11 * state.d_pRot[i];
	}

	outRot.x = warpReduce(outRot.x);	 outRot.y = warpReduce(outRot.y);	  outRot.z = warpReduce(outRot.z);
	outTrans.x = warpReduce(outTrans.x); outTrans.y = warpReduce(outTrans.y); outTrans.z = warpReduce(outTrans.z);
}

///////////////////////////////////////////////////////////////////
// camera functions
///////////////////////////////////////////////////////////////////
__inline__ __device__ bool computeAngleDiff(const float4x4& transform, float angleThresh)
{
	float3 x = normalize(make_float3(1.0f, 1.0f, 1.0f));
	float3 v = transform.getFloat3x3() * x;
	float angle = acos(clamp(dot(x, v), -1.0f, 1.0f));

	if (fabs(angle) < angleThresh) return true;
	return false;
}


#endif
