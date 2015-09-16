#pragma once
#ifndef CUDA_SURFACE_AREA_H
#define CUDA_SURFACE_AREA_H

#include "cuda_EigenValue.h"
#include "SIFTImageManager.h"
#include "cudaUtil.h"


__shared__ matNxM<3, 3> V;
__shared__ matNxM<3, 1> mean;

__device__ void computeKeyPointMatchesCovariance(
	const SIFTKeyPoint* d_keyPointsGlobal, const uint2* d_filteredMatchKeyPointIndicesGlobal,
	unsigned int numMatches, const float4x4& colorIntrinsicsInv, unsigned int which)
{

	float3 pt = make_float3(0.0f);
	if (threadIdx.x < numMatches) {
		const unsigned int i = threadIdx.x;
		unsigned int keyPointIdx = ((unsigned int*)&d_filteredMatchKeyPointIndicesGlobal[i])[which];
		const SIFTKeyPoint& key = d_keyPointsGlobal[keyPointIdx];
		pt = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
	}
	pt.x = warpReduceSum(pt.x);
	pt.y = warpReduceSum(pt.y);
	pt.z = warpReduceSum(pt.z);
	
	if (threadIdx.x == 0) {
		mean = matNxM<3, 1>(pt) / (float)numMatches;
	}

	__syncthreads();

	matNxM<3, 3> curr;	curr.setZero();
	if (threadIdx.x < numMatches) {
		const unsigned int i = threadIdx.x;
		unsigned int keyPointIdx = ((unsigned int*)&d_filteredMatchKeyPointIndicesGlobal[i])[which];
		const SIFTKeyPoint& key = d_keyPointsGlobal[keyPointIdx];
		float3 pt = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
		const matNxM<3, 1> p((const float*)&pt);
		curr = (p - mean) * (p - mean).getTranspose();
	}
	for (unsigned int j = 0; j < 9; j++) {
		curr(j) = warpReduceSum(curr(j));
	}

	if (threadIdx.x == 0) {
		V = curr / (float)numMatches;
	}

	__syncthreads();
}


__shared__ matNxM<2, 1> mean2x1;
__shared__ matNxM<2, 2> cov2x2;

__device__ void computeCovariance2d(volatile float2* points, unsigned int numPoints)
{
	float2 p0 = make_float2(0.0f);
	if (threadIdx.x < numPoints) {
		const unsigned int i = threadIdx.x;
		p0 = make_float2(points[i].x, points[i].y);
	}
	p0.x = warpReduceSum(p0.x);
	p0.y = warpReduceSum(p0.y);
		
	if (threadIdx.x == 0) {
		mean2x1 = matNxM<2,1>(p0) / (float)numPoints;
	}

	__syncthreads();

	matNxM<2, 2> curr;	curr.setZero();
	if (threadIdx.x < numPoints) {
		const unsigned int i = threadIdx.x;
		const matNxM<2, 1> p((volatile const float*)&points[i]);
		curr = (p - mean2x1) * (p - mean2x1).getTranspose();
	}
	for (unsigned int j = 0; j < 4; j++) {
		curr(j) = warpReduceSum(curr(j));
	}

	if (threadIdx.x == 0) {
		cov2x2 = curr / (float)numPoints;
	}

	__syncthreads();
}


__device__ float computeAreaOrientedBoundingBox2(volatile float2* points, unsigned int numPoints)
{
	
	
	computeCovariance2d(points, numPoints);
	matNxM<2, 2> cov = cov2x2;


	float2 evs = computeEigenValues((float2x2)cov);
	float2 ev0 = computeEigenVector((float2x2)cov, evs.x);
	float2 ev1 = computeEigenVector((float2x2)cov, evs.y);

	float2 axis0 = normalize(ev0);
	float2 axis1 = normalize(ev1);

	// find bounds
	float2x2 worldToOBBSpace(axis0, axis1);


	float2 minValues = make_float2(FLT_MAX, FLT_MAX);
	float2 maxValues = make_float2(-FLT_MAX, -FLT_MAX);

	//for (unsigned int i = 0; i < numPoints; i++) {
	if (threadIdx.x < numPoints) {
		const unsigned int i = threadIdx.x;
		const float2 p = make_float2(points[i].x, points[i].y);
		float2 curr = worldToOBBSpace * p;
		minValues = curr;
		maxValues = curr;
	}

	minValues.x = warpReduceMin(minValues.x);
	minValues.y = warpReduceMin(minValues.y);

	maxValues.x = warpReduceMax(maxValues.x);
	maxValues.y = warpReduceMax(maxValues.y);

	float extentX = maxValues.x - minValues.x;
	float extentY = maxValues.y - minValues.y;

	if (extentX < 0.00001f || extentY < 0.00001f) return 0.0f;
	else return extentX * extentY;

}

__device__ void projectKeysToPlane(volatile float2* pointsProj,
	const SIFTKeyPoint* d_keyPointsGlobal, const uint2* d_filteredMatchKeyPointIndicesGlobal,
	unsigned int numMatches, const float4x4& colorIntrinsicsInv, unsigned int which,
	const float3& ev0, const float3& ev1, const float3& ev2, const float3& mean)
{

	//for (unsigned int i = 0; i < numMatches; i++) {
	const unsigned int i = threadIdx.x;
	if (i < numMatches) {
		
		unsigned int keyPointIdx = ((unsigned int*)&d_filteredMatchKeyPointIndicesGlobal[i])[which];
		const SIFTKeyPoint& key = d_keyPointsGlobal[keyPointIdx];
		float3 pt = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
		float3 s = (pt - dot(ev2, pt - mean) * ev2) - mean;
		pointsProj[i].x = dot(s, ev0);
		pointsProj[i].y = dot(s, ev1);
	} else {
		pointsProj[i].x = 0.0f;
		pointsProj[i].y = 0.0f;
	}
	//for (unsigned int i = numMatches; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
	//	pointsProj[i] = make_float2(0.0f);
	//}
}

#endif //CUDA_SURFACE_AREA_H