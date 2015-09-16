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
		pt.x = warpReduceSum(pt.x);
		pt.y = warpReduceSum(pt.y);
		pt.z = warpReduceSum(pt.z);
	}

	if (threadIdx.x == 0) {
		mean = matNxM<3, 1>(pt) / (float)numMatches;
		V.setZero();
	}

	matNxM<3, 3> curr;	curr.setZero();
	if (threadIdx.x < numMatches) {
		const unsigned int i = threadIdx.x;
		unsigned int keyPointIdx = ((unsigned int*)&d_filteredMatchKeyPointIndicesGlobal[i])[which];
		const SIFTKeyPoint& key = d_keyPointsGlobal[keyPointIdx];
		float3 pt = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
		const matNxM<3, 1> p((const float*)&pt);
		curr = (p - mean) * (p - mean).getTranspose();
		for (unsigned int j = 0; j < 9; j++) {
			curr(j) = warpReduceSum(curr(j));
		}
	}

	if (threadIdx.x == 0) {
		V = curr / (float)numMatches;
	}
}


__device__ void computeCovariance2d(
	const float2* points, unsigned int numPoints, matNxM<2,2>& cov)
{
	matNxM<2, 1> p0;	p0.setZero();
	for (unsigned int i = 0; i < numPoints; i++) {
		p0 = p0 + matNxM<2, 1>((const float*)&points[i]);
	}
	p0 = p0 / (float)numPoints;

	cov.setZero();	//covariance matrix	
	for (unsigned int i = 0; i < numPoints; i++) {
		const matNxM<2, 1> p((const float*)&points[i]);
		cov += (p - p0) * (p - p0).getTranspose();
	}
	cov /= (float)numPoints;
}


__device__ float computeAreaOrientedBoundingBox2(const float2* points, unsigned int numPoints)
{
	matNxM<2,2> cov;
	computeCovariance2d(points, numPoints, cov);
	float2 evs = computeEigenValues((float2x2)cov);
	float2 ev0 = computeEigenVector((float2x2)cov, evs.x);
	float2 ev1 = computeEigenVector((float2x2)cov, evs.y);
	
	float2 axis0 = normalize(ev0);
	float2 axis1 = normalize(ev1);

	// find bounds
	float2x2 worldToOBBSpace(axis0, axis1);
	float2 minValues = make_float2(FLT_MAX, FLT_MAX);
	float2 maxValues = make_float2(-FLT_MAX, -FLT_MAX);
	for (unsigned int i = 0; i < numPoints; i++) {
		float2 curr = worldToOBBSpace * points[i];
		if (curr.x < minValues.x)	minValues.x = curr.x;
		if (curr.y < minValues.y)	minValues.y = curr.y;

		if (curr.x > maxValues.x)	maxValues.x = curr.x;
		if (curr.y > maxValues.y)	maxValues.y = curr.y;
	}
	float extentX = maxValues.x - minValues.x;
	float extentY = maxValues.y - minValues.y;

	if (extentX < 0.00001f || extentY < 0.00001f) return 0.0f;
	return extentX * extentY;
}

__device__ void projectKeysToPlane(float2* pointsProj,
	const SIFTKeyPoint* d_keyPointsGlobal, const uint2* d_filteredMatchKeyPointIndicesGlobal,
	unsigned int numMatches, const float4x4& colorIntrinsicsInv, unsigned int which,
	const float3& ev0, const float3& ev1, const float3& ev2, const float3& mean)
{
	for (unsigned int i = 0; i < numMatches; i++) {
		unsigned int keyPointIdx = ((unsigned int*)&d_filteredMatchKeyPointIndicesGlobal[i])[which];
		const SIFTKeyPoint& key = d_keyPointsGlobal[keyPointIdx];
		float3 pt = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
		float3 s = (pt - dot(ev2, pt - mean) * ev2) - mean;
		pointsProj[i] = make_float2(dot(s, ev0), dot(s, ev1));
	}
	//for (unsigned int i = numMatches; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
	//	pointsProj[i] = make_float2(0.0f);
	//}
}

#endif //CUDA_SURFACE_AREA_H