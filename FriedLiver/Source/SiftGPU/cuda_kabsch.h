

#ifndef CUDA_KABSCH_H
#define CUDA_KABSCH_H

#include "GlobalDefines.h"
#include "cuda_svd3.h"
#include "cuda_SVD.h"
#include "cuda_EigenValue.h"
#include "SIFTImageManager.h"

__host__ __device__ float3x3 cov(volatile float3* source, unsigned numPoints) {
	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> P;

	matNxM<3, 1> mean;	mean(0, 0) = mean(1, 0) = mean(2, 0) = 0.0f;

	for (unsigned int i = 0; i < numPoints; i++) {
		P(0, i) = source[i].x;
		P(1, i) = source[i].y;
		P(2, i) = source[i].z;

		mean(0, 0) += P(0, i);
		mean(1, 0) += P(1, i);
		mean(2, 0) += P(2, i);
	}

	mean /= (float)(numPoints);

	for (unsigned int i = numPoints; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
		P(0, i) = 0.0f;
		P(1, i) = 0.0f;
		P(2, i) = 0.0f;
	}

	for (unsigned int i = 0; i < numPoints; i++) {
		P(0, i) -= mean(0, 0);
		P(1, i) -= mean(1, 0);
		P(2, i) -= mean(2, 0);
	}


	matNxM<3, 3> C = P * P.getTranspose() / (numPoints - 1.0f);	//this line can be easily parallelized

	return C;
}

__host__ __device__ float3 pca(volatile float3* source, unsigned numPoints, float3 &ev0, float3 &ev1, float3& ev2) {

	float3 evs = make_float3(0.0f, 0.0f, 0.0f);

	float3x3 c = cov(source, numPoints);
	bool res = MYEIGEN::eigenSystem(c, evs, ev0, ev1, ev2);
	return evs;
}

__host__ __device__ float debugResidual(volatile float3* source, volatile float3* target, unsigned int numPoints, const float4x4& tra) {

	float res = 0.0f;
	for (unsigned int i = 0; i < numPoints; i++) {
		float3 s = make_float3(source[i].x, source[i].y, source[i].z);
		float3 t = make_float3(target[i].x, target[i].y, target[i].z);
		res += dot(tra*s - t, tra*s - t);
	}
	return res;
}

__host__ __device__ void swapTwo(float& a, float& b) {
	float tmp = a;
	a = b;
	b = tmp;
}

__host__ __device__ float4x4 kabsch(volatile float3* source, volatile float3* target, unsigned numPoints, float3& evs) {

	// centroids
	matNxM<3, 1> p0;	p0.setZero();
	matNxM<3, 1> q0;	q0.setZero();
	
	for (unsigned int i = 0; i < numPoints; i++) {
		p0 = p0 + matNxM<3, 1>((volatile const float*)&source[i]);
		q0 = q0 + matNxM<3, 1>((volatile const float*)&target[i]);
	}

	p0 = p0 / (float)numPoints;
	q0 = q0 / (float)numPoints;

	matNxM<3, 3> V;	V.setZero();	//covariance matrix	
	for (unsigned int i = 0; i < numPoints; i++) {
		const matNxM<3, 1> p((volatile const float*)&source[i]);
		const matNxM<3, 1> q((volatile const float*)&target[i]);
		V += (p - p0) * (q - q0).getTranspose();
	}
	V /= (float)numPoints;

	//matNxM<3, 1> S;
	matNxM<3, 3> W;

	//((float3x3*)&V)->print();
	//matNxM<3, 3> orig = V;
	//float resGT = 0.0f;
	//if (true)
	//{
	//	SVD::decompose3x3((float*)V.getPointer(), (float*)W.getPointer(), (float*)S.getPointer());
	//	//printf("V:\n");
	//	//((float3x3*)&V)->print();
	//	//printf("W:\n");
	//	//((float3x3*)&W)->print();

	//	matNxM<3, 3> I; I.setIdentity();
	//	if ((V * W.getTranspose()).det() < 0) {
	//		I(3 - 1, 3 - 1) = -1;
	//	}

	//	matNxM<3, 3> resRot = W * I * V.getTranspose();
	//	matNxM<3, 1> resTrans = q0 - resRot*p0;

	//	float4x4 ret;
	//	for (unsigned int i = 0; i < 3; i++) {
	//		for (unsigned int j = 0; j < 3; j++) {
	//			ret(i, j) = resRot(i, j);
	//		}
	//	}
	//	ret(3, 0) = ret(3, 1) = ret(3, 2) = 0;	ret(3, 3) = 1;
	//	ret(0, 3) = resTrans(0);
	//	ret(1, 3) = resTrans(1);
	//	ret(2, 3) = resTrans(2);

	//	resGT = debugResidual(source, target, numPoints, ret);
	//	//printf("GT\n");
	//	//ret.print();
	//}

	//SVD::decompose3x3((float*)V.getPointer(), (float*)W.getPointer(), (float*)&evs);
	svd(V, W, *(matNxM<3, 1>*)&evs);
	if (evs.x < evs.y) swapTwo(evs.x, evs.y);
	if (evs.y < evs.z) swapTwo(evs.y, evs.z);
	if (evs.x < evs.y) swapTwo(evs.x, evs.y);

	//printf("V_:\n");
	//((float3x3*)&V_)->print();
	//printf("W_:\n");
	//((float3x3*)&W)->print();
	//printf("[%.3f %.3f %.3f] [%.3f %.3f %.3f]\n", S(0, 0), S(1, 0), S(2, 0), s_(0, 0), s_(1, 0), s_(2, 0));


	//// sort eigenvalues
	//if (S(0) > S(1) && S(0) > S(2)) {
	//	evs.x = S(0);
	//	if (S(1) > S(2)) {
	//		evs.y = S(1);
	//		evs.z = S(2);
	//	}
	//	else {
	//		evs.y = S(2);
	//		evs.z = S(1);
	//	}
	//}
	//else if (S(1) > S(0) && S(1) > S(2)) {
	//	evs.x = S(1);
	//	if (S(0) > S(2)) {
	//		evs.y = S(0);
	//		evs.z = S(2);
	//	}
	//	else {
	//		evs.y = S(2);
	//		evs.z = S(0);
	//	}
	//}
	//else {
	//	evs.x = S(2);
	//	if (S(1) > S(2)) {
	//		evs.y = S(1);
	//		evs.z = S(2);
	//	}
	//	else{
	//		evs.y = S(2);
	//		evs.z = S(1);
	//	}
	//}

	matNxM<3, 3> I; I.setIdentity();
	if ((V * W.getTranspose()).det() < 0) {
		I(3 - 1, 3 - 1) = -1;
	}

	matNxM<3, 3> resRot = W * I * V.getTranspose();
	matNxM<3, 1> resTrans = q0 - resRot*p0;

	float4x4 ret;
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < 3; j++) {
			ret(i, j) = resRot(i, j);
		}
	}
	ret(3, 0) = ret(3, 1) = ret(3, 2) = 0;	ret(3, 3) = 1;
	ret(0, 3) = resTrans(0);
	ret(1, 3) = resTrans(1);
	ret(2, 3) = resTrans(2);

	//float resFast = debugResidual(source, target, numPoints, ret);
	////if (resFast*1.1f < resGT*1.0f) {
	//if (resFast*1.0f > resGT*1.10f) {
	////if (true) {
	//	//printf("Fast\n");
	//	//ret.print();
	//	printf("resCmp: [%.6f %.6f] [n=%d]\n", resGT, resFast, numPoints);
	//	//((float3x3*)&orig)->print();
	//	printf("[%.3f %.3f %.3f] [%.3f %.3f %.3f]\n", S(0, 0), S(1, 0), S(2, 0), s_(0, 0), s_(1, 0), s_(2, 0));
	//}
	return ret;
}

__host__ __device__ float3 covarianceSVD(volatile float3* source, unsigned numPoints) {

	matNxM<3, 1> p0;	p0.setZero();
	for (unsigned int i = 0; i < numPoints; i++) {
		p0 = p0 + matNxM<3, 1>((volatile const float*)&source[i]);
	}
	p0 = p0 / (float)numPoints;

	matNxM<3, 3> V;	V.setZero();	//covariance matrix	
	for (unsigned int i = 0; i < numPoints; i++) {
		const matNxM<3, 1> p((volatile const float*)&source[i]);
		V += (p - p0) * (p - p0).getTranspose();
	}
	V /= (float)numPoints;

	float3 _s = computeEigenValues(*(float3x3*)&V);
	return _s;

	//matNxM<3, 1> S;
	//matNxM<3, 3> W;
	////SVD::decompose3x3((float*)V.getPointer(), (float*)W.getPointer(), (float*)S.getPointer());
	//svd(V, W, S);
	////printf("[%.3f %.3f %.3f] [%.3f %.3f %.3f]\n", S(0, 0), S(1, 0), S(2, 0), _s.x, _s.y, _s.z);
	//

	//float3 evs = make_float3(0.0f);
	//// sort eigenvalues
	//if (S(0) > S(1) && S(0) > S(2)) {
	//	evs.x = S(0);
	//	if (S(1) > S(2)) {
	//		evs.y = S(1);
	//		evs.z = S(2);
	//	}
	//	else {
	//		evs.y = S(2);
	//		evs.z = S(1);
	//	}
	//}
	//else if (S(1) > S(0) && S(1) > S(2)) {
	//	evs.x = S(1);
	//	if (S(0) > S(2)) {
	//		evs.y = S(0);
	//		evs.z = S(2);
	//	}
	//	else {
	//		evs.y = S(2);
	//		evs.z = S(0);
	//	}
	//}
	//else {
	//	evs.x = S(2);
	//	if (S(1) > S(2)) {
	//		evs.y = S(1);
	//		evs.z = S(2);
	//	}
	//	else{
	//		evs.y = S(2);
	//		evs.z = S(1);
	//	}
	//}

	//return evs;
}

//!!!todo params or defines???
#define MATCH_FILTER_PIXEL_DIST_THRESH 5
#define KABSCH_CONDITION_THRESH 100.0f

__host__ __device__ bool addMatch(const uint2& addkeyIndices, const SIFTKeyPoint* keyPoints, volatile uint2* keyIndices, unsigned int curNumMatches)
{
	const float pixelDistThresh = 5;
	const float2 ai = keyPoints[addkeyIndices.x].pos;
	const float2 aj = keyPoints[addkeyIndices.y].pos;

	for (unsigned int i = 0; i < curNumMatches; i++) {
		const float2 ki = keyPoints[keyIndices[i].x].pos;
		const float2 kj = keyPoints[keyIndices[i].y].pos;

		if (length(ai - ki) <= pixelDistThresh || length(aj - kj) <= pixelDistThresh)
			return false;
	}
	return true;
}

__host__ __device__  void getKeySourceAndTargetPoints(const SIFTKeyPoint* keyPoints, volatile uint2* keyIndices, unsigned int numMatches, volatile float3* srcPts, volatile float3* tgtPts, const float4x4& colorIntrinsicsInv)
{
	//const float _colorIntrinsicsInverse[16] = {
	//	0.000847232877f, 0.0f, -0.549854159f, 0.0f,
	//	0.0f, 0.000850733835f, -0.411329806f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f };
	//float4x4 colorIntrinsicsInv(_colorIntrinsicsInverse);

	for (unsigned int i = 0; i < numMatches; i++) {
		// source points
		const SIFTKeyPoint& key0 = keyPoints[keyIndices[i].x];
		float3 src = colorIntrinsicsInv * (key0.depth * make_float3(key0.pos.x, key0.pos.y, 1.0f));
		srcPts[i].x = src.x;
		srcPts[i].y = src.y;
		srcPts[i].z = src.z;

		// target points
		const SIFTKeyPoint& key1 = keyPoints[keyIndices[i].y];
		float3 tgt = colorIntrinsicsInv * (key1.depth * make_float3(key1.pos.x, key1.pos.y, 1.0f));
		tgtPts[i].x = tgt.x;
		tgtPts[i].y = tgt.y;
		tgtPts[i].z = tgt.z;
	}
}


__host__ __device__  bool cmpAndSwapResidual(
	float* residual0, volatile float3* srcPtrs0, volatile float3* tgtPtrs0, volatile uint2* keyPointIndices0, volatile float* matchDistances0,
	float* residual1, volatile float3* srcPtrs1, volatile float3* tgtPtrs1, volatile uint2* keyPointIndices1, volatile float* matchDistances1) {
	if (*residual0 > *residual1) {
		float tmpRes = *residual0;
		*residual0 = *residual1;
		*residual1 = tmpRes;

		float tmpSrcX = srcPtrs0->x;
		float tmpSrcY = srcPtrs0->y;
		float tmpSrcZ = srcPtrs0->z;
		srcPtrs0->x = srcPtrs1->x;
		srcPtrs0->y = srcPtrs1->y;
		srcPtrs0->z = srcPtrs1->z;
		srcPtrs1->x = tmpSrcX;
		srcPtrs1->y = tmpSrcY;
		srcPtrs1->z = tmpSrcZ;

		float tmptgtX = tgtPtrs0->x;
		float tmptgtY = tgtPtrs0->y;
		float tmptgtZ = tgtPtrs0->z;
		tgtPtrs0->x = tgtPtrs1->x;
		tgtPtrs0->y = tgtPtrs1->y;
		tgtPtrs0->z = tgtPtrs1->z;
		tgtPtrs1->x = tmptgtX;
		tgtPtrs1->y = tmptgtY;
		tgtPtrs1->z = tmptgtZ;


		uint tmpKeyPointIndicesX = keyPointIndices0->x;
		uint tmpKeyPointIndicesY = keyPointIndices0->y;
		keyPointIndices0->x = keyPointIndices1->x;
		keyPointIndices0->y = keyPointIndices1->y;
		keyPointIndices1->x = tmpKeyPointIndicesX;
		keyPointIndices1->y = tmpKeyPointIndicesY;

		float tmpMatchDistances = *matchDistances0;
		*matchDistances0 = *matchDistances1;
		*matchDistances1 = tmpMatchDistances;
		return true;
	}
	else {
		return false;
	}
}

__host__ __device__  void sortKabschResiduals(float* residuals, unsigned int num, volatile float3* srcPts, volatile float3* tgtPts, volatile uint2* keyPointIndices, volatile float* matchDistances) {
	for (unsigned int i = 0; i < num; i++) {
		for (unsigned int j = i; j < num; j++) {
			bool res = cmpAndSwapResidual(
				&residuals[i], &srcPts[i], &tgtPts[i], &keyPointIndices[i], &matchDistances[i],
				&residuals[j], &srcPts[j], &tgtPts[j], &keyPointIndices[j], &matchDistances[j]);
		}
	}
}

__host__ __device__  bool ComputeReprojection(
	volatile float3* srcPts, 
	volatile float3* tgtPts, 
	unsigned int numMatches, 
	float* residuals, 
	float4x4& transformEstimate,
	volatile uint2* keyPointIndices, 
	volatile float* matchDistances) {

	// kabsch
	float3 eigenvalues;
	transformEstimate = kabsch(srcPts, tgtPts, numMatches, eigenvalues);

	// kabsch residuals
	for (unsigned int i = 0; i < numMatches; i++) {
		float3 d = transformEstimate * make_float3(srcPts[i].x, srcPts[i].y, srcPts[i].z) - make_float3(tgtPts[i].x, tgtPts[i].y, tgtPts[i].z);
		residuals[i] = dot(d, d);
	}
	// sort srcPts/tgtPts/matches by residual
	sortKabschResiduals(residuals, numMatches, srcPts, tgtPts, keyPointIndices, matchDistances);

	float c1 = eigenvalues.x / eigenvalues.y; // ok if coplanar
	eigenvalues = covarianceSVD(srcPts, numMatches);
	float cp = eigenvalues.x / eigenvalues.y; // ok if coplanar
	eigenvalues = covarianceSVD(tgtPts, numMatches);
	float cq = eigenvalues.x / eigenvalues.y; // ok if coplanar

	if (isnan(c1) || isnan(cp) || isnan(cq) || fabs(c1) > KABSCH_CONDITION_THRESH || fabs(cp) > KABSCH_CONDITION_THRESH || fabs(cq) > KABSCH_CONDITION_THRESH)
		return false;
	if (isnan(c1) || isnan(cp) || isnan(cq) || fabs(c1) > KABSCH_CONDITION_THRESH || fabs(cp) > KABSCH_CONDITION_THRESH || fabs(cq) > KABSCH_CONDITION_THRESH)
		return false;
	return true;
}



//! assumes sorted by distance, ascending
#ifdef __CUDACC__
__device__ 
#else
__host__ 
#endif
unsigned int filterKeyPointMatches(
	const SIFTKeyPoint* keyPoints, 
	volatile uint2* keyPointIndices,
	volatile float* matchDistances, 
	unsigned int numRawMatches, 
	float4x4& transformEstimate, const float4x4& colorIntrinsicsInverse, unsigned int minNumMatches,
	float maxKabschRes2)
	//, bool printDebug)
{
	const float maxResThresh = maxKabschRes2;

#ifdef __CUDACC__
	__shared__ float3 srcPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	__shared__ float3 tgtPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	__shared__ float residuals[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
#else 
	float3 srcPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	float3 tgtPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	float residuals[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
#endif

	bool done = false;
	unsigned int idx = 0;
	unsigned int curNumMatches = 0;
	float curMaxResidual = 100.0f; // some arbitrary large
	bool validTransform = false;

	transformEstimate.setIdentity();

	while (!done) {
		if (idx == numRawMatches || curNumMatches >= MAX_MATCHES_PER_IMAGE_PAIR_FILTERED) {
			if (curNumMatches < minNumMatches || curMaxResidual >= maxResThresh || !validTransform) { // invalid
				//if (printDebug) printf("INVALID: cur#matches = %d, curMaxRes = %f, valid = %d\n", curNumMatches, curMaxResidual, validTransform);
				curNumMatches = 0;
			}
			//else if (printDebug) printf("VALID: cur#matches = %d, curMaxRes = %f, valid = %d\n", curNumMatches, curMaxResidual, validTransform);
			done = true;
			break;
		}
		else if (addMatch(make_uint2(keyPointIndices[idx].x, keyPointIndices[idx].y), keyPoints, keyPointIndices, curNumMatches)) {
			// add match
			keyPointIndices[curNumMatches].x = keyPointIndices[idx].x;
			keyPointIndices[curNumMatches].y = keyPointIndices[idx].y;
			matchDistances[curNumMatches] = matchDistances[idx];
			curNumMatches++;

			// check geometric consistency
			if (curNumMatches >= 3) {
				getKeySourceAndTargetPoints(keyPoints, keyPointIndices, curNumMatches, srcPts, tgtPts, colorIntrinsicsInverse);
				validTransform = ComputeReprojection(srcPts, tgtPts, curNumMatches, residuals, transformEstimate, keyPointIndices, matchDistances);
				bool b = validTransform;
				float4x4 prevTransform = transformEstimate;
				curMaxResidual = residuals[curNumMatches - 1];

				if (curMaxResidual > maxResThresh) { // some bad matches
					float lastRes = -1;
					int startIdx = (int)curNumMatches - 1;
					for (int i = startIdx; i >= 3; i--) { // remove until max < maxResThresh
						lastRes = residuals[i];
						curNumMatches--;
						validTransform = ComputeReprojection(srcPts, tgtPts, curNumMatches, residuals, transformEstimate, keyPointIndices, matchDistances);
						curMaxResidual = residuals[curNumMatches - 1];
						if (curNumMatches == 3 && (curMaxResidual > maxResThresh || b && !validTransform)) {
							curNumMatches++; // removing made it worse... continue
							curMaxResidual = lastRes;
							validTransform = b;
							transformEstimate = prevTransform;
							break;
						}
						// check if good
						if (curMaxResidual < maxResThresh) break;
					} // removing
				}
			} // enough matches to check geo consistency
		} // added match

		idx++;
	} // while (!done)

	return curNumMatches;
}




#endif