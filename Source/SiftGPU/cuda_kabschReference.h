#pragma once

#ifndef CUDA_KABSCH_REFERENCE_H
#define CUDA_KABSCH_REFERENCE_H

#include "GlobalDefines.h"

#include "cuda_SVD.h"
#include "cuda_SimpleMatrixUtil.h"
#include "SIFTImageManager.h"




__host__ __device__ void getKeySourceAndTargetPointsForIndex(const SIFTKeyPoint* keyPoints, const uint2* keyIndices,
	unsigned int index, float3* srcPt, float3* tgtPt, const float4x4& colorIntrinsicsInv)
{
	// source points
	const SIFTKeyPoint& key0 = keyPoints[keyIndices[index].x];
	srcPt[0] = colorIntrinsicsInv * (key0.depth * make_float3(key0.pos.x, key0.pos.y, 1.0f));

	// target points
	const SIFTKeyPoint& key1 = keyPoints[keyIndices[index].y];
	tgtPt[0] = colorIntrinsicsInv * (key1.depth * make_float3(key1.pos.x, key1.pos.y, 1.0f));
}

__host__ __device__ void getKeySourceAndTargetPointsByIndices(const SIFTKeyPoint* keyPoints, const uint2* keyIndices,
	const unsigned int* indices, unsigned int numIndices, float3* srcPts, float3* tgtPts, const float4x4& colorIntrinsicsInv)
{
	for (unsigned int j = 0; j < numIndices; j++) {
		unsigned int i = indices[j];

		// source points
		const SIFTKeyPoint& key0 = keyPoints[keyIndices[i].x];
		srcPts[j] = colorIntrinsicsInv * (key0.depth * make_float3(key0.pos.x, key0.pos.y, 1.0f));

		// target points
		const SIFTKeyPoint& key1 = keyPoints[keyIndices[i].y];
		tgtPts[j] = colorIntrinsicsInv * (key1.depth * make_float3(key1.pos.x, key1.pos.y, 1.0f));
	}
}

__host__ __device__ float3x3 covReference(volatile float3* source, unsigned numPoints) {
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

__host__ __device__ float3 pcaReference(volatile float3* source, unsigned numPoints, float3 &ev0, float3 &ev1, float3& ev2) {

	float3 evs = make_float3(0.0f, 0.0f, 0.0f);

	float3x3 c = covReference(source, numPoints);
	bool res = MYEIGEN::eigenSystem(c, evs, ev0, ev1, ev2);
	return evs;
}

__host__ __device__ float4x4 kabschReference(volatile float3* source, volatile float3* target, unsigned numPoints, float3& evs) {

	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> P; 
	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> Q;
	matNxM<MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, 1> weights;

	float sumWeights = 0.0f;
	for (unsigned int i = 0; i < numPoints; i++) {
		P(0, i) = source[i].x;
		P(1, i) = source[i].y;
		P(2, i) = source[i].z;

		Q(0, i) = target[i].x;
		Q(1, i) = target[i].y;
		Q(2, i) = target[i].z;

		weights(i) = 1.0f;
		sumWeights += weights(i);
	}

	for (unsigned int i = numPoints; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
		P(0, i) = 0.0f;
		P(1, i) = 0.0f;
		P(2, i) = 0.0f;
		  
		Q(0, i) = 0.0f;
		Q(1, i) = 0.0f;
		Q(2, i) = 0.0f;

		weights(i) = 0.0f;
	}

	weights /= sumWeights;

	// centroids
	matNxM<3, 1> p0 = P * weights;
	matNxM<3, 1> q0 = Q * weights;

	matNxM<1, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> v1;
	for (unsigned int i = 0; i < numPoints; i++) {
		v1(i) = 1.0f;
	}
	for (unsigned int i = numPoints; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
		v1(i) = 0.0f;
	}

	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> P_centered = P - p0 * v1;
	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> Q_centered = Q - q0 * v1;


	matNxM<3, 3> C = P_centered * matNxM<MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED>::getDiagonalMatrix(weights) * Q_centered.getTranspose();
	//matNxM<3, 3> V, S, W, I;
	//I.setIdentity();
	//svd(C, V, S, W);

	matNxM<3, 3> V = C;
	matNxM<3, 1> S;
	matNxM<3, 3> W;
	matNxM<3, 3> I; I.setIdentity();
	SVD::decompose3x3((float*)V.getPointer(), (float*)W.getPointer(), (float*)S.getPointer());

	// SVD
	//Eigen::Matrix3d _C = MatrixConversion::MatToEig(C).cast<double>();
	//Eigen::JacobiSVD<Eigen::Matrix3d> svd(_C, Eigen::ComputeFullU | Eigen::ComputeFullV);
	//Eigen::Matrix3d _V = svd.matrixU();
	//Eigen::Vector3d _S = svd.singularValues();
	//Eigen::Matrix3d _W = svd.matrixV();
	//Eigen::Matrix3d _I = Eigen::Matrix3d::Identity();
	//if ((_V * _W.transpose()).determinant() < 0)
	//	_I(3 - 1, 3 - 1) = -1;
	//Eigen::Matrix3d _resRot = _W * _I * _V.transpose();

	//V = MatrixConversion::toCUDA(_V);
	//W = MatrixConversion::toCUDA(_W);
	//S.setZero(); S(0) = _S(0);	S(1) = _S(1);	S(2) = _S(2);

	// sort eigenvalues
	if (S(0) > S(1) && S(0) > S(2)) {
		evs.x = S(0);
		if (S(1) > S(2)) {
			evs.y = S(1);
			evs.z = S(2);
		}
		else {
			evs.y = S(2);
			evs.z = S(1);
		}
	}
	else if (S(1) > S(0) && S(1) > S(2)) {
		evs.x = S(1);
		if (S(0) > S(2)) {
			evs.y = S(0);
			evs.z = S(2);
		}
		else {
			evs.y = S(2);
			evs.z = S(0);
		}
	}
	else {
		evs.x = S(2);
		if (S(1) > S(2)) {
			evs.y = S(1);
			evs.z = S(2);
		}
		else{
			evs.y = S(2);
			evs.z = S(1);
		}
	}

	if ((V * W.getTranspose()).det() < 0) {
		I(3 - 1, 3 - 1) = -1;
	}

	matNxM<3, 3> resRot = W * I * V.getTranspose();
	//matNxM<3, 3> resRot = MatrixConversion::toCUDA(_resRot.cast<float>());
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
	return ret;
}

__host__ __device__ float3 covarianceSVDReference(volatile float3* source, unsigned numPoints) {

	float3 evs = make_float3(0.0f, 0.0f, 0.0f);
	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> P;
	matNxM<MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, 1> weights;

	float sumWeights = 0.0f;
	for (unsigned int i = 0; i < numPoints; i++) {
		P(0, i) = source[i].x;
		P(1, i) = source[i].y;
		P(2, i) = source[i].z;

		weights(i) = 1.0f;
		sumWeights += weights(i);
	}

	for (unsigned int i = numPoints; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
		P(0, i) = 0.0f;
		P(1, i) = 0.0f;
		P(2, i) = 0.0f;

		weights(i) = 0.0f;
	}

	weights /= sumWeights;

	// centroids
	matNxM<3, 1> p0 = P * weights;

	matNxM<1, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> v1;
	for (unsigned int i = 0; i < numPoints; i++) {
		v1(i) = 1.0f;
	}
	for (unsigned int i = numPoints; i < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED; i++) {
		v1(i) = 0.0f;
	}

	matNxM<3, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED> P_centered = P - p0 * v1;


	matNxM<3, 3> C = P_centered * matNxM<MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, MAX_MATCHES_PER_IMAGE_PAIR_FILTERED>::getDiagonalMatrix(weights) * P_centered.getTranspose();
	matNxM<3, 3> V = C;
	matNxM<3, 1> S;
	matNxM<3, 3> W;
	matNxM<3, 3> I; I.setIdentity();
	SVD::decompose3x3((float*)V.getPointer(), (float*)W.getPointer(), (float*)S.getPointer());

	// sort eigenvalues
	if (S(0) > S(1) && S(0) > S(2)) {
		evs.x = S(0);
		if (S(1) > S(2)) {
			evs.y = S(1);
			evs.z = S(2);
		}
		else {
			evs.y = S(2);
			evs.z = S(1);
		}
	}
	else if (S(1) > S(0) && S(1) > S(2)) {
		evs.x = S(1);
		if (S(0) > S(2)) {
			evs.y = S(0);
			evs.z = S(2);
		}
		else {
			evs.y = S(2);
			evs.z = S(0);
		}
	}
	else {
		evs.x = S(2);
		if (S(1) > S(2)) {
			evs.y = S(1);
			evs.z = S(2);
		}
		else{
			evs.y = S(2);
			evs.z = S(1);
		}
	}

	return evs;
}

__host__ __device__ float computeKabschReprojError(float3* srcPts, float3* tgtPts, unsigned int numMatches, float3& eigenvalues, float4x4& transformEstimate) {

	// kabsch
	transformEstimate = kabschReference(srcPts, tgtPts, numMatches, eigenvalues);

	// kabsch residuals
	float maxResidual = 0.0f;
	for (unsigned int i = 0; i < numMatches; i++) {
		float3 d = transformEstimate * srcPts[i] - tgtPts[i];
		float residual_i = dot(d, d);
		if (residual_i > maxResidual) maxResidual = residual_i;
	}
	return maxResidual;
}

//!!!todo params or defines???
//#define MAX_KABSCH_RESIDUAL_THRESH 0.03f
//#define MIN_NUM_MATCHES_FILTERED 4
#define MATCH_FILTER_PIXEL_DIST_THRESH 5
#define KABSCH_CONDITION_THRESH 100.0f

__host__ __device__ bool addMatchReference(const uint2& addkeyIndices, const SIFTKeyPoint* keyPoints, uint2* keyIndices, unsigned int curNumMatches)
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

__host__ __device__ void getKeySourceAndTargetPointsReference(const SIFTKeyPoint* keyPoints, const uint2* keyIndices, unsigned int numMatches, float3* srcPts, float3* tgtPts, const float4x4& colorIntrinsicsInv)
{
	//const float _colorIntrinsicsInverse[16] = { 
	//	0.000847232877f, 0.0f, -0.549854159f, 0.0f,
	//	0.0f, 0.000850733835f, -0.411329806f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f};
	//float4x4 colorIntrinsicsInv(_colorIntrinsicsInverse);

	for (unsigned int i = 0; i < numMatches; i++) {
		// source points
		const SIFTKeyPoint& key0 = keyPoints[keyIndices[i].x];
		srcPts[i] = colorIntrinsicsInv * (key0.depth * make_float3(key0.pos.x, key0.pos.y, 1.0f));

		// target points
		const SIFTKeyPoint& key1 = keyPoints[keyIndices[i].y];
		tgtPts[i] = colorIntrinsicsInv * (key1.depth * make_float3(key1.pos.x, key1.pos.y, 1.0f));
	}
}


__host__ __device__ bool cmpAndSwapResidualReference(
	float* residual0, float3* srcPtrs0, float3* tgtPts0, uint2* keyPointIndices0, float* matchDistances0,
	float* residual1, float3* srcPtrs1, float3* tgtPts1, uint2* keyPointIndices1, float* matchDistances1) {
	if (*residual0 > *residual1) {
		float tmpRes = *residual0;
		*residual0 = *residual1;
		*residual1 = tmpRes;

		float3 tmpSrc = *srcPtrs0;
		*srcPtrs0 = *srcPtrs1;
		*srcPtrs1 = tmpSrc;

		float3 tmpTgtPts = *tgtPts0;
		*tgtPts0 = *tgtPts1;
		*tgtPts1 = tmpTgtPts;

		uint2 tmpKeyPointIndices = *keyPointIndices0;
		*keyPointIndices0 = *keyPointIndices1;
		*keyPointIndices1 = tmpKeyPointIndices;

		float tmpMatchDistances = *matchDistances0;
		*matchDistances0 = *matchDistances1;
		*matchDistances1 = tmpMatchDistances;
		return true;
	}
	else {
		return false;
	}
}

__host__ __device__ void sortKabschResidualsReference(float* residuals, unsigned int num, float3* srcPts, float3* tgtPts, uint2* keyPointIndices, float* matchDistances) {
	for (unsigned int i = 0; i < num; i++) {
		for (unsigned int j = i; j < num; j++) {
			bool res = cmpAndSwapResidualReference(
				&residuals[i], &srcPts[i], &tgtPts[i], &keyPointIndices[i], &matchDistances[i],
				&residuals[j], &srcPts[j], &tgtPts[j], &keyPointIndices[j], &matchDistances[j]);
		}
	}
}

__host__ __device__ bool ComputeReprojectionReference(float3* srcPts, float3* tgtPts, unsigned int numMatches, float* residuals, float4x4& transformEstimate,
	uint2* keyPointIndices, float* matchDistances) {
	
	// kabsch
	float3 eigenvalues;
	transformEstimate = kabschReference(srcPts, tgtPts, numMatches, eigenvalues);

	// kabsch residuals
	for (unsigned int i = 0; i < numMatches; i++) {
		float3 d = transformEstimate * srcPts[i] - tgtPts[i];
		residuals[i] = dot(d, d);
	}
	// sort srcPts/tgtPts/matches by residual
	sortKabschResidualsReference(residuals, numMatches, srcPts, tgtPts, keyPointIndices, matchDistances);

	float c1 = eigenvalues.x / eigenvalues.y; // ok if coplanar
	eigenvalues = covarianceSVDReference(srcPts, numMatches);
	float cp = eigenvalues.x / eigenvalues.y; // ok if coplanar
	eigenvalues = covarianceSVDReference(tgtPts, numMatches);
	float cq = eigenvalues.x / eigenvalues.y; // ok if coplanar

	if (isnan(c1) || isnan(cp) || isnan(cq) || fabs(c1) > KABSCH_CONDITION_THRESH || fabs(cp) > KABSCH_CONDITION_THRESH || fabs(cq) > KABSCH_CONDITION_THRESH)
		return false;
	return true;
}

//! assumes sorted by distance, ascending
__host__ __device__ unsigned int filterKeyPointMatchesReference(const SIFTKeyPoint* keyPoints,
	/*volatile*/ uint2* keyPointIndices, /*volatile*/ float* matchDistances, unsigned int numRawMatches, float4x4& transformEstimate,
	const float4x4& colorIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool printDebug)
{
	bool done = false;
	unsigned int idx = 0;
	unsigned int curNumMatches = 0;
	float curMaxResidual = 100.0f; // some arbitrary large
	bool validTransform = false;

	transformEstimate.setIdentity();

	uint2 curKeyIndices[MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	float curMatchDistances[MAX_MATCHES_PER_IMAGE_PAIR_RAW];

	float3 srcPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	float3 tgtPts[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	float residuals[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];

	while (!done) {
		if (idx == numRawMatches || curNumMatches >= MAX_MATCHES_PER_IMAGE_PAIR_FILTERED) {
			if (printDebug) printf("done: #matches = %d, maxRes = %f, validTransform = %d\n", curNumMatches, curMaxResidual, validTransform);
			if (curNumMatches < minNumMatches || curMaxResidual >= maxResThresh2 || !validTransform) { // invalid
				curNumMatches = 0;
			}
			done = true;
			break;
		}
		else if (addMatchReference(keyPointIndices[idx], keyPoints, curKeyIndices, curNumMatches)) {
			// add match
			curKeyIndices[curNumMatches] = keyPointIndices[idx];
			curMatchDistances[curNumMatches] = matchDistances[idx];
			curNumMatches++;

			if (printDebug) printf("added %d\n", idx);

			// check geometric consistency
			if (curNumMatches >= 3) {
				getKeySourceAndTargetPointsReference(keyPoints, curKeyIndices, curNumMatches, srcPts, tgtPts, colorIntrinsicsInv);
				bool prevValid = validTransform;
				validTransform = ComputeReprojectionReference(srcPts, tgtPts, curNumMatches, residuals, transformEstimate, curKeyIndices, curMatchDistances);
				bool b = validTransform;
				float4x4 prevTransform = transformEstimate;
				curMaxResidual = residuals[curNumMatches - 1];
				if (printDebug) printf("\tmax res = %f, valid = %d\n", curMaxResidual, validTransform);
				if (curMaxResidual > maxResThresh2) { // some bad matches
					float lastRes = -1;
					int startIdx = (int)curNumMatches - 1;
					for (int i = startIdx; i >= 3; i--) { // remove until max < maxResThresh
						lastRes = residuals[i];
						curNumMatches--;
						validTransform = ComputeReprojectionReference(srcPts, tgtPts, curNumMatches, residuals, transformEstimate, curKeyIndices, curMatchDistances);
						curMaxResidual = residuals[curNumMatches - 1];
						if (printDebug) printf("\tRemoved: max res = %f, valid = %d\n", curMaxResidual, validTransform);
						if (curNumMatches == 3 && (curMaxResidual > maxResThresh2 || b && !validTransform)) {
							curNumMatches++; // removing made it worse... continue
							curMaxResidual = lastRes;
							validTransform = b;
							transformEstimate = prevTransform;
							break;
						}
						// check if good
						if (curMaxResidual < maxResThresh2) break;
					} // removing
				}
			} // enough matches to check geo consistency
		} // added match

		idx++;
	} // while (!done)

	if (curNumMatches > 0) {
		if (printDebug) {
			printf("found %d matches:\n", curNumMatches);
			for (unsigned int i = 0; i < curNumMatches; i++)
				printf("\t(%d,%d)\n", curKeyIndices[i].x, curKeyIndices[i].y);
			for (unsigned int i = 0; i < numRawMatches; i++) 
				printf("\t(%d,%d) %d\n", keyPointIndices[i].x, keyPointIndices[i].y, i);

			std::cout << "waiting..." << std::endl;
			getchar();
		}

		// copy back
		for (unsigned int i = 0; i < curNumMatches; i++) {
			keyPointIndices[i] = curKeyIndices[i];
			matchDistances[i] = curMatchDistances[i];
		}
	}
	return curNumMatches;
}



#endif