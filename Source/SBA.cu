
#include "mLibCuda.h"
#include "GlobalDefines.h"
#include <math_constants.h>

#define THREADS_PER_BLOCK 8

#ifdef USE_LIE_SPACE

__inline__ __device__ float3x3 VectorToSkewSymmetricMatrix(const float3& v) {
	float3x3 res; res.setZero();
	res(1, 0) = v.z;
	res(2, 0) = -v.y;
	res(2, 1) = v.x;
	res(0, 1) = -v.z;
	res(0, 2) = v.y;
	res(1, 2) = -v.x;
	return res;
}

__inline__ __device__ float3 SkewSymmetricMatrixToVector(const float3x3& m) {
	return make_float3(m(2, 1), m(0, 2), m(1, 0));
}

//! exponential map: so(3) -> SO(3) / w -> R3x3
__inline__ __device__ float3x3 LieAlgebraToLieGroupSO3(const float3& w) {
	float3x3 res; res.setIdentity();
	float norm = length(w);
	if (norm < FLT_EPSILON)
		return res;
	float3x3 wHat = VectorToSkewSymmetricMatrix(w);

	res += wHat * (sin(norm) / norm);
	res += (wHat * wHat) * ((1.0f - cos(norm)) / (norm*norm));
	return res;
}

//! logarithm map: SO(3) -> so(3) / R3x3 -> w
__inline__ __device__ float3 LieGroupToLieAlgebraSO3(const float3x3& R) {
	float tmp = (R.trace() - 1.0f) / 2.0f;

	if (tmp < -1.0f)
		tmp = -1.0f;
	if (tmp > 1.0f)
		tmp = 1.0f;
	float angleOfRotation = acos(tmp);
	if (angleOfRotation == 0.0f)
		return make_float3(0.0f);
	float3x3 lnR = (R - R.getTranspose()) * (angleOfRotation / (2.0f * sin(angleOfRotation)));
	return SkewSymmetricMatrixToVector(lnR);
}

__device__ void matrixToPose(const float4x4& matrix, float3& rot, float3& trans)
{
	float3x3 R = matrix.getFloat3x3();
	float3 tr = matrix.getTranslation();

	rot = LieGroupToLieAlgebraSO3(R);
	trans = tr;

	float norm = length(rot);
	float3x3 skewSymmetricW = VectorToSkewSymmetricMatrix(rot);
	float3x3 V; V.setIdentity();
	if (norm > FLT_EPSILON)	{
		V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
		V += skewSymmetricW * skewSymmetricW * ((norm - sin(norm)) / (norm * norm * norm));
		trans = V.getInverse() * tr;
	}
}

__device__ void poseToMatrix(const float3& rot, const float3& trans, float4x4& matrix)
{
	matrix.setIdentity();

	float norm = length(rot);
	float3 resTrans;
	float3x3 resRot = LieAlgebraToLieGroupSO3(rot);
	if (norm == 0.0f) {
		resTrans = make_float3(0.0f);
	}
	else {
		float3x3 skewSymmetricW = VectorToSkewSymmetricMatrix(rot);
		float3x3 V; V.setIdentity();
		V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
		V += skewSymmetricW * skewSymmetricW * ((norm - sin(norm)) / (norm * norm * norm));
		resTrans = V * trans;
	}
	//set rotation
	matrix.setFloat3x3(resRot);
	//set translation
	matrix.m14 = resTrans.x;
	matrix.m24 = resTrans.y;
	matrix.m34 = resTrans.z;
}
#else
//! assumes z-y-x rotation composition (euler angles)
__device__ void matrixToPose(const float4x4& matrix, float3& rot, float3& trans) {
	trans = make_float3(matrix(0,3), matrix(1,3), matrix(2,3));
	rot = make_float3(0.0f);

	const float eps = 0.0001f;

	float psi, theta, phi; // x,y,z axis angles
	if (matrix(2, 0) > -1+eps && matrix(2, 0) < 1-eps) { // R(2, 0) != +/- 1
		theta = -asin(matrix(2, 0)); // \pi - theta
		float costheta = cos(theta);
		psi = atan2(matrix(2, 1) / costheta, matrix(2, 2) / costheta);
		phi = atan2(matrix(1, 0) / costheta, matrix(0, 0) / costheta);
	}
	else {
		phi = 0;
		if (abs(matrix(2, 0) + 1) < eps) { // R(2, 0) == -1
			theta = CUDART_PI_F / 2.0f;
			psi = phi + atan2(matrix(0, 1), matrix(0, 2));
		}
		else {
			theta = -CUDART_PI_F / 2.0f;
			psi = -phi + atan2(-matrix(0, 1), -matrix(0, 2));
		}
	}

	rot = make_float3(psi, theta, phi);
}

//! assumes z-y-x rotation composition (euler angles)
__device__ void poseToMatrix(const float3& rot, const float3& trans, float4x4& matrix) {

	// rotation
	const float CosAlpha = cos(rot.x); float CosBeta = cos(rot.y); float CosGamma = cos(rot.z);
	const float SinAlpha = sin(rot.x); float SinBeta = sin(rot.y); float SinGamma = sin(rot.z);

	matrix.m11 = CosGamma*CosBeta;
	matrix.m12 = -SinGamma*CosAlpha+CosGamma*SinBeta*SinAlpha;
	matrix.m13 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;

	matrix.m21 = SinGamma*CosBeta;
	matrix.m22 = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	matrix.m23 = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;

	matrix.m31 = -SinBeta;
	matrix.m32 = CosBeta*SinAlpha;
	matrix.m33 = CosBeta*CosAlpha;

	// translation
	matrix.m14 = trans.x;
	matrix.m24 = trans.y;
	matrix.m34 = trans.z;

	matrix.m41 = 0.0f;
	matrix.m42 = 0.0f;
	matrix.m43 = 0.0f;
	matrix.m44 = 1.0f;
}
#endif

__global__ void convertMatricesToPosesCU_Kernel(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numTransforms) {
		matrixToPose(d_transforms[idx], d_rot[idx], d_trans[idx]);
	}
}


extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans)
{
	const unsigned int N = numTransforms;

	convertMatricesToPosesCU_Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_transforms, numTransforms, d_rot, d_trans);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void convertPosesToMatricesCU_Kernel(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numImages) {
		poseToMatrix(d_rot[idx], d_trans[idx], d_transforms[idx]);
	}
}

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms)
{
	const unsigned int N = numImages;

	convertPosesToMatricesCU_Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_rot, d_trans, numImages, d_transforms);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}