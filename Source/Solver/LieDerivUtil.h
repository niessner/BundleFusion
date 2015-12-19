#ifndef _LIE_DERIV_UTIL_
#define _LIE_DERIV_UTIL_

#include "GlobalDefines.h"
#ifdef USE_LIE_SPACE

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "../../SiftGPU/cuda_SimpleMatrixUtil.h"

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

__inline__ __device__ void matrixToPose(const float4x4& matrix, float3& rot, float3& trans)
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

__inline__ __device__ void poseToMatrix(const float3& rot, const float3& trans, float4x4& matrix)
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

__inline__ __device__ float4x4 poseToMatrix(const float3& rot, const float3& trans)
{
	float4x4 res;
	poseToMatrix(rot, trans, res);
	return res;
}

/////////////////////////////////////////////////////////////////////////
// deriv wrt e^e * T * p; pTransformed = T * p [alpha,beta,gamma,tx,ty,tz]
/////////////////////////////////////////////////////////////////////////

__inline__ __device__ float3 evalLie_dAlpha(const float3& pTransformed)
{
	return make_float3(0.0f, -pTransformed.z, pTransformed.y);
}
__inline__ __device__ float3 evalLie_dBeta(const float3& pTransformed)
{
	return make_float3(pTransformed.z, 0.0f, -pTransformed.x);
}
__inline__ __device__ float3 evalLie_dGamma(const float3& pTransformed)
{
	return make_float3(-pTransformed.y, pTransformed.x, 0.0f);
}
//__inline__ __device__ float3 evalLie_dX()
//{
//	return make_float3(1.0f, 0.0f, 0.0f);
//}
//__inline__ __device__ float3 evalLie_dY()
//{
//	return make_float3(0.0f, 1.0f, 0.0f);
//}
//__inline__ __device__ float3 evalLie_dZ()
//{
//	return make_float3(0.0f, 0.0f, 1.0f);
//}


/////////////////////////////////////////////////////////////////////////
// Lie Update
/////////////////////////////////////////////////////////////////////////

__inline__ __device__ void computeLieUpdate(const float3& updateW, const float3& updateT, const float3& curW, const float3& curT,
	float3& newW, float3& newT)
{
	const float4x4 update = poseToMatrix(updateW, updateT);
	const float4x4 cur = poseToMatrix(curW, curT);
	matrixToPose(update * cur, newW, newT);
}

#endif //USE_LIE_SPACE
#endif // _LIE_DERIV_UTIL_
