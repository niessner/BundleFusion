#ifndef _LIE_DERIV_UTIL_
#define _LIE_DERIV_UTIL_

#include "GlobalDefines.h"
#ifdef USE_LIE_SPACE

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>
#include <math_constants.h>

#include "../../SiftGPU/cuda_SimpleMatrixUtil.h"

#define ONE_TWENTIETH 0.05f
#define ONE_SIXTH 0.16666667f

//! compute a rotation exponential using the Rodrigues Formula.
//		rotation axis w (theta = |w|); A = sin(theta) / theta; B = (1 - cos(theta)) / theta^2
__inline__ __device__ void rodrigues_so3_exp(const float3& w, float A, float B, float3x3& R)
{
	{
		const float wx2 = w.x * w.x;
		const float wy2 = w.y * w.y;
		const float wz2 = w.z * w.z;
		R(0, 0) = 1.0f - B*(wy2 + wz2);
		R(1, 1) = 1.0f - B*(wx2 + wz2);
		R(2, 2) = 1.0f - B*(wx2 + wy2);
	}
	{
		const float a = A*w.z;
		const float b = B*(w.x * w.y);
		R(0, 1) = b - a;
		R(1, 0) = b + a;
	}
	{
		const float a = A*w.y;
		const float b = B*(w.x * w.z);
		R(0, 2) = b + a;
		R(2, 0) = b - a;
	}
	{
		const float a = A*w.x;
		const float b = B*(w.y * w.z);
		R(1, 2) = b - a;
		R(2, 1) = b + a;
	}
}

//! exponentiate a vector in the Lie algebra to generate a new SO3(a 3x3 rotation matrix).
__inline__ __device__ float3x3 exp_rotation(const float3& w)
{
	const float theta_sq = dot(w, w);
	const float theta = std::sqrt(theta_sq);
	float A, B;
	//Use a Taylor series expansion near zero. This is required for
	//accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
	if (theta_sq < 1e-8) {
		A = 1.0f - ONE_SIXTH * theta_sq;
		B = 0.5f;
	}
	else {
		if (theta_sq < 1e-6) {
			B = 0.5f - 0.25f * ONE_SIXTH * theta_sq;
			A = 1.0f - theta_sq * ONE_SIXTH*(1.0f - ONE_TWENTIETH * theta_sq);
		}
		else {
			const float inv_theta = 1.0f / theta;
			A = sin(theta) * inv_theta;
			B = (1 - cos(theta)) * (inv_theta * inv_theta);
		}
	}

	float3x3 result;
	rodrigues_so3_exp(w, A, B, result);
	return result;
}

//! logarithm of the 3x3 rotation matrix, generating the corresponding vector in the Lie Algebra
__inline__ __device__ float3 ln_rotation(const float3x3& rotation)
{
	float3 result; // skew symm matrix = (R - R^T) * angle / (2 * sin(angle))

	const float cos_angle = (rotation.trace() - 1.0f) * 0.5f;
	//(R - R^T) / 2
	result.x = (rotation(2, 1) - rotation(1, 2))*0.5f;
	result.y = (rotation(0, 2) - rotation(2, 0))*0.5f;
	result.z = (rotation(1, 0) - rotation(0, 1))*0.5f;

	float sin_angle_abs = length(result); //sqrt(result*result);
	if (cos_angle > (float)0.70710678118654752440) { // [0 - Pi/4[ use asin
		if (sin_angle_abs > 0) {
			result *= asin(sin_angle_abs) / sin_angle_abs;
		}
	}
	else if (cos_angle > -(float)0.70710678118654752440) { // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
		float angle = acos(cos_angle);
		result *= angle / sin_angle_abs;
	}
	else
	{  // rest use symmetric part
		// antisymmetric part vanishes, but still large rotation, need information from symmetric part
		const float angle = CUDART_PI_F - asin(sin_angle_abs);
		const float
			d0 = rotation(0, 0) - cos_angle,
			d1 = rotation(1, 1) - cos_angle,
			d2 = rotation(2, 2) - cos_angle;
		float3 r2;
		if (fabs(d0) > fabs(d1) && fabs(d0) > fabs(d2))
		{ // first is largest, fill with first column
			r2.x = d0;
			r2.y = (rotation(1, 0) + rotation(0, 1))*0.5f;
			r2.z = (rotation(0, 2) + rotation(2, 0))*0.5f;
		}
		else if (fabs(d1) > fabs(d2))
		{ 			    // second is largest, fill with second column
			r2.x = (rotation(1, 0) + rotation(0, 1))*0.5f;
			r2.y = d1;
			r2.z = (rotation(2, 1) + rotation(1, 2))*0.5f;
		}
		else
		{							    // third is largest, fill with third column
			r2.x = (rotation(0, 2) + rotation(2, 0))*0.5f;
			r2.y = (rotation(2, 1) + rotation(1, 2))*0.5f;
			r2.z = d2;
		}
		// flip, if we point in the wrong direction!
		if (dot(r2, result) < 0)
			r2 *= -1;
		result = r2;
		result *= (angle / length(r2));
	}
	return result;
}

__inline__ __device__ void matrixToPose(const float4x4& matrix, float3& rot, float3& trans)
{	
	const float3x3 R = matrix.getFloat3x3();
	const float3 t = matrix.getTranslation();
	rot = ln_rotation(R);
	const float theta = length(rot);

	float shtot = 0.5f;
	if (theta > 0.00001f)
		shtot = sin(theta*0.5f) / theta;

	// now do the rotation
	float3 rot_half = rot;
	rot_half *= -0.5f;
	const float3x3 halfrotator = exp_rotation(rot_half);

	trans = halfrotator * t;

	if (theta > 0.001f)
		trans -= rot * (dot(t, rot) * (1 - 2 * shtot) / dot(rot, rot));
	else					 
		trans -= rot * (dot(t, rot) / 24);
	trans *= 1.0f / (2 * shtot);
}

__inline__ __device__ void poseToMatrix(const float3& rot, const float3& trans, float4x4& matrix)
{
	matrix.setIdentity();

	float3 translation;
	float3x3 rotation;

	const float theta_sq = dot(rot, rot);
	const float theta = std::sqrt(theta_sq);
	float A, B;

	float3 cr = cross(rot, trans);

	if (theta_sq < 1e-8)
	{
		A = 1.0f - ONE_SIXTH * theta_sq;
		B = 0.5f;
		translation = trans + 0.5f * cr;
	}
	else
	{
		float C;
		if (theta_sq < 1e-6) {
			C = ONE_SIXTH*(1.0f - ONE_TWENTIETH * theta_sq);
			A = 1.0f - theta_sq * C;
			B = 0.5f - 0.25f * ONE_SIXTH * theta_sq;
		}
		else {
			const float inv_theta = 1.0f / theta;
			A = sin(theta) * inv_theta;
			B = (1 - cos(theta)) * (inv_theta * inv_theta);
			C = (1 - A) * (inv_theta * inv_theta);
		}

		float3 w_cross = cross(rot, cr);
		translation = trans + B * cr + C * w_cross;
	}

	// 3x3 rotation part:
	rodrigues_so3_exp(rot, A, B, rotation);

	//set rotation
	matrix.setFloat3x3(rotation);
	//set translation
	matrix.m14 = translation.x;
	matrix.m24 = translation.y;
	matrix.m34 = translation.z;
}

__inline__ __device__ float4x4 poseToMatrix(const float3& rot, const float3& trans)
{
	float4x4 res;
	poseToMatrix(rot, trans, res);
	return res;
}

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

/////////////////////////////////////////////////////////////////////////
// deriv for Ti: (A * e^e * D)^{-1} * p; A = Tj^{-1}; D = Ti
/////////////////////////////////////////////////////////////////////////
__inline__ __device__ matNxM<3, 6> evalLie_derivI(const float4x4& A, const float4x4& D, const float3& p)
{
	matNxM<3, 12> j0; matNxM<12, 6> j1;
	const float4x4 transform = A * D;
	float3 pt = p - transform.getTranslation();
	j0.setZero();	j1.setZero();
	j0(0, 0) = pt.x;	j0(0, 1) = pt.y;	j0(0, 2) = pt.z;
	j0(1, 3) = pt.x;	j0(1, 4) = pt.y;	j0(1, 5) = pt.z;
	j0(2, 6) = pt.x;	j0(2, 7) = pt.y;	j0(2, 8) = pt.z;
	for (unsigned int r = 0; r < 3; r++) {
		for (unsigned int c = 0; c < 3; c++) {
			j0(r, c + 9) = -transform(c, r); //-R(AD)^T
			j1(r + 9, c) = A(r, c);	 // R(A)
		}
	}
	const float3x3 RA = A.getFloat3x3();
	for (unsigned int k = 0; k < 4; k++) {
		float3x3 m = RA * VectorToSkewSymmetricMatrix(make_float3(D(0, k), D(1, k), D(2, k))) * -1.0f; //RA * col k of D
		for (unsigned int r = 0; r < 3; r++) {
			for (unsigned int c = 0; c < 3; c++)
				j1(3 * k + r, 3 + c) = m(r, c);
		}
	}

	return (j0 * j1);
}

/////////////////////////////////////////////////////////////////////////
// deriv for Tj: (A * e^e * D) * p; A = Ti^{-1}; D = Tj
/////////////////////////////////////////////////////////////////////////
__inline__ __device__ matNxM<3, 6> evalLie_derivJ(const float4x4& A, const float4x4& D, const float3& p)
{
	float3 dr1 = make_float3(D(0, 0), D(0, 1), D(0, 2));	//rows of D (rotation part)
	float3 dr2 = make_float3(D(1, 0), D(1, 1), D(1, 2));
	float3 dr3 = make_float3(D(2, 0), D(2, 1), D(2, 2));
	float dtx = D(0, 3);	//translation of D
	float dty = D(1, 3);
	float dtz = D(2, 3);
	matNxM<3, 6> jac;
	jac(0, 0) = 1.0f;	jac(0, 1) = 0.0f;	jac(0, 2) = 0.0f;
	jac(1, 0) = 0.0f;	jac(1, 1) = 1.0f;	jac(1, 2) = 0.0f;
	jac(2, 0) = 0.0f;	jac(2, 1) = 0.0f;	jac(2, 2) = 1.0f;
	jac(0, 3) = 0.0f;					jac(0, 4) = dot(p, dr3) + dtz;		jac(0, 5) = -(dot(p, dr2) + dty);
	jac(1, 3) = -(dot(p, dr3) + dtz);	jac(1, 4) = 0.0f;					jac(1, 5) = dot(p, dr1) + dtx;
	jac(2, 3) = dot(p, dr2) + dty;		jac(2, 4) = -(dot(p, dr1) + dtx);	jac(2, 5) = 0.0f;

	jac = mat3x3(A.getFloat3x3()) * jac;
	return jac;
}

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
