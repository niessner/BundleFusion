#ifndef _ICP_UTIL_
#define _ICP_UTIL_

#include "GlobalDefines.h"

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "../../SiftGPU/cuda_SimpleMatrixUtil.h"


// color stuff
inline __device__ mat2x3 dCameraToScreen(const float3& p, float fx, float fy)
{
	mat2x3 res; res.setZero();
	const float wSquared = p.z*p.z;

	res(0, 0) = fx / p.z;
	res(1, 1) = fy / p.z;
	res(0, 2) = -fx * p.x / wSquared;
	res(1, 2) = -fy * p.y / wSquared;

	return res;
}

inline __device__ float2 bilinearInterpolationFloat2(float x, float y, const float2* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float2 s0 = make_float2(0.0f); float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float2 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float2 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float2 s1 = make_float2(0.0f); float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float2 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float2 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float2 p0 = s0 / w0;
	const float2 p1 = s1 / w1;

	float2 ss = make_float2(0.0f); float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return make_float2(MINF);
}

inline __device__ float bilinearInterpolationFloat(float x, float y, const float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if(v10 != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if(v11 != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float p0 = s0 / w0;
	const float p1 = s1 / w1;

	float ss = 0.0f; float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return MINF; 
}
inline __device__ float4 bilinearInterpolationFloat4(float x, float y, const float4* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta = y - p00.y;

	float4 s0 = make_float4(0.0f); float w0 = 0.0f;
	if (p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if (v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if (v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float4 s1 = make_float4(0.0f); float w1 = 0.0f;
	if (p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float4 p0 = s0 / w0;
	const float4 p1 = s1 / w1;

	float4 ss = make_float4(0.0f); float ww = 0.0f;
	if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
	if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

	if (ww > 0.0f) return ss / ww;
	else		  return make_float4(MINF);
}


#ifndef USE_LIE_SPACE

inline __device__ float2 dehomogenize(const float3& v)
{
	return make_float2(v.x/v.z, v.y/v.z);
}

inline __device__ mat2x3 dehomogenizeDerivative(const float3& v)
{
	mat2x3 res; res.setZero();

	const float wSquared = v.z*v.z;

	res(0, 0) = 1.0f/v.z;
	res(1, 1) = 1.0f/v.z;
	res(0, 2) = -v.x/wSquared;
	res(1, 2) = -v.y/wSquared;

	return res;
}

inline __device__ float3x3 evalRMat(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = CosGamma*CosBeta;
	R.m12 = -SinGamma*CosAlpha+CosGamma*SinBeta*SinAlpha;
	R.m13 = SinGamma*SinAlpha+CosGamma*SinBeta*CosAlpha;

	R.m21 = SinGamma*CosBeta;
	R.m22 = CosGamma*CosAlpha+SinGamma*SinBeta*SinAlpha;
	R.m23 = -CosGamma*SinAlpha+SinGamma*SinBeta*CosAlpha;

	R.m31 = -SinBeta;
	R.m32 = CosBeta*SinAlpha;
	R.m33 = CosBeta*CosAlpha;

	return R;
}

inline __device__ float3x3 evalRMat(const float3& angles)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	return evalRMat(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma);
}

inline __device__ float4x4 evalRtMat(const float3& angles, const float3& translation)
{
	const float cosAlpha = cos(angles.x); float cosBeta = cos(angles.y); float cosGamma = cos(angles.z);
	const float sinAlpha = sin(angles.x); float sinBeta = sin(angles.y); float sinGamma = sin(angles.z);

	float4x4 trans; trans.setIdentity();
	trans.setFloat3x3(evalRMat(cosAlpha, cosBeta, cosGamma, sinAlpha, sinBeta, sinGamma));
	trans.m14 = translation.x;
	trans.m24 = translation.y;
	trans.m34 = translation.z;
	return trans;
}

// Rotation Matrix dAlpha
inline __device__ float3x3 evalRMat_dAlpha(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = 0.0f;
	R.m12 = SinGamma*SinAlpha+CosGamma*SinBeta*CosAlpha;
	R.m13 = SinGamma*CosAlpha-CosGamma*SinBeta*SinAlpha;

	R.m21 = 0.0f;
	R.m22 = -CosGamma*SinAlpha+SinGamma*SinBeta*CosAlpha;
	R.m23 = -CosGamma*CosAlpha-SinGamma*SinBeta*SinAlpha;

	R.m31 = 0.0f;
	R.m32 = CosBeta*CosAlpha;
	R.m33 = -CosBeta*SinAlpha;

	return R;
}

inline __device__ float3x3 evalR_dAlpha(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dAlpha(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dBeta
inline __device__ float3x3 evalRMat_dBeta(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = -CosGamma*SinBeta;
	R.m12 = CosGamma*CosBeta*SinAlpha;
	R.m13 = CosGamma*CosBeta*CosAlpha;

	R.m21 = -SinGamma*SinBeta;
	R.m22 = SinGamma*CosBeta*SinAlpha;
	R.m23 = SinGamma*CosBeta*CosAlpha;

	R.m31 = -CosBeta;
	R.m32 = -SinBeta*SinAlpha;
	R.m33 = -SinBeta*CosAlpha;

	return R;
}

inline __device__ float3x3 evalR_dBeta(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dBeta(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

// Rotation Matrix dGamma
inline __device__ float3x3 evalRMat_dGamma(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma)
{
	float3x3 R;
	R.m11 = -SinGamma*CosBeta;
	R.m12 = -CosGamma*CosAlpha-SinGamma*SinBeta*SinAlpha;
	R.m13 = CosGamma*SinAlpha-SinGamma*SinBeta*CosAlpha;

	R.m21 = CosGamma*CosBeta;
	R.m22 = -SinGamma*CosAlpha+CosGamma*SinBeta*SinAlpha;
	R.m23 = SinGamma*SinAlpha+CosGamma*SinBeta*CosAlpha;

	R.m31 = 0.0f;
	R.m32 = 0.0f;
	R.m33 = 0.0f;

	return R;
}

inline __device__ float3x3 evalR_dGamma(float3 angles) // angles = [alpha, beta, gamma]
{
	return evalRMat_dGamma(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z));
}

//----------------------- rigid transform inverse derivatives ------------------------------------

// Rigid Transform Matrix Inverse dAlpha
inline __device__ float4x4 evalRtInverseMat_dAlpha(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = 0.0f;
	Rt.m12 = 0.0f;
	Rt.m13 = 0.0f;
	Rt.m14 = 0.0f;

	Rt.m21 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	Rt.m22 = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	Rt.m23 = CosBeta*CosAlpha;
	Rt.m24 = -(SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha)*x - (-CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha)*y - CosAlpha*CosBeta*z;

	Rt.m31 = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;
	Rt.m32 = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	Rt.m33 = -CosBeta*SinAlpha;
	Rt.m34 = -(SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha)*x - (-CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha)*y + CosBeta*SinAlpha*z;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dAlpha(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dAlpha(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}

// Rigid Transform Matrix Inverse dBeta
inline __device__ float4x4 evalRtInverseMat_dBeta(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = -CosGamma*SinBeta;
	Rt.m12 = -SinGamma*SinBeta;
	Rt.m13 = -CosBeta;
	Rt.m14 = CosGamma*SinBeta*x + SinGamma*SinBeta*y + CosBeta*z;

	Rt.m21 = CosGamma*CosBeta*SinAlpha;
	Rt.m22 = SinGamma*CosBeta*SinAlpha;
	Rt.m23 = -SinBeta*SinAlpha;
	Rt.m24 = -CosGamma*CosBeta*SinAlpha*x - SinGamma*CosBeta*SinAlpha*y + SinBeta*SinAlpha*z;

	Rt.m31 = CosGamma*CosBeta*CosAlpha;
	Rt.m32 = SinGamma*CosBeta*CosAlpha;
	Rt.m33 = -SinBeta*CosAlpha;
	Rt.m34 = -CosGamma*CosBeta*CosAlpha*x - SinGamma*CosBeta*CosAlpha*y + CosAlpha*SinBeta*z;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dBeta(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dBeta(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}

// Rigid Transform Matrix Inverse dGamma
inline __device__ float4x4 evalRtInverseMat_dGamma(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = -SinGamma*CosBeta;
	Rt.m12 = CosGamma*CosBeta;
	Rt.m13 = 0.0f;
	Rt.m14 = SinGamma*CosBeta*x - CosGamma*CosBeta*y;

	Rt.m21 = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;
	Rt.m22 = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	Rt.m23 = 0.0f;
	Rt.m24 = -(-CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha)*x - (-SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha)*y;

	Rt.m31 = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha;
	Rt.m32 = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	Rt.m33 = 0.0f;
	Rt.m34 = -(CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha)*x - (SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha)*y;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dGamma(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dGamma(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}

// Rigid Transform Matrix Inverse dx
inline __device__ float4x4 evalRtInverseMat_dX(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = 0.0f;
	Rt.m12 = 0.0f;
	Rt.m13 = 0.0f;
	Rt.m14 = -CosGamma*CosBeta;

	Rt.m21 = 0.0f;
	Rt.m22 = 0.0f;
	Rt.m23 = 0.0f;
	Rt.m24 = SinGamma*CosAlpha - CosGamma*SinBeta*SinAlpha;

	Rt.m31 = 0.0f;
	Rt.m32 = 0.0f;
	Rt.m33 = 0.0f;
	Rt.m34 = -CosGamma*SinBeta*CosAlpha - SinGamma*SinAlpha;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dX(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dX(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}

// Rigid Transform Matrix Inverse dy
inline __device__ float4x4 evalRtInverseMat_dY(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = 0.0f;
	Rt.m12 = 0.0f;
	Rt.m13 = 0.0f;
	Rt.m14 = -SinGamma*CosBeta;

	Rt.m21 = 0.0f;
	Rt.m22 = 0.0f;
	Rt.m23 = 0.0f;
	Rt.m24 = -CosGamma*CosAlpha - SinGamma*SinBeta*SinAlpha;

	Rt.m31 = 0.0f;
	Rt.m32 = 0.0f;
	Rt.m33 = 0.0f;
	Rt.m34 = CosGamma*SinAlpha - SinGamma*SinBeta*CosAlpha;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dY(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dY(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}

// Rigid Transform Matrix Inverse dz
inline __device__ float4x4 evalRtInverseMat_dZ(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	float4x4 Rt;
	Rt.m11 = 0.0f;
	Rt.m12 = 0.0f;
	Rt.m13 = 0.0f;
	Rt.m14 = SinBeta;

	Rt.m21 = 0.0f;
	Rt.m22 = 0.0f;
	Rt.m23 = 0.0f;
	Rt.m24 = -CosBeta*SinAlpha;

	Rt.m31 = 0.0f;
	Rt.m32 = 0.0f;
	Rt.m33 = 0.0f;
	Rt.m34 = -CosBeta*CosAlpha;

	Rt.m41 = 0.0f;
	Rt.m42 = 0.0f;
	Rt.m43 = 0.0f;
	Rt.m44 = 1.0f;

	return Rt;
}

inline __device__ float4x4 evalRtInverse_dZ(const float3& angles, const float3& translation) // angles = [alpha, beta, gamma]
{
	return evalRtInverseMat_dZ(cos(angles.x), cos(angles.y), cos(angles.z), sin(angles.x), sin(angles.y), sin(angles.z),
		translation.x, translation.y, translation.z);
}



//#define MINF __int_as_float(0xff800000)
inline __device__ float4 bilinearInterpolationFloat4(float x, float y, float4* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != MINF && v00.y != MINF && v00.z != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != MINF && v10.y != MINF && v10.z != MINF) { s0 +=		alpha *v10; w0 +=		alpha ; } }

	float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != MINF && v01.y != MINF && v01.z != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != MINF && v11.y != MINF && v11.z != MINF) { s1 +=		alpha *v11; w1 +=		alpha ;} }

	const float4 p0 = s0/w0;
	const float4 p1 = s1/w1;

	float4 ss = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return make_float4(MINF, MINF, MINF, MINF);
}

inline __device__ float bilinearInterpolationFloat(float x, float y, float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if(v10 != MINF) { s0 +=		alpha *v10; w0 +=		alpha ; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if(v11 != MINF) { s1 +=		alpha *v11; w1 +=		alpha ;} }

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ss = 0.0f; float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return MINF;
}

// Nearest neighbour
inline __device__ float4 getValueNearestNeighbour(const float x, const float y, const float4* inputMap, unsigned int imageWidth, unsigned int imageHeight)
{
	const int u = (int)(x + 0.5f);
	const int v = (int)(y + 0.5f);

	if(u < 0 || u >= imageWidth || v < 0 || v >= imageHeight) return make_float4(MINF, MINF, MINF, MINF);

	return inputMap[v*imageWidth + u];
}

//wrt tx, ty, tz, a, b, c
inline __device__ matNxM<12, 6> evalRtDeriv(float CosAlpha, float CosBeta, float CosGamma, float SinAlpha, float SinBeta, float SinGamma, float x, float y, float z)
{
	matNxM<12, 6> deriv; deriv.setZero();
	deriv(9, 0) = 1.0f;		//d tx
	deriv(10, 1) = 1.0f;	//d ty
	deriv(11, 2) = 1.0f;	//d tz

	deriv(0, 3) = 0.0f;														deriv(0, 4) = -SinBeta * CosGamma;									deriv(0, 5) = -CosBeta * SinGamma;	//d cy * cz
	deriv(1, 3) = 0.0f;														deriv(1, 4) = -SinBeta * SinGamma;									deriv(1, 5) = CosBeta * SinGamma;	//d cy * sz
	deriv(2, 3) = 0.0f;														deriv(2, 4) = -CosBeta;												deriv(2, 5) = 0.0f;					//d -sy
	deriv(3, 3) = CosGamma * CosAlpha * SinBeta + SinAlpha * SinGamma;		deriv(3, 4) = CosGamma * SinAlpha * CosBeta - CosAlpha * SinGamma;	deriv(3, 5) = -SinGamma * SinAlpha * SinBeta - CosAlpha * CosGamma; //d cz*sx*sy-cx*sz
	deriv(4, 3) = -SinAlpha * CosGamma + CosAlpha * SinBeta * SinGamma;		deriv(4, 4) = CosAlpha * CosGamma + CosAlpha * SinBeta * SinGamma;	deriv(4, 5) = -CosAlpha * SinGamma + SinAlpha * SinBeta * CosGamma; // d cx*cz+sx*sy*sz
	deriv(5, 3) = CosAlpha * CosBeta;										deriv(5, 4) = -SinBeta * SinAlpha;									deriv(5, 5) = 0.0f;					//d cy*sx
	deriv(6, 3) = -SinAlpha * CosGamma * SinBeta + CosAlpha * SinGamma;		deriv(6, 4) = CosAlpha * CosGamma * CosBeta + SinAlpha * SinGamma;	deriv(6, 5) = -CosAlpha * SinGamma * SinBeta + SinAlpha * CosGamma; //d cx*cz*sy+sx*sz
	deriv(7, 3) = -CosGamma * SinAlpha - CosAlpha * SinBeta * SinGamma;		deriv(7, 4) = -CosGamma * SinAlpha + CosAlpha * CosBeta * SinGamma; deriv(7, 5) = SinGamma * SinAlpha + CosAlpha * SinBeta * CosGamma;  //d -cz*sx+cx*sy*sz
	deriv(8, 3) = -SinAlpha * CosBeta;										deriv(8, 4) = -CosAlpha * SinBeta;									deriv(8, 5) = 0.0f;					//d cx*cy
}

#endif //USE_LIE_SPACE

#endif // _ICP_UTIL_
