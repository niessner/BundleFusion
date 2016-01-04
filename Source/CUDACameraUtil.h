#pragma once
#ifndef _CUDA_CAMERA_UTIL
#define _CUDA_CAMERA_UTIL

#include "GlobalDefines.h"
#include <cutil_inline.h>
#include <cutil_math.h>

__inline__ __device__ float2 cameraToDepth(float fx, float fy, float cx, float cy, const float3& pos)
{
	return make_float2(
		pos.x*fx / pos.z + cx,
		pos.y*fy / pos.z + cy);
}
__inline__ __device__ float3 depthToCamera(float fx, float fy, float cx, float cy, const int2& loc, float depth)	{
	const float x = ((float)loc.x - cx) / fx;
	const float y = ((float)loc.y - cy) / fy;
	return make_float3(depth*x, depth*y, depth);
}

#endif //_CUDA_CAMERA_UTIL