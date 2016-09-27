#pragma once

#include "stdafx.h"

#include <D3D11.h>
#include "DX11Utils.h"
#include "TimingLogDepthSensing.h"
#include "DepthCameraUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>

class CUDAImageHelper
{
	public:
		static void applyProjectiveCorrespondences(
			float4* dInput, float4* dInputNormals, float4* dInputColors, 
			float4* dTarget, float4* dTargetNormals, float4* dTargetColors, 
			float4* dOutput, float4* dOutputNormals, 
			const float4x4& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, float distThres, float normalThres, float levelFactor, const mat4f& intrinsic, const DepthCameraData& depthCameraData);
};
