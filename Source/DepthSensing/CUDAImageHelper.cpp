#include "stdafx.h"

#include "CUDAImageHelper.h"

#include "GlobalAppState.h"

#include <vector>
#include <iostream>

#include <cutil_inline.h>
#include <cutil_math.h>
#include "cuda_SimpleMatrixUtil.h"

extern "C" void projectiveCorrespondences(
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int imageWidth, unsigned int imageHeight,
	float distThres, float normalThres, float levelFactor,
	const float4x4* transform, const float4x4* intrinsic, const DepthCameraData& depthCameraData);

void CUDAImageHelper::applyProjectiveCorrespondences(
	float4* dInput, float4* dInputNormals, float4* dInputColors, 
	float4* dTarget, float4* dTargetNormals, float4* dTargetColors, 
	float4* dOutput, float4* dOutputNormals, 
	const float4x4& deltaTransform, unsigned int imageWidth, unsigned int imageHeight, float distThres, float normalThres, float levelFactor, const mat4f& intrinsic, const DepthCameraData& depthCameraData) 
{
	projectiveCorrespondences(
		dInput, dInputNormals, dInputColors, 
		dTarget, dTargetNormals, dTargetColors, 
		dOutput, dOutputNormals, 
		imageWidth, imageHeight, distThres, normalThres, levelFactor, 
		&deltaTransform, (const float4x4*)&intrinsic, depthCameraData);
}
