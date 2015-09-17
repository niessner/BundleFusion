#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

#define MINF __int_as_float(0xff800000)
#define MAXF __int_as_float(0x7F7FFFFF)

__device__ inline bool isValid(float4 p)
{
	return (p.x != MINF);
}

__device__ inline bool isValidCol(float4 p)
{
	return p.w != 0.0f;
}

__device__ inline void getBestCorrespondence1x1(
	uint2 screenPos, float4 pInput, float4 nInput, float4 cInput, float4& pTarget, float4& nTarget,
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int width, unsigned int height
	)
{
	const unsigned int idx = screenPos.x + screenPos.y*width;
	pTarget = d_Target[idx];
	nTarget = d_TargetNormals[idx];
}

__device__ inline void getBestCorrespondence1x1(
	uint2 screenPos, float4& pTarget, float4& nTarget,
	float4* d_Target, float4* d_TargetNormals, unsigned int width, unsigned int height,
	unsigned int& idx
	)
{
	idx = screenPos.x + screenPos.y*width;
	pTarget = d_Target[idx];
	nTarget = d_TargetNormals[idx];
}

//__device__ inline float2 cameraToKinectScreenFloat(float3 pos, float4x4 intrinsic)
//{
//	float4 p = make_float4(pos.x, pos.y, 0.0f, pos.z);
//	float4 proj = intrinsic*p;
//	return make_float2(proj.x/proj.w, proj.y/proj.w);
//}
//__device__ inline int2 cameraToKinectScreenInt(float3 pos, float4x4 intrinsic)
//{
//	float2 pImage = cameraToKinectScreenFloat(pos, intrinsic);
//	return make_int2(pImage + make_float2(0.5f, 0.5f));
//}
//
//__device__ inline float cameraToKinectProjZ(float z)
//{
//	#define DEPTH_WORLD_MAX 8.0f
//	#define DEPTH_WORLD_MIN 0.1f
//	return (z - DEPTH_WORLD_MIN)/(DEPTH_WORLD_MAX - DEPTH_WORLD_MIN);
//}

__global__ void projectiveCorrespondencesKernel(	
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int width, unsigned int height,
	float distThres, float normalThres, float levelFactor,
	float4x4 transform, float4x4 intrinsic, DepthCameraData depthCameraData)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;
	d_Output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);
	d_OutputNormals[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	float4 pInput = d_Input[y*width+x];
	float4 nInput = d_InputNormals[y*width+x];
	float4 cInput = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // = d_InputColors[y*width+x];

	if(isValid(pInput) && isValid(nInput))
	{
		pInput.w = 1.0f; // assert it is a point
		//float4 pTransInput = mul(pInput, transform);
		float4 pTransInput = transform * pInput;

		nInput.w = 0.0f;  // assert it is a vector
		//float4 nTransInput = mul(nInput, transform); // transformation is a rotation M^(-1)^T = M, translation is ignored because it is a vector
		float4 nTransInput = transform * nInput;

		//if(pTransInput.z > FLT_EPSILON) // really necessary
		{
			//int2 screenPos = cameraToKinectScreenInt(make_float3(pTransInput), intrinsic);
			int2 screenPos = depthCameraData.cameraToKinectScreenInt(make_float3(pTransInput));
			screenPos = make_int2(screenPos.x/levelFactor, screenPos.y/levelFactor);

			if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < width && screenPos.y < height) {
				float4 pTarget, nTarget;
				getBestCorrespondence1x1(make_uint2(screenPos), pTransInput, nTransInput, cInput, pTarget, nTarget,
					d_Input, d_InputNormals, d_InputColors, d_Target, d_TargetNormals, d_TargetColors, d_Output, d_OutputNormals, width, height);
				if (isValid(pTarget) && isValid(nTarget)) {
					float d = length(make_float3(pTransInput)-make_float3(pTarget));
					float dNormal = dot(make_float3(nTransInput), make_float3(nTarget));

					if (d <= distThres && dNormal >= normalThres)
					{
						d_Output[y*width+x] = pTarget;

						//nTarget.w = max(0.0, 0.5f*((1.0f-d/distThres)+(1.0f-cameraToKinectProjZ(pTransInput.z)))); // for weighted ICP;
						nTarget.w = max(0.0, 0.5f*((1.0f-d/distThres)+(1.0f-depthCameraData.cameraToKinectProjZ(pTransInput.z)))); // for weighted ICP;
						
						d_OutputNormals[y*width+x] = nTarget;
					}
				}
			}
		}
	}
}

extern "C" void projectiveCorrespondences(
	float4* d_Input, float4* d_InputNormals, float4* d_InputColors, 
	float4* d_Target, float4* d_TargetNormals, float4* d_TargetColors, 
	float4* d_Output, float4* d_OutputNormals, unsigned int imageWidth, unsigned int imageHeight,
	float distThres, float normalThres, float levelFactor,
	const float4x4* transform, const float4x4* intrinsic, const DepthCameraData& depthCameraData)
{
	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
	
	projectiveCorrespondencesKernel<<<gridSize, blockSize>>>(
		d_Input, d_InputNormals, NULL, d_Target, d_TargetNormals, NULL, d_Output, d_OutputNormals, imageWidth, imageHeight,
		distThres, normalThres, levelFactor, *transform, *intrinsic, depthCameraData
		);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}
