
#include <cutil_inline.h>
#include <cutil_math.h>
#include <cuda_runtime.h>

#include "cuda_SimpleMatrixUtil.h"

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024

texture<float, cudaTextureType2D, cudaReadModeElementType> rayMinTextureRef;
texture<float, cudaTextureType2D, cudaReadModeElementType> rayMaxTextureRef;

__global__ void renderKernel(HashDataStruct hashData, RayCastData rayCastData) 
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	const RayCastParams& rayCastParams = c_rayCastParams;

	if (x < rayCastParams.m_width && y < rayCastParams.m_height) {
		rayCastData.d_depth[y*rayCastParams.m_width+x] = MINF;
		rayCastData.d_depth4[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
		rayCastData.d_normals[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
		rayCastData.d_colors[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);

		float3 camDir = normalize(RayCastData::depthToCamera(x, y, 1.0f));
		float3 worldCamPos = rayCastParams.m_viewMatrixInverse * make_float3(0.0f, 0.0f, 0.0f);
		float4 w = rayCastParams.m_viewMatrixInverse * make_float4(camDir, 0.0f);
		float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

		float minInterval = tex2D(rayMinTextureRef, x, y);
		float maxInterval = tex2D(rayMaxTextureRef, x, y);

		//float minInterval = rayCastParams.m_minDepth;
		//float maxInterval = rayCastParams.m_maxDepth;

		//if (minInterval == 0 || minInterval == MINF) minInterval = rayCastParams.m_minDepth;
		//if (maxInterval == 0 || maxInterval == MINF) maxInterval = rayCastParams.m_maxDepth;
		//TODO MATTHIAS: shouldn't this return in the case no interval is found?
		if (minInterval == 0 || minInterval == MINF) return;
		if (maxInterval == 0 || maxInterval == MINF) return;

		minInterval = max(minInterval, rayCastParams.m_minDepth);
		maxInterval = min(maxInterval, rayCastParams.m_maxDepth);

		// debugging 
		//if (maxInterval < minInterval) {
		//	printf("ERROR (%d,%d): [ %f, %f ]\n", x, y, minInterval, maxInterval);
		//}

		rayCastData.traverseCoarseGridSimpleSampleAll(hashData, worldCamPos, worldDir, camDir, make_int3(x,y,1), minInterval, maxInterval);
	} 
}

extern "C" void renderCS(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams) 
{

	const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1)/T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaBindTextureToArray(rayMinTextureRef, rayCastData.d_rayIntervalSplatMinArray, channelDesc);
	cudaBindTextureToArray(rayMaxTextureRef, rayCastData.d_rayIntervalSplatMaxArray, channelDesc);

	renderKernel<<<gridSize, blockSize>>>(hashData, rayCastData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}  


/////////////////////////////////////////////////////////////////////////
// ray interval splatting
/////////////////////////////////////////////////////////////////////////

__global__ void resetRayIntervalSplatKernel(RayCastData data) 
{
	uint idx = blockIdx.x + blockIdx.y * NUM_GROUPS_X;
	data.d_vertexBuffer[idx] = make_float4(MINF);
}

extern "C" void resetRayIntervalSplatCUDA(RayCastData& data, const RayCastParams& params)
{
	const dim3 gridSize(NUM_GROUPS_X, (params.m_maxNumVertices + NUM_GROUPS_X - 1) / NUM_GROUPS_X, 1); // ! todo check if need third dimension?
	const dim3 blockSize(1, 1, 1);

	resetRayIntervalSplatKernel<<<gridSize, blockSize>>>(data);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void rayIntervalSplatKernel(HashDataStruct hashData, RayCastData rayCastData) 
{
	uint idx = blockIdx.x + blockIdx.y * NUM_GROUPS_X;

	const HashEntry& entry = hashData.d_hashCompactified[idx];
	if (entry.ptr != FREE_ENTRY) {
		if (!hashData.isSDFBlockInCameraFrustumApprox(entry.pos)) return;
		const RayCastParams &params = c_rayCastParams;
		const float4x4& viewMatrix = params.m_viewMatrix;

		float3 worldCurrentVoxel = hashData.SDFBlockToWorld(entry.pos);

		float3 MINV = worldCurrentVoxel - c_hashParams.m_virtualVoxelSize / 2.0f;

		float3 maxv = MINV+SDF_BLOCK_SIZE*c_hashParams.m_virtualVoxelSize;

		//float3 proj000 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, MINV.y, MINV.z));
		//float3 proj100 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, MINV.y, MINV.z));
		//float3 proj010 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, maxv.y, MINV.z));
		//float3 proj001 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, MINV.y, maxv.z));
		//float3 proj110 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, maxv.y, MINV.z));
		//float3 proj011 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(MINV.x, maxv.y, maxv.z));
		//float3 proj101 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, MINV.y, maxv.z));
		//float3 proj111 = DepthCameraData::cameraToKinectProj(viewMatrix * make_float3(maxv.x, maxv.y, maxv.z));
		float3 proj000 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, MINV.y, MINV.z));
		float3 proj100 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, MINV.y, MINV.z));
		float3 proj010 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, maxv.y, MINV.z));
		float3 proj001 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, MINV.y, maxv.z));
		float3 proj110 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, maxv.y, MINV.z));
		float3 proj011 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(MINV.x, maxv.y, maxv.z));
		float3 proj101 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, MINV.y, maxv.z));
		float3 proj111 = RayCastData::cameraToDepthProj(viewMatrix * make_float3(maxv.x, maxv.y, maxv.z));

		// Tree Reduction Min
		float3 min00 = fminf(proj000, proj100);
		float3 min01 = fminf(proj010, proj001);
		float3 min10 = fminf(proj110, proj011);
		float3 min11 = fminf(proj101, proj111);

		float3 min0 = fminf(min00, min01);
		float3 min1 = fminf(min10, min11);

		float3 minFinal = fminf(min0, min1);

		// Tree Reduction Max
		float3 max00 = fmaxf(proj000, proj100);
		float3 max01 = fmaxf(proj010, proj001);
		float3 max10 = fmaxf(proj110, proj011);
		float3 max11 = fmaxf(proj101, proj111);

		float3 max0 = fmaxf(max00, max01);
		float3 max1 = fmaxf(max10, max11);

		float3 maxFinal = fmaxf(max0, max1);

		float depth = maxFinal.z;
		if(params.m_splatMinimum == 1) {
			depth = minFinal.z;
		}
		//float depthWorld = DepthCameraData::kinectProjToCameraZ(depth);
		float depthWorld = RayCastData::depthProjToCameraZ(depth);

		//uint addr = idx*4;
		//rayCastData.d_vertexBuffer[addr] = make_float4(maxFinal.x, minFinal.y, depth, depthWorld);
		//rayCastData.d_vertexBuffer[addr+1] = make_float4(minFinal.x, minFinal.y, depth, depthWorld);
		//rayCastData.d_vertexBuffer[addr+2] = make_float4(maxFinal.x, maxFinal.y, depth, depthWorld);
		//rayCastData.d_vertexBuffer[addr+3] = make_float4(minFinal.x, maxFinal.y, depth, depthWorld);
		uint addr = idx*6;
		rayCastData.d_vertexBuffer[addr] = make_float4(maxFinal.x, minFinal.y, depth, depthWorld);
		rayCastData.d_vertexBuffer[addr+1] = make_float4(minFinal.x, minFinal.y, depth, depthWorld);
		rayCastData.d_vertexBuffer[addr+2] = make_float4(maxFinal.x, maxFinal.y, depth, depthWorld);
		rayCastData.d_vertexBuffer[addr+3] = make_float4(minFinal.x, minFinal.y, depth, depthWorld);
		rayCastData.d_vertexBuffer[addr+4] = make_float4(maxFinal.x, maxFinal.y, depth, depthWorld);
		rayCastData.d_vertexBuffer[addr+5] = make_float4(minFinal.x, maxFinal.y, depth, depthWorld);
	}
}

extern "C" void rayIntervalSplatCUDA(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams) 
{

	const dim3 gridSize(NUM_GROUPS_X, (rayCastParams.m_numOccupiedSDFBlocks + NUM_GROUPS_X - 1) / NUM_GROUPS_X, 1);
	const dim3 blockSize(1, 1, 1);

	rayIntervalSplatKernel<<<gridSize, blockSize>>>(hashData, rayCastData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}  

