
#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"

#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"

#define T_PER_BLOCK 8

texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;

extern "C" void bindInputDepthColorTextures(const DepthCameraData& depthCameraData, unsigned int width, unsigned int height) 
{
	//cutilSafeCall(cudaBindTextureToArray(depthTextureRef, depthCameraData.d_depthArray, depthCameraData.h_depthChannelDesc));
	//cutilSafeCall(cudaBindTextureToArray(colorTextureRef, depthCameraData.d_colorArray, depthCameraData.h_colorChannelDesc));

	cutilSafeCall(cudaBindTexture2D(0, depthTextureRef, depthCameraData.d_depthData, depthTextureRef.channelDesc, width, height, sizeof(float)*width));
	cutilSafeCall(cudaBindTexture2D(0, colorTextureRef, depthCameraData.d_colorData, colorTextureRef.channelDesc, width, height, sizeof(uchar4)*width));

	depthTextureRef.filterMode = cudaFilterModePoint;
	colorTextureRef.filterMode = cudaFilterModePoint;
}

__global__ void resetHeapKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx == 0) {
		hashData.d_heapCounter[0] = hashParams.m_numSDFBlocks - 1;	//points to the last element of the array
	}
	
	if (idx < hashParams.m_numSDFBlocks) {

		hashData.d_heap[idx] = hashParams.m_numSDFBlocks - idx - 1;
		uint blockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
		uint base_idx = idx * blockSize;
		for (uint i = 0; i < blockSize; i++) {
			hashData.deleteVoxel(base_idx+i);
		}
	}
}

__global__ void resetHashKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		hashData.deleteHashEntry(hashData.d_hash[idx]);
		hashData.deleteHashEntry(hashData.d_hashCompactified[idx]);
	}
}


__global__ void resetHashBucketMutexKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets) {
		hashData.d_hashBucketMutex[idx] = FREE_ENTRY;
	}
}

extern "C" void resetCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	{
		//resetting the heap and SDF blocks
		const dim3 gridSize((hashParams.m_numSDFBlocks + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHeapKernel<<<gridSize, blockSize>>>(hashData);


		#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
		#endif

	}

	{
		//resetting the hash
		const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHashKernel<<<gridSize, blockSize>>>(hashData);

		#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
		#endif
	}

	{
		//resetting the mutex
		const dim3 gridSize((hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
		const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

		resetHashBucketMutexKernel<<<gridSize, blockSize>>>(hashData);

		#ifdef _DEBUG
			cutilSafeCall(cudaDeviceSynchronize());
			cutilCheckMsg(__FUNCTION__);
		#endif
	}


}

extern "C" void resetHashBucketMutexCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const dim3 gridSize((hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	resetHashBucketMutexKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__device__
unsigned int linearizeChunkPos(const int3& chunkPos)
{
	int3 p = chunkPos-c_hashParams.m_streamingMinGridPos;
	return  p.z * c_hashParams.m_streamingGridDimensions.x * c_hashParams.m_streamingGridDimensions.y +
			p.y * c_hashParams.m_streamingGridDimensions.x +
			p.x;
}

__device__
int3 worldToChunks(const float3& posWorld)
{
	float3 p;
	p.x = posWorld.x/c_hashParams.m_streamingVoxelExtents.x;
	p.y = posWorld.y/c_hashParams.m_streamingVoxelExtents.y;
	p.z = posWorld.z/c_hashParams.m_streamingVoxelExtents.z;

	float3 s;
	s.x = (float)sign(p.x);
	s.y = (float)sign(p.y);
	s.z = (float)sign(p.z);

	return make_int3(p+s*0.5f);
}

__device__
bool isSDFBlockStreamedOut(const int3& sdfBlock, const HashDataStruct& hashData, const unsigned int* d_bitMask)	//TODO MATTHIAS (-> move to HashData)
{
	if (!d_bitMask) return false;	//TODO can statically disable streaming??


	float3 posWorld = hashData.virtualVoxelPosToWorld(hashData.SDFBlockToVirtualVoxelPos(sdfBlock)); // sdfBlock is assigned to chunk by the bottom right sample pos

	uint index = linearizeChunkPos(worldToChunks(posWorld));
	uint nBitsInT = 32;
	return ((d_bitMask[index/nBitsInT] & (0x1 << (index%nBitsInT))) != 0x0);
}

__global__ void allocKernel(HashDataStruct hashData, DepthCameraData cameraData, const unsigned int* d_bitMask) 
{
	const HashParams& hashParams = c_hashParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;

	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x < cameraParams.m_imageWidth && y < cameraParams.m_imageHeight)
	{

		float d = tex2D(depthTextureRef, x, y);
		
		//if (d == MINF || d < cameraParams.m_sensorDepthWorldMin || d > cameraParams.m_sensorDepthWorldMax)	return;
		if (d == MINF || d == 0.0f)	return;


		if (d >= hashParams.m_maxIntegrationDistance) return;

		float t = hashData.getTruncation(d);
		float minDepth = min(hashParams.m_maxIntegrationDistance, d-t);
		float maxDepth = min(hashParams.m_maxIntegrationDistance, d+t);
		if (minDepth >= maxDepth) return;

		float3 rayMin = DepthCameraData::kinectDepthToSkeleton(x, y, minDepth);
		rayMin = hashParams.m_rigidTransform * rayMin;
		float3 rayMax = DepthCameraData::kinectDepthToSkeleton(x, y, maxDepth);
		rayMax = hashParams.m_rigidTransform * rayMax;

		
		float3 rayDir = normalize(rayMax - rayMin);
	
		int3 idCurrentVoxel = hashData.worldToSDFBlock(rayMin);
		int3 idEnd = hashData.worldToSDFBlock(rayMax);
		
		float3 step = make_float3(sign(rayDir));
		float3 boundaryPos = hashData.SDFBlockToWorld(idCurrentVoxel+make_int3(clamp(step, 0.0, 1.0f)))-0.5f*hashParams.m_virtualVoxelSize;
		float3 tMax = (boundaryPos-rayMin)/rayDir;
		float3 tDelta = (step*SDF_BLOCK_SIZE*hashParams.m_virtualVoxelSize)/rayDir;
		int3 idBound = make_int3(make_float3(idEnd)+step);

		//#pragma unroll
		//for(int c = 0; c < 3; c++) {
		//	if (rayDir[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
		//	if (boundaryPos[c] - rayMin[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
		//}
		if (rayDir.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }
		if (boundaryPos.x - rayMin.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }

		if (rayDir.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }
		if (boundaryPos.y - rayMin.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }

		if (rayDir.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }
		if (boundaryPos.z - rayMin.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }


		unsigned int iter = 0; // iter < g_MaxLoopIterCount
		unsigned int g_MaxLoopIterCount = 1024;	//TODO MATTHIAS MOVE TO GLOBAL APP STATE
#pragma unroll 1
		while(iter < g_MaxLoopIterCount) {

			//check if it's in the frustum and not checked out
			if (hashData.isSDFBlockInCameraFrustumApprox(idCurrentVoxel) && !isSDFBlockStreamedOut(idCurrentVoxel, hashData, d_bitMask)) {		
				hashData.allocBlock(idCurrentVoxel);
			}

			// Traverse voxel grid
			if(tMax.x < tMax.y && tMax.x < tMax.z)	{
				idCurrentVoxel.x += step.x;
				if(idCurrentVoxel.x == idBound.x) return;
				tMax.x += tDelta.x;
			}
			else if(tMax.z < tMax.y) {
				idCurrentVoxel.z += step.z;
				if(idCurrentVoxel.z == idBound.z) return;
				tMax.z += tDelta.z;
			}
			else	{
				idCurrentVoxel.y += step.y;
				if(idCurrentVoxel.y == idBound.y) return;
				tMax.y += tDelta.y;
			}

			iter++;
		}
	}
}

extern "C" void allocCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask) 
{
	const dim3 gridSize((depthCameraParams.m_imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (depthCameraParams.m_imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	allocKernel<<<gridSize, blockSize>>>(hashData, depthCameraData, d_bitMask);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}



__global__ void fillDecisionArrayKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		hashData.d_hashDecision[idx] = 0;
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos)) 
			{
				hashData.d_hashDecision[idx] = 1;	//yes
			}
		}
	}
}

extern "C" void fillDecisionArrayCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	fillDecisionArrayKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}

__global__ void compactifyHashKernel(HashDataStruct hashData) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hashDecision[idx] == 1) {
			hashData.d_hashCompactified[hashData.d_hashDecisionPrefix[idx]-1] = hashData.d_hash[idx];
		}
	}
}

extern "C" void compactifyHashCUDA(HashDataStruct& hashData, const HashParams& hashParams) 
{
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	compactifyHashKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
//#define COMPACTIFY_HASH_SIMPLE
__global__ void compactifyHashAllInOneKernel(HashDataStruct hashData)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
#ifdef COMPACTIFY_HASH_SIMPLE
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos))
			{
				int addr = atomicAdd(hashData.d_hashCompactifiedCounter, 1);
				hashData.d_hashCompactified[addr] = hashData.d_hash[idx];
			}
		}
	}
#else	
	__shared__ int localCounter;
	if (threadIdx.x == 0) localCounter = 0;
	__syncthreads();
	
	int addrLocal = -1;
	if (idx < hashParams.m_hashNumBuckets * HASH_BUCKET_SIZE) {
		if (hashData.d_hash[idx].ptr != FREE_ENTRY) {
			if (hashData.isSDFBlockInCameraFrustumApprox(hashData.d_hash[idx].pos))
			{
				addrLocal = atomicAdd(&localCounter, 1);
			}
		}
	}

	__syncthreads();

	__shared__ int addrGlobal;
	if (threadIdx.x == 0 && localCounter > 0) {
		addrGlobal = atomicAdd(hashData.d_hashCompactifiedCounter, localCounter);
	}
	__syncthreads();

	if (addrLocal != -1) {
		const unsigned int addr = addrGlobal + addrLocal;
		hashData.d_hashCompactified[addr] = hashData.d_hash[idx];
	}
#endif
}

extern "C" unsigned int compactifyHashAllInOneCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const unsigned int threadsPerBlock = COMPACTIFY_HASH_THREADS_PER_BLOCK;
	const dim3 gridSize((HASH_BUCKET_SIZE * hashParams.m_hashNumBuckets + threadsPerBlock - 1) / threadsPerBlock, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	cutilSafeCall(cudaMemset(hashData.d_hashCompactifiedCounter, 0, sizeof(int)));
	compactifyHashAllInOneKernel << <gridSize, blockSize >> >(hashData);
	unsigned int res = 0;
	cutilSafeCall(cudaMemcpy(&res, hashData.d_hashCompactifiedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	return res;
}


inline __device__ float4 bilinearFilterColor(const float2& screenPos) {
	const DepthCameraParams& cameraParams = c_depthCameraParams;
	const int imageWidth = cameraParams.m_imageWidth;
	const int imageHeight = cameraParams.m_imageHeight;
	const int2 p00 = make_int2(screenPos.x+0.5f, screenPos.y+0.5f);
	const int2 dir = sign(make_float2(screenPos.x - p00.x, screenPos.y - p00.y));

	const int2 p01 = p00 + make_int2(0.0f, dir.y);
	const int2 p10 = p00 + make_int2(dir.x, 0.0f);
	const int2 p11 = p00 + make_int2(dir.x, dir.y);

	const float alpha = (screenPos.x - p00.x)*dir.x;
	const float beta  = (screenPos.y - p00.y)*dir.y;

	float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if (p00.x >= 0 && p00.x < imageWidth && p00.y >= 0 && p00.y < imageHeight) { uchar4 v00_uc = tex2D(colorTextureRef, p00.x, p00.y); float4 v00 = make_float4(v00_uc.x, v00_uc.y, v00_uc.z, v00_uc.w);	if (v00.x != MINF) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
	if (p10.x >= 0 && p10.x < imageWidth && p10.y >= 0 && p10.y < imageHeight) { uchar4 v10_uc = tex2D(colorTextureRef, p10.x, p10.y); float4 v10 = make_float4(v10_uc.x, v10_uc.y, v10_uc.z, v10_uc.w);    if (v10.x != MINF) { s0 += alpha *v10; w0 += alpha; } }

	float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if (p01.x >= 0 && p01.x < imageWidth && p01.y >= 0 && p01.y < imageHeight) { uchar4 v01_uc = tex2D(colorTextureRef, p01.x, p01.y); float4 v01 = make_float4(v01_uc.x, v01_uc.y, v01_uc.z, v01_uc.w);    if (v01.x != MINF) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
	if (p11.x >= 0 && p11.x < imageWidth && p11.y >= 0 && p11.y < imageHeight) { uchar4 v11_uc = tex2D(colorTextureRef, p11.x, p11.y); float4 v11 = make_float4(v11_uc.x, v11_uc.y, v11_uc.z, v11_uc.w);    if (v11.x != MINF) { s1 += alpha *v11; w1 += alpha; } }

	const float4 p0 = s0/w0;
	const float4 p1 = s1/w1;

	float4 ss = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return make_float4(MINF, MINF, MINF, MINF);
}

template<bool deIntegrate = false>
__global__ void integrateDepthMapKernel(HashDataStruct hashData, DepthCameraData cameraData) {
	const HashParams& hashParams = c_hashParams;
	const DepthCameraParams& cameraParams = c_depthCameraParams;

	const HashEntry& entry = hashData.d_hashCompactified[blockIdx.x];

	int3 pi_base = hashData.SDFBlockToVirtualVoxelPos(entry.pos);

	uint i = threadIdx.x;	//inside of an SDF block
	int3 pi = pi_base + make_int3(hashData.delinearizeVoxelIndex(i));
	float3 pf = hashData.virtualVoxelPosToWorld(pi);

	pf = hashParams.m_rigidTransformInverse * pf;
	uint2 screenPos = make_uint2(cameraData.cameraToKinectScreenInt(pf));


	if (screenPos.x < cameraParams.m_imageWidth && screenPos.y < cameraParams.m_imageHeight) {	//on screen

		//float depth = g_InputDepth[screenPos];
		float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);
		float4 color  = make_float4(MINF, MINF, MINF, MINF);
		if (cameraData.d_colorData) {
			uchar4 color_uc = tex2D(colorTextureRef, screenPos.x, screenPos.y);
			color = make_float4(color_uc.x, color_uc.y, color_uc.z, color_uc.w);
			//color = bilinearFilterColor(cameraData.cameraToKinectScreenFloat(pf));
		}

		if (color.x != MINF && depth != MINF) { // valid depth and color
		//if (depth != MINF) {	//valid depth

			if (depth < hashParams.m_maxIntegrationDistance) {
				float depthZeroOne = cameraData.cameraToKinectProjZ(depth);

				float sdf = depth - pf.z;
				float truncation = hashData.getTruncation(depth);
				//if (sdf > -truncation) 
				if (abs(sdf) < truncation)
				{
					if (sdf >= 0.0f) {
						sdf = fminf(truncation, sdf);
					} else {
						sdf = fmaxf(-truncation, sdf);
					}

					float weightUpdate = max(hashParams.m_integrationWeightSample * 1.5f * (1.0f-depthZeroOne), 1.0f);
					weightUpdate = 1.0f;	//TODO remove that again

					Voxel curr;	//construct current voxel
					curr.sdf = sdf;
					curr.weight = weightUpdate;

					if (cameraData.d_colorData) {
						curr.color = make_uchar4(color.x, color.y, color.z, 255);
					} else {
						curr.color = make_uchar4(0,255,0,0);
					}

					uint idx = entry.ptr + i;
					
					const Voxel& oldVoxel = hashData.d_SDFBlocks[idx];
					Voxel newVoxel;

					float3 oldColor = make_float3(oldVoxel.color.x, oldVoxel.color.y, oldVoxel.color.z);
					float3 currColor = make_float3(curr.color.x, curr.color.y, curr.color.z);

					if (!deIntegrate) {	//integration
						//hashData.combineVoxel(hashData.d_SDFBlocks[idx], curr, newVoxel);
						float3 res;
						if (oldVoxel.weight == 0) res = currColor;
						//else res = (currColor + oldColor) / 2;
						else res = 0.2f * currColor + 0.8f * oldColor;
						//float3 res = (currColor*curr.weight + oldColor*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
						res = make_float3(round(res.x), round(res.y), round(res.z));
						res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
						//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
						newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
						newVoxel.sdf = (curr.sdf*curr.weight + oldVoxel.sdf*oldVoxel.weight) / (curr.weight + oldVoxel.weight);
						newVoxel.weight = min((float)c_hashParams.m_integrationWeightMax, curr.weight + oldVoxel.weight);
					}
					else {				//deintegration
						//float3 res = 2 * c0 - c1;
						float3 res = (oldColor*oldVoxel.weight - currColor*curr.weight) / (oldVoxel.weight - curr.weight);
						res = make_float3(round(res.x), round(res.y), round(res.z));
						res = fmaxf(make_float3(0.0f), fminf(res, make_float3(254.5f)));
						//newVoxel.color.x = (uchar)(res.x + 0.5f);	newVoxel.color.y = (uchar)(res.y + 0.5f);	newVoxel.color.z = (uchar)(res.z + 0.5f);
						newVoxel.color = make_uchar4(res.x, res.y, res.z, 255);
						newVoxel.sdf = (oldVoxel.sdf*oldVoxel.weight - curr.sdf*curr.weight) / (oldVoxel.weight - curr.weight);
						newVoxel.weight = max(0.0f, oldVoxel.weight - curr.weight);
						if (newVoxel.weight <= 0.001f) {
							newVoxel.sdf = 0.0f;
							newVoxel.color = make_uchar4(0,0,0,0);
							newVoxel.weight = 0.0f;
						}
					}

					hashData.d_SDFBlocks[idx] = newVoxel;
				}
			}
		}
	}
}


extern "C" void integrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	integrateDepthMapKernel<false> <<<gridSize, blockSize>>>(hashData, depthCameraData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void deIntegrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	integrateDepthMapKernel<true> <<<gridSize, blockSize >>>(hashData, depthCameraData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void starveVoxelsKernel(HashDataStruct hashData) {

	const uint idx = blockIdx.x;
	const HashEntry& entry = hashData.d_hashCompactified[idx];

	//is typically exectued only every n'th frame
	int weight = hashData.d_SDFBlocks[entry.ptr + threadIdx.x].weight;
	weight = max(0, weight-1);	
	hashData.d_SDFBlocks[entry.ptr + threadIdx.x].weight = weight;
}

extern "C" void starveVoxelsKernelCUDA(HashDataStruct& hashData, const HashParams& hashParams)
{
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	starveVoxelsKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


//__shared__ float	shared_MinSDF[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint		shared_MaxWeight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];


__global__ void garbageCollectIdentifyKernel(HashDataStruct hashData) {

	const unsigned int hashIdx = blockIdx.x;
	const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
	
	//uint h = hashData.computeHashPos(entry.pos);
	//hashData.d_hashDecision[hashIdx] = 1;
	//if (hashData.d_hashBucketMutex[h] == LOCK_ENTRY)	return;

	//if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before
	//const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	const unsigned int idx0 = entry.ptr + 2*threadIdx.x+0;
	const unsigned int idx1 = entry.ptr + 2*threadIdx.x+1;

	Voxel v0 = hashData.d_SDFBlocks[idx0];
	Voxel v1 = hashData.d_SDFBlocks[idx1];

	//if (v0.weight == 0)	v0.sdf = PINF;
	//if (v1.weight == 0)	v1.sdf = PINF;

	//shared_MinSDF[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
	shared_MaxWeight[threadIdx.x] = max(v0.weight, v1.weight);
		
#pragma unroll 1
	for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
		__syncthreads();
		if ((threadIdx.x  & (stride-1)) == (stride-1)) {
			//shared_MinSDF[threadIdx.x] = min(shared_MinSDF[threadIdx.x-stride/2], shared_MinSDF[threadIdx.x]);
			shared_MaxWeight[threadIdx.x] = max(shared_MaxWeight[threadIdx.x-stride/2], shared_MaxWeight[threadIdx.x]);
		}
	}

	__syncthreads();

	if (threadIdx.x == blockDim.x - 1) {
		//float minSDF = shared_MinSDF[threadIdx.x];
		uint maxWeight = shared_MaxWeight[threadIdx.x];

		//float t = hashData.getTruncation(c_depthCameraParams.m_sensorDepthWorldMax);
		//if (minSDF >= t || maxWeight == 0) {
		if (maxWeight == 0) {
			hashData.d_hashDecision[hashIdx] = 1;
		} else {
			hashData.d_hashDecision[hashIdx] = 0; 
		}
	}
}
 
extern "C" void garbageCollectIdentifyCUDA(HashDataStruct& hashData, const HashParams& hashParams) {
	
	const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;
	const dim3 gridSize(hashParams.m_numOccupiedBlocks, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	garbageCollectIdentifyKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


__global__ void garbageCollectFreeKernel(HashDataStruct hashData) {

	//const uint hashIdx = blockIdx.x;
	const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;


	if (hashIdx < c_hashParams.m_numOccupiedBlocks && hashData.d_hashDecision[hashIdx] != 0) {	//decision to delete the hash entry

		const HashEntry& entry = hashData.d_hashCompactified[hashIdx];
		//if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

		if (hashData.deleteHashEntryElement(entry.pos)) {	//delete hash entry from hash (and performs heap append)
			const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

			#pragma unroll 1
			for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
				hashData.deleteVoxel(entry.ptr + i);
			}
		}
	}
}


extern "C" void garbageCollectFreeCUDA(HashDataStruct& hashData, const HashParams& hashParams) {
	
	const unsigned int threadsPerBlock = T_PER_BLOCK*T_PER_BLOCK;
	const dim3 gridSize((hashParams.m_numOccupiedBlocks + threadsPerBlock - 1) / threadsPerBlock, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	garbageCollectFreeKernel<<<gridSize, blockSize>>>(hashData);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

