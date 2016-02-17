
#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"


#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"

#define T_PER_BLOCK 8

struct SDFBlockDesc {
	int3 pos;
	int ptr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Streaming from GPU to CPU: copies only selected blocks/hashEntries to the CPU if outside of the frustum //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


//-------------------------------------------------------
// Pass 1: Find all SDFBlocks that have to be transfered
//-------------------------------------------------------

__global__ void integrateFromGlobalHashPass1Kernel(HashDataStruct hashData, uint start, float radius, float3 cameraPosition, uint* d_outputCounter, SDFBlockDesc* d_output) 
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x + start;
	//uint bucketID = start+groupthreads*(GID.x + GID.y * NUM_GROUPS_X)+GI;

	const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	if (bucketID < hashParams.m_hashNumBuckets*HASH_BUCKET_SIZE) {

		//HashEntry entry = getHashEntry(g_Hash, bucketID);
		HashEntry& entry = hashData.d_hash[bucketID];

		float3 posWorld = hashData.SDFBlockToWorld(entry.pos);
		float d = length(posWorld - cameraPosition);

		if (entry.ptr != FREE_ENTRY && d >= radius) {
		
			// Write
			SDFBlockDesc d;
			d.pos = entry.pos;
			d.ptr = entry.ptr;

			#ifndef HANDLE_COLLISIONS
				uint addr = atomicAdd(&d_outputCounter[0], 1);
				d_output[addr] = d;
				hashData.appendHeap(entry.ptr/linBlockSize);
				hashData.deleteHashEntry(bucketID);
			#endif
			#ifdef HANDLE_COLLISIONS
				//if there is an offset or hash doesn't belong to the bucket (linked list)
				if (entry.offset != 0 || hashData.computeHashPos(entry.pos) != bucketID / HASH_BUCKET_SIZE) {
					
					if (hashData.deleteHashEntryElement(entry.pos)) {
						hashData.appendHeap(entry.ptr/linBlockSize);
						uint addr = atomicAdd(&d_outputCounter[0], 1);
						d_output[addr] = d;
					}
				} else {
					uint addr = atomicAdd(&d_outputCounter[0], 1);
					d_output[addr] = d;
					hashData.appendHeap(entry.ptr/linBlockSize);
					hashData.deleteHashEntry(entry);
				}
			#endif
		}
	}
}

extern "C" void integrateFromGlobalHashPass1CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint threadsPerPart, uint start, float radius, const float3& cameraPosition, uint* d_outputCounter, SDFBlockDesc* d_output)
{
	const dim3 gridSize((threadsPerPart + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	if (threadsPerPart > 0) {
		integrateFromGlobalHashPass1Kernel<<<gridSize, blockSize>>>(hashData, start, radius, cameraPosition, d_outputCounter, d_output);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


//-------------------------------------------------------
// Pass 2: Copy SDFBlocks to output buffer
//-------------------------------------------------------


__global__ void integrateFromGlobalHashPass2Kernel(HashDataStruct hashData, const SDFBlockDesc* d_SDFBlockDescs, Voxel* d_output, unsigned int nSDFBlocks)
{
	const uint idxBlock = blockIdx.x;

	if (idxBlock < nSDFBlocks) {

		const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
		const uint idxInBlock = threadIdx.x;
		const SDFBlockDesc& desc = d_SDFBlockDescs[idxBlock];

		// Copy SDF block to CPU
		d_output[idxBlock*linBlockSize + idxInBlock] = hashData.d_SDFBlocks[desc.ptr + idxInBlock];

		//// Reset SDF Block
		hashData.deleteVoxel(desc.ptr + idxInBlock);
	}
}

extern "C" void integrateFromGlobalHashPass2CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint threadsPerPart, const SDFBlockDesc* d_SDFBlockDescs, Voxel* d_output, unsigned int nSDFBlocks)
{
	const uint threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
	const dim3 gridSize(threadsPerPart, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	if (threadsPerPart > 0) {
		integrateFromGlobalHashPass2Kernel<<<gridSize, blockSize>>>(hashData, d_SDFBlockDescs, d_output, nSDFBlocks);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



///////////////////////////////////////////////////////////////////////
// Streaming from CPU to GPU: copies an entire chunk back to the GPU //
///////////////////////////////////////////////////////////////////////



//-------------------------------------------------------
// Pass 1: Allocate memory
//-------------------------------------------------------

__global__ void  chunkToGlobalHashPass1Kernel(HashDataStruct hashData, uint numSDFBlockDescs, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks)
{
	const unsigned int bucketID = blockIdx.x*blockDim.x + threadIdx.x;
	const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

	if (bucketID < numSDFBlockDescs)	{  
		
		uint ptr = hashData.d_heap[heapCountPrev - bucketID]*linBlockSize;	//mass alloc

		HashEntry entry;
		entry.pos = d_SDFBlockDescs[bucketID].pos;
		entry.offset = 0;
		entry.ptr = ptr;

		//TODO MATTHIAS check this: if this is false, we have a memory leak... -> we need to make sure that this works! (also the next kernel will randomly fill memory)
		bool ok = hashData.insertHashEntry(entry);
	}
}

extern "C" void chunkToGlobalHashPass1CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint numSDFBlockDescs, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks)
{
	const dim3 gridSize((numSDFBlockDescs + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

	if (numSDFBlockDescs > 0) {
		chunkToGlobalHashPass1Kernel<<<gridSize, blockSize>>>(hashData, numSDFBlockDescs, heapCountPrev, d_SDFBlockDescs, d_SDFBlocks);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

//-------------------------------------------------------
// Pass 2: Copy input to SDFBlocks
//-------------------------------------------------------

__global__ void chunkToGlobalHashPass2Kernel(HashDataStruct hashData, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks)
{
	const uint blockID = blockIdx.x;
	const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
		
	uint ptr = hashData.d_heap[heapCountPrev-blockID]*linBlockSize;
	hashData.d_SDFBlocks[ptr + threadIdx.x] = d_SDFBlocks[blockIdx.x*blockDim.x + threadIdx.x];
	//hashData.d_SDFBlocks[ptr + threadIdx.x].color = make_uchar3(255,0,0);
}


extern "C" void chunkToGlobalHashPass2CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint numSDFBlockDescs, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks)
{
	const uint threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
	const dim3 gridSize(numSDFBlockDescs, 1);
	const dim3 blockSize(threadsPerBlock, 1);

	if (numSDFBlockDescs > 0) {
		chunkToGlobalHashPass2Kernel<<<gridSize, blockSize>>>(hashData, heapCountPrev, d_SDFBlockDescs, d_SDFBlocks);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}