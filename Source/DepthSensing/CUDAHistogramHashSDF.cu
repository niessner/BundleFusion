

#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"


#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"

#define T_PER_BLOCK 8



__global__ void computeHistrogramKernel(unsigned int* d_data, HashDataStruct hashData)
{
	const HashParams& hashParams = c_hashParams;
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx < hashParams.m_hashNumBuckets) {
		unsigned int numEntiresPerBucket = hashData.getNumHashEntriesPerBucket(idx);
		atomicAdd(&d_data[numEntiresPerBucket], 1);
		unsigned int listLength = hashData.getNumHashLinkedList(idx);
		listLength = min(listLength, hashParams.m_hashMaxCollisionLinkedListSize);
		atomicAdd(&d_data[listLength + HASH_BUCKET_SIZE + 1], 1);
	}
}


extern "C" void computeHistogramCUDA(unsigned int* d_data, const HashDataStruct& hashData, const HashParams& hashParams) 
{
	const dim3 gridSize((hashParams.m_hashNumBuckets + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
	const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);


	computeHistrogramKernel<<<gridSize, blockSize>>>(d_data, hashData);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}







__global__ void resetHistrogramKernel(unsigned int* d_data)
{
	d_data[blockIdx.x] = 0;
}


extern "C" void resetHistrogramCUDA(unsigned int* d_data, unsigned int numValues) 
{
	const dim3 gridSize(numValues, 1, 1);
	const dim3 blockSize(1,1,1);

	resetHistrogramKernel<<<gridSize, blockSize>>>(d_data);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}