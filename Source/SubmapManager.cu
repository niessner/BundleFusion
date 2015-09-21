
#include "mLibCuda.h"

#define THREADS_PER_BLOCK 128

__global__ void updateTrajectoryCU_Kernel(float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory,
	int* d_imageInvalidateList)
{
	const unsigned int idxComplete = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int submapSize = numLocalTransformsPerTrajectory - 1;

	if (idxComplete < numCompleteTransforms) {
		const unsigned int idxGlobal = idxComplete / submapSize;
		const unsigned int idxLocal = idxComplete % submapSize;

		if (d_imageInvalidateList[idxComplete] == 0) {
			d_completeTrajectory[idxComplete].setValue(MINF);
		}
		else {
			d_completeTrajectory[idxComplete] = d_globalTrajectory[idxGlobal] * d_localTrajectories[idxGlobal * numLocalTransformsPerTrajectory + idxLocal];
		}
	}
}

extern "C" void updateTrajectoryCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory, unsigned int numLocalTrajectories,
	int* d_imageInvalidateList) 
{
	const unsigned int N = numCompleteTransforms;

	updateTrajectoryCU_Kernel <<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(
		d_globalTrajectory, numGlobalTransforms,
		d_completeTrajectory, numCompleteTransforms,
		d_localTrajectories, numLocalTransformsPerTrajectory,
		d_imageInvalidateList);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



__global__ void initNextGlobalTransformCU_Kernel(float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory)
{
	d_globalTrajectory[numGlobalTransforms] = d_globalTrajectory[numGlobalTransforms - 1] * d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - 1];
}

extern "C" void initNextGlobalTransformCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory)
{
	initNextGlobalTransformCU_Kernel <<< 1, 1 >>>(
		d_globalTrajectory, numGlobalTransforms,
		d_localTrajectories, numLocalTransformsPerTrajectory);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}



