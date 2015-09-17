
#include "mLibCuda.h"

#define THREADS_PER_BLOCK 128

__global__ void updateTrajectoryCU_Kernel(float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransforms)
{
	const unsigned int idxComplete = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int submapSize = numLocalTransforms - 1;

	if (idxComplete < numCompleteTransforms) {
		const unsigned int idxGlobal = idxComplete / submapSize;
		const unsigned int idxLocal = idxComplete % submapSize;

		d_globalTrajectory[idxComplete] = d_globalTrajectory[idxGlobal] * d_localTrajectories[idxGlobal * numLocalTransforms + idxLocal];
	}
}

extern "C" void updateTrajectoryCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransforms, unsigned int numLocalTrajectories) 
{
	const unsigned int N = numCompleteTransforms;

	updateTrajectoryCU_Kernel << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(
		d_globalTrajectory, numGlobalTransforms,
		d_completeTrajectory, numCompleteTransforms,
		d_localTrajectories, numLocalTransforms);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

extern "C" void initNextGlobalTransformCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_localTrajectories, unsigned int numLocalTransforms, unsigned int numLocalTrajectories
	)
{
	//globalTrajectory.push_back(globalTrajectory.back() *currentLocalTrajectory.back()); //initialize next one
}