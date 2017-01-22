
#include "mLibCuda.h"

#define THREADS_PER_BLOCK 128

__global__ void getSiftTransformCU_Kernel(unsigned int curFrameIndex,
	const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform,
	float4x4* d_siftTrajectory, unsigned int curFrameIndexAll,
	const int* d_currNumFilteredMatchesPerImagePair,
	const float4x4* d_filteredTransformsInv, float4x4* d_currIntegrateTrans)
{
	for (int i = (int)curFrameIndex - 1; i >= 0; i--) {
		////debugging
		//printf("[frame %d | %d] to frame %d: #match %d\n", curFrameIndexAll, curFrameIndex, i, d_currNumFilteredMatchesPerImagePair[i]);
		////debugging

		if (d_currNumFilteredMatchesPerImagePair[i] > 0) {
			float4x4 transform;
			const unsigned int idxPrevSiftKnown = curFrameIndexAll - (curFrameIndex - i);
			d_siftTrajectory[curFrameIndexAll] = d_siftTrajectory[idxPrevSiftKnown] * d_filteredTransformsInv[i];

			////debugging
			//printf("\tidxPrevSiftKnown = %d\n", idxPrevSiftKnown);
			//printf("d_filteredTransformsInv[%d]\n", i);
			//d_filteredTransformsInv[i].print();
			//printf("d_siftTrajectory[%d]\n", idxPrevSiftKnown);
			//d_siftTrajectory[idxPrevSiftKnown].print();
			//printf("d_siftTrajectory[%d]\n", curFrameIndexAll);
			//d_siftTrajectory[curFrameIndexAll].print();
			////debugging

			if (lastValidCompleteTransform == 0) {
				transform = d_siftTrajectory[curFrameIndexAll];
			}
			else if (idxPrevSiftKnown < lastValidCompleteTransform) {
				transform = d_completeTrajectory[idxPrevSiftKnown] * d_filteredTransformsInv[i];
			}
			else {
				const float4x4 offset = d_siftTrajectory[lastValidCompleteTransform].getInverse() * d_siftTrajectory[idxPrevSiftKnown];
				transform = d_completeTrajectory[lastValidCompleteTransform] * offset * d_filteredTransformsInv[i];
			}

			d_currIntegrateTrans[0] = transform;

			////debugging
			//printf("transform\n");
			//transform.print();
			////debugging

			break;
		}
	}
}

extern "C" void computeSiftTransformCU(const float4x4* d_currFilteredTransformsInv, const int* d_currNumFilteredMatchesPerImagePair,
	const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform,
	float4x4* d_siftTrajectory, unsigned int curFrameIndexAll, unsigned int curFrameIndex, float4x4* d_currIntegrateTrans)
{
	if (curFrameIndex == 0) return;

	getSiftTransformCU_Kernel << <1, 1 >> >(curFrameIndex,
		d_completeTrajectory, lastValidCompleteTransform,
		d_siftTrajectory, curFrameIndexAll,
		d_currNumFilteredMatchesPerImagePair, d_currFilteredTransformsInv,
		d_currIntegrateTrans);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void updateTrajectoryCU_Kernel(const float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
	float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	const float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory,
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
	const float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, float4x4* d_completeTrajectory, unsigned int numCompleteTransforms,
	const float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory, unsigned int numLocalTrajectories,
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



__global__ void initNextGlobalTransformCU_Kernel(float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, unsigned int initGlobalIdx,
	float4x4* d_localTrajectories, unsigned int lastValidLocal, unsigned int numLocalTransformsPerTrajectory)
{
	//if (d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - 1].m11 == MINF) {
	//	printf("[ERROR initNextGlobalTransformCU_Kernel]: d_localTrajectories[%d*%d-1] INVALID!\n", numGlobalTransforms, numLocalTransformsPerTrajectory);//debugging
	//}
	//d_globalTrajectory[numGlobalTransforms] = d_globalTrajectory[initGlobalIdx] * d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - 1];

	if (d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - (numLocalTransformsPerTrajectory - lastValidLocal)].m11 == MINF) {
		printf("[ERROR initNextGlobalTransformCU_Kernel]: d_localTrajectories[%d*%d-1] INVALID!\n", numGlobalTransforms, numLocalTransformsPerTrajectory);//debugging
	}
	d_globalTrajectory[numGlobalTransforms] = d_globalTrajectory[initGlobalIdx] * d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - (numLocalTransformsPerTrajectory - lastValidLocal)];
}

extern "C" void initNextGlobalTransformCU(
	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms, unsigned int initGlobalIdx,
	float4x4* d_localTrajectories, unsigned int lastValidLocal, unsigned int numLocalTransformsPerTrajectory)
{
	initNextGlobalTransformCU_Kernel <<< 1, 1 >>>(
		d_globalTrajectory, numGlobalTransforms, initGlobalIdx,
		d_localTrajectories, lastValidLocal, numLocalTransformsPerTrajectory);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////update with local opt result
//__global__ void refineNextGlobalTransformCU_Kernel(float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
//	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory, unsigned int lastValidLocal)
//{
//	//d_globalTrajectory[numGlobalTransforms] already init from above
//	d_globalTrajectory[numGlobalTransforms] = d_globalTrajectory[numGlobalTransforms] * d_localTrajectories[numGlobalTransforms*numLocalTransformsPerTrajectory - 1];
//}
//
//extern "C" void refineNextGlobalTransformCU(
//	float4x4* d_globalTrajectory, unsigned int numGlobalTransforms,
//	unsigned int initGlobalIdx,
//	float4x4* d_localTrajectories, unsigned int numLocalTransformsPerTrajectory)
//{
//	initNextGlobalTransformCU_Kernel <<< 1, 1 >>>(
//		d_globalTrajectory, numGlobalTransforms, initGlobalIdx,
//		d_localTrajectories, numLocalTransformsPerTrajectory);
//
//#ifdef _DEBUG
//	cutilSafeCall(cudaDeviceSynchronize());
//	cutilCheckMsg(__FUNCTION__);
//#endif
//}


