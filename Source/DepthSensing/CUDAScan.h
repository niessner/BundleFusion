#pragma once

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include "cudaUtil.h"

class CUDAScan
{
	public:

		CUDAScan();
		~CUDAScan();

		unsigned int prefixSum(unsigned int numElements, int* d_input, int* d_output);

		unsigned int getMaxScanSize();

	private:

		unsigned int m_BucketSize;
		unsigned int m_BucketBlockSize;

		unsigned int m_NumBuckets;
		unsigned int m_NumBucketBlocks;

		int* m_pBucketResults;
		int* m_pBucketBlockResults;
};
