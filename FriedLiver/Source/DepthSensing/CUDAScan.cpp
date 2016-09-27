#include "stdafx.h"
#include "CUDAScan.h"
#include <assert.h> 

extern "C" void prefixSumStub(int* d_input, int* d_bucketResults, int* d_bucketBlockResults, int* d_output, unsigned int numElements, unsigned int bucketSize, unsigned int bucketBlockSize, unsigned int numBuckets, unsigned int numBucketBlocks);

CUDAScan::CUDAScan()    
{
	const unsigned int mySize = 512;	// about 130 mio, keep consistent with GPU
	
	m_BucketSize = mySize;
	m_BucketBlockSize = mySize;
	m_NumBuckets = mySize*mySize;
	m_NumBucketBlocks = mySize;

	m_pBucketResults = NULL;
	m_pBucketBlockResults = NULL;

	cutilSafeCall(cudaMalloc(&m_pBucketResults,		 sizeof(int) * m_NumBuckets));
	cutilSafeCall(cudaMalloc(&m_pBucketBlockResults, sizeof(int) * m_NumBucketBlocks));
}

CUDAScan::~CUDAScan()    
{
	cutilSafeCall(cudaFree(m_pBucketResults));
	cutilSafeCall(cudaFree(m_pBucketBlockResults));
}

unsigned int CUDAScan::prefixSum(unsigned int numElements, int* d_input, int* d_output)
{
	assert(numElements <= getMaxScanSize());

	prefixSumStub(d_input, m_pBucketResults, m_pBucketBlockResults, d_output, numElements, m_BucketSize, m_BucketBlockSize, m_NumBuckets, m_NumBucketBlocks);

	int sum = 0;
	cutilSafeCall(cudaMemcpy(&sum, &d_output[numElements-1], sizeof(int), cudaMemcpyDeviceToHost));
	
	return sum;
}

unsigned int CUDAScan::getMaxScanSize()
{
	return m_NumBuckets * m_BucketSize;
}
