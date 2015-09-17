#include <cutil_inline.h>
#include <cutil_math.h>

#define BLOCKSIZE 512
#define DEFAULT_SIZE 512
#define BUCKET_SIZE DEFAULT_SIZE
#define BUCKET_BLOCK_SIZE DEFAULT_SIZE
#define NUM_BUCKET_BLOCKS DEFAULT_SIZE
#define T_PER_BLOCK BUCKET_SIZE

#define DISPATCH_THREADS_X 128

__shared__ int bucketScan[BLOCKSIZE];

__device__ inline unsigned int WarpScan(volatile int* sdata, unsigned int lane, unsigned int i)
{
	uint4 access = i - make_uint4(1,2,4,8);
	if(lane > 0)  sdata[i] += sdata[access.x]; // IMPLICIT MEM BARRIER
	if(lane > 1)  sdata[i] += sdata[access.y]; // IMPLICIT MEM BARRIER
	if(lane > 3)  sdata[i] += sdata[access.z]; // IMPLICIT MEM BARRIER
	if(lane > 7)  sdata[i] += sdata[access.w]; // IMPLICIT MEM BARRIER
	if(lane > 15) sdata[i] += sdata[i-0x10];   // IMPLICIT MEM BARRIER
	
	return sdata[i];
}

__device__ inline  void ScanHelper(volatile int* sdata, unsigned int DTid, unsigned int GI, int x, int* d_output, unsigned int numElements)
{
	sdata[GI] = x;
	
	uint lane = GI & 31u;
	uint warp = GI >> 5u;
 
	x = WarpScan(sdata, lane, GI);
	__syncthreads();

	if (lane == 31) sdata[warp] = x;	
	__syncthreads();

	if (warp == 0) WarpScan(sdata, lane, lane);
	__syncthreads();

	if (warp > 0) x += sdata[warp-1];
	
	if(DTid < numElements) d_output[DTid] = x;
}

// Pass 1
__global__ void ScanBucketDevice(int* d_input, int* d_output, unsigned int numElements)
{
	const unsigned int idx = (blockIdx.x + blockIdx.y*DISPATCH_THREADS_X)*T_PER_BLOCK + threadIdx.x;
	
	int x = 0;
	if(idx < numElements) x = d_input[idx];
	ScanHelper(bucketScan, idx, threadIdx.x, x, d_output, numElements);
}

// Pass 2
__global__ void ScanBucketResults(int* d_input, int* d_output, unsigned int numElements)
{
	const unsigned int DTid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = (DTid+1) * BUCKET_SIZE - 1;
	
	int x = 0;
	if(idx < numElements) x = d_input[idx];
	ScanHelper(bucketScan, DTid, threadIdx.x, x, d_output, numElements);
}

// Pass 3
__global__ void ScanBucketBlockResults(int* d_input, int* d_output, unsigned int numElements)
{
	const unsigned int DTid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = threadIdx.x * BUCKET_BLOCK_SIZE - 1;
	
	int x = 0;
	if(idx < numElements) x = d_input[idx];
	ScanHelper(bucketScan, DTid, threadIdx.x, x, d_output,numElements);
}

// Pass 4
__global__ void ScanApplyBucketBlockResults(int* d_input, int* d_output, unsigned int numElements)
{
	const unsigned int DTid = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = DTid/BUCKET_BLOCK_SIZE;

	int x = 0;
	if(idx < numElements)  x = d_input[idx];
	if(DTid < numElements) d_output[DTid] += x;
}

// Pass 5
__global__ void ScanApplyBucketResults(int* d_input, int* d_output, unsigned int numElement)
{
	const unsigned int idx  = (blockIdx.x + blockIdx.y*DISPATCH_THREADS_X)*T_PER_BLOCK + threadIdx.x;
	const unsigned int idx2 = ((idx)/BUCKET_SIZE)-1;

	int x = 0;
	if(idx2 < numElement) x = d_input[idx2];
	if(idx < numElement)  d_output[idx] += x;
}

extern "C" void prefixSumStub(int* d_input, int* d_bucketResults, int* d_bucketBlockResults, int* d_output, unsigned int numElements, unsigned int bucketSize, unsigned int bucketBlockSize, unsigned int numBuckets, unsigned int numBucketBlocks)
{
	unsigned int groupsPass1 = (numElements + bucketSize - 1) / bucketSize;
	unsigned int dimX = DISPATCH_THREADS_X;
	unsigned int dimY = (groupsPass1 + dimX - 1) / dimX;
	
	dim3 blockSize(bucketSize, 1);
	dim3 gridSize(dimX, dimY);

	// Pass 1
	ScanBucketDevice<<<gridSize, blockSize>>>(d_input, d_output, numElements);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	// Pass 2
	unsigned int groupsPass2 = (groupsPass1 + BUCKET_BLOCK_SIZE - 1) / BUCKET_BLOCK_SIZE;
	blockSize = dim3(BUCKET_BLOCK_SIZE, 1, 1);
	gridSize  = dim3(groupsPass2, 1, 1);

	ScanBucketResults<<<gridSize, blockSize>>>(d_output, d_bucketResults, numElements);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	// Pass 3
	blockSize = dim3(NUM_BUCKET_BLOCKS, 1, 1);
	gridSize  = dim3(1, 1, 1);

	ScanBucketBlockResults<<<gridSize, blockSize>>>(d_bucketResults, d_bucketBlockResults, numElements);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	// Pass 4
	blockSize = dim3(BUCKET_BLOCK_SIZE, 1, 1);
	gridSize  = dim3(groupsPass2, 1, 1);

	ScanApplyBucketBlockResults<<<gridSize, blockSize>>>(d_bucketBlockResults, d_bucketResults, numElements);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif

	// Pass 5
	blockSize = dim3(bucketSize, 1, 1);
	gridSize  = dim3(dimX, dimY, 1);

	ScanApplyBucketResults<<<gridSize, blockSize>>>(d_bucketResults, d_output, numElements);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}
