
#include "SIFTImageManager.h"
#include "CUDATimer.h"
#include "cuda_kabsch.h"
#include "cuda_svd3.h"

#define SORT_NUM_BLOCK_THREADS_X (MAX_MATCHES_PER_IMAGE_PAIR_RAW / 2)

int CheckErrorCUDA(const char* location)
{
#if (defined(_DEBUG) || defined(DEBUG))
	cudaError_t e = cudaGetLastError();
	if (e)
	{
		if (location) fprintf(stderr, "%s:\t", location);
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		assert(0);
		return 1;
	}
	else
	{
		return 0;
	}
#else
	return 0;
#endif
}

__device__ bool cmpAndSawp (
	volatile float* dist0,
	volatile uint2* idx0,
	volatile float* dist1,
	volatile uint2* idx1
	) 
{
	if (dist0[0] > dist1[0]) {
		float tmpDist = dist0[0];
		dist0[0] = dist1[0];
		dist1[0] = tmpDist;

		const unsigned int tmpIdxX = idx0[0].x;
		idx0[0].x = idx1[0].x;
		idx1[0].x = tmpIdxX;

		const unsigned int tmpIdxY = idx0[0].y;
		idx0[0].y = idx1[0].y;
		idx1[0].y = tmpIdxY;
		return true;
	}
	else {
		return false;
	}
}

//we launch 1 thread for two array entries
void __global__ SortKeyPointMatchesCU_Kernel(
	const int* d_numMatchesPerImagePair,
	float* d_matchDistancesGlobal, 
	uint2* d_matchKeyPointIndicesGlobal)
{
	unsigned int imagePairIdx = blockIdx.x;
	unsigned int tidx = threadIdx.x;
	
	float* d_matchDistances = &d_matchDistancesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	uint2* d_matchKeyPointIndices = &d_matchKeyPointIndicesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	const unsigned int numMatches = d_numMatchesPerImagePair[imagePairIdx];

	if (numMatches == 0)	return;

	__shared__ float matchDistances[MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	__shared__ uint2 matchKeyPointIndices[MAX_MATCHES_PER_IMAGE_PAIR_RAW];

	if (2*tidx < numMatches) {
		matchDistances[2 * tidx + 0] = d_matchDistances[2 * tidx + 0];
		matchKeyPointIndices[2 * tidx + 0] = d_matchKeyPointIndices[2 * tidx + 0];

		if (2 * tidx + 1 < numMatches) {
			matchDistances[2 * tidx + 1] = d_matchDistances[2 * tidx + 1];
			matchKeyPointIndices[2 * tidx + 1] = d_matchKeyPointIndices[2 * tidx + 1];
		}
		else {
			matchDistances[2 * tidx + 1] = 999.0f;
			matchKeyPointIndices[2 * tidx + 1] = make_uint2((unsigned int)-1, (unsigned int)-1);
		}
	} else {
		matchDistances[2 * tidx + 0] = 999.0f;
		matchKeyPointIndices[2 * tidx + 0] = make_uint2((unsigned int)-1, (unsigned int)-1);

		matchDistances[2 * tidx + 1] = 999.0f;
		matchKeyPointIndices[2 * tidx + 1] = make_uint2((unsigned int)-1, (unsigned int)-1);
	}

#if !(SORT_NUM_BLOCK_THREADS_X == 32)
	__syncthreads();
#endif

	__shared__ bool swapped;
	swapped = true;
	unsigned int run = 0;
	while (swapped) {
		swapped = false;

		unsigned int idx0 = 2 * tidx + 0;
		unsigned int idx1 = 2 * tidx + 1;
		
		//odd phase
		if (run & 0x1) {
			idx0 += 1;
			idx1 += 1;
		}
		
		bool res = false;
		if (idx1 < MAX_MATCHES_PER_IMAGE_PAIR_RAW) {
			res = cmpAndSawp(&matchDistances[idx0], &matchKeyPointIndices[idx0], &matchDistances[idx1], &matchKeyPointIndices[idx1]);
		}

		if (res) swapped = true;

		run++;
#if !(SORT_NUM_BLOCK_THREADS_X == 32)
		__syncthreads();
#endif
	}

	//write results back
	if (2 * tidx < numMatches) {
		d_matchDistances[2 * tidx + 0] = matchDistances[2 * tidx + 0];
		d_matchKeyPointIndices[2 * tidx + 0] = matchKeyPointIndices[2 * tidx + 0];

		if (2 * tidx + 1 < numMatches) {
			d_matchDistances[2 * tidx + 1] = matchDistances[2 * tidx + 1];
			d_matchKeyPointIndices[2 * tidx + 1] = matchKeyPointIndices[2 * tidx + 1];
		}
	}

}


void SIFTImageManager::SortKeyPointMatchesCU(unsigned int numImagePairs) {

	if (numImagePairs == 0) return;

	dim3 grid(numImagePairs);
	dim3 block(SORT_NUM_BLOCK_THREADS_X);

	//CUDATimer timer;
	//timer.startEvent(__FUNCTION__);

	SortKeyPointMatchesCU_Kernel << <grid, block >> >(d_currNumMatchesPerImagePair, d_currMatchDistances, d_currMatchKeyPointIndices);

	//timer.endEvent();
	//timer.evaluate();

	CheckErrorCUDA(__FUNCTION__);

	//std::vector<uint2> indices(numImagePairs * MAX_MATCHES_PER_IMAGE_PAIR_RAW, make_uint2((unsigned int)-1,(unsigned int)-1));
	//std::vector<float> distances(numImagePairs * MAX_MATCHES_PER_IMAGE_PAIR_RAW, -1.0f);
	//std::vector<unsigned int> numMatchesPerImagePair(numImagePairs * MAX_MATCHES_PER_IMAGE_PAIR_RAW, 0);
	//cutilSafeCall(cudaMemcpy(distances.data(), d_matchDistances, sizeof(float) * numImagePairs * MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(indices.data(), d_matchKeyPointIndices, sizeof(uint2) * numImagePairs * MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(numMatchesPerImagePair.data(), d_numMatchesPerImagePair, sizeof(unsigned int) * MAX_MATCHES_PER_IMAGE_PAIR_RAW, cudaMemcpyDeviceToHost));

	//std::cout << "\n\nafter:\n";
	//for (size_t i = 0; i < numMatchesPerImagePair.size(); i++) {
	//	unsigned int numMatches = numMatchesPerImagePair[i];
	//	unsigned int baseIdx = i * MAX_MATCHES_PER_IMAGE_PAIR_RAW;
	//	for (size_t j = 0; j < numMatches; j++) {
	//		std::cout << distances[baseIdx + j] << " ";
	//	}	
	//}
	//std::cout << std::endl << std::endl;
}



#define FILTER_NUM_BLOCK_THREADS_X MAX_MATCHES_PER_IMAGE_PAIR_RAW


//we launch 1 thread for two array entries
void __global__ FilterKeyPointMatchesCU_Kernel(
	const SIFTKeyPoint* d_keyPointsGlobal,
	const int* d_numMatchesPerImagePair,
	const float* d_matchDistancesGlobal,
	const uint2* d_matchKeyPointIndicesGlobal,
	int* d_numFilteredMatchesPerImagePair,
	float* d_filteredMatchDistancesGlobal,
	uint2* d_filteredMatchKeyPointIndicesGlobal,
	float4x4* d_filteredTransforms)
{
	const unsigned int imagePairIdx = blockIdx.x;
	//const unsigned int imagePairIdx = 1;
	
	const unsigned int tidx = threadIdx.x;

	const float* d_matchDistances = &d_matchDistancesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	const uint2* d_matchKeyPointIndices = &d_matchKeyPointIndicesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	const unsigned int numMatches = d_numMatchesPerImagePair[imagePairIdx];

	if (numMatches == 0)	return;

	__shared__ float matchDistances[MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	__shared__ uint2 matchKeyPointIndices[MAX_MATCHES_PER_IMAGE_PAIR_RAW];

	if (tidx < numMatches) {
		matchDistances[tidx] = d_matchDistances[tidx];
		matchKeyPointIndices[tidx] = d_matchKeyPointIndices[tidx];
	} else {
		matchDistances[tidx] = 999.0f;
		matchKeyPointIndices[tidx] = make_uint2((unsigned int)-1, (unsigned int)-1);
	}

	
#if !(FILTER_NUM_BLOCK_THREADS_X == 32)
	__syncthreads();
#endif

	__shared__ unsigned int numFilteredMatches;

	if (tidx == 0) 	{
		float4x4 trans;
		unsigned int curr = filterKeyPointMatches(d_keyPointsGlobal, matchKeyPointIndices, matchDistances, numMatches, trans);
		//if (tidx == 0) {
			numFilteredMatches = curr;
			d_filteredTransforms[imagePairIdx] = trans;
		//}
	}

#if !(FILTER_NUM_BLOCK_THREADS_X == 32)
	__syncthreads();
#endif

	//write results back
	if (tidx == 0) {
		d_numFilteredMatchesPerImagePair[imagePairIdx] = numFilteredMatches;
	}

	if (tidx < numFilteredMatches) {
		d_filteredMatchDistancesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx] = matchDistances[tidx];
		d_filteredMatchKeyPointIndicesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx] = matchKeyPointIndices[tidx];
	}
	else if (tidx < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED) {
		d_filteredMatchDistancesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx] = 999.0f;
		d_filteredMatchKeyPointIndicesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx] = make_uint2((unsigned int)-1, (unsigned int)-1);
	}
}



void SIFTImageManager::FilterKeyPointMatchesCU(unsigned int numCurrImagePairs) {

	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	dim3 block(FILTER_NUM_BLOCK_THREADS_X);

	CUDATimer timer;
	timer.startEvent(__FUNCTION__);

	FilterKeyPointMatchesCU_Kernel << <grid, block >> >(
		d_keyPoints,
		d_currNumMatchesPerImagePair, 
		d_currMatchDistances, 
		d_currMatchKeyPointIndices,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchDistances,
		d_currFilteredMatchKeyPointIndices,
		d_currFilteredTransforms);

	timer.endEvent();
	timer.evaluate();

	CheckErrorCUDA(__FUNCTION__);

	//DEBUG
	{
		std::vector<int> numMatches(numCurrImagePairs);
		std::vector<float> matchDistancesGlob(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED*numCurrImagePairs);
		std::vector<float4x4> transforms(numCurrImagePairs);
		cutilSafeCall(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(int)*numCurrImagePairs, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(matchDistancesGlob.data(), d_currFilteredMatchDistances, sizeof(float)*numCurrImagePairs*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(transforms.data(), d_currFilteredTransforms, sizeof(float4x4)*numCurrImagePairs, cudaMemcpyDeviceToHost));
		for (unsigned int i = 0; i < numCurrImagePairs; i++) {
			unsigned int newNumMatches = numMatches[i];
			float checkSum = 0.0f;
			for (unsigned int k = 0; k < newNumMatches; k++) {
				checkSum += matchDistancesGlob[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED*i + k];
			}
			std::cout << "checkSum: " << checkSum << std::endl;
		}
		for (unsigned int i = 0; i < numCurrImagePairs; i++) {
			transforms[i].print();
		}
	}
}




void __global__ AddCurrToResidualsCU_Kernel(
	EntryJ* d_globMatches,
	uint2* d_globMatchesKeyPointIndices,
	int* d_globNumImagePairs,
	const int* d_currNumFilteredMatchesPerImagePair,
	const uint2* d_currFilteredMatchKeyPointIndices,
	const SIFTKeyPoint* d_keyPoints,
	const unsigned int maxKeyPointsPerImage
	)
{
	const unsigned int tidx = threadIdx.x;
	const unsigned int numMatches = d_currNumFilteredMatchesPerImagePair[blockIdx.x];
	__shared__ unsigned int basePtr;
	if (tidx == 0) {
		basePtr = atomicAdd(&d_globNumImagePairs[0], numMatches);
	}
	__syncthreads();

	//if (tidx == 0) {
	//	printf("[%d] baseAddr=%d\n", blockIdx.x, basePtr);
	//}

	if (tidx < numMatches) {
		const unsigned int srcAddr = blockIdx.x*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + tidx;

		uint2 currFilteredMachtKeyPointIndices = d_currFilteredMatchKeyPointIndices[srcAddr];

		//printf("[%d] = [%d %d]\n", blockIdx.x, currFilteredMachtKeyPointIndices.x, currFilteredMachtKeyPointIndices.y);

		const SIFTKeyPoint& k_i = d_keyPoints[currFilteredMachtKeyPointIndices.x];
		const SIFTKeyPoint& k_j = d_keyPoints[currFilteredMachtKeyPointIndices.y];
		
		//!!!TODO PARAMS
		const float _colorIntrinsicsInverse[16] = {
			0.000847232877f, 0.0f, -0.549854159f, 0.0f,
			0.0f, 0.000850733835f, -0.411329806f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f };
		float4x4 colorIntrinsicsInv(_colorIntrinsicsInverse);

		EntryJ e;
		const unsigned int imageIdx0 = blockIdx.x;
		const unsigned int imageIdx1 = gridDim.x;
		e.imgIdx_i = imageIdx0;
		e.imgIdx_j = imageIdx1;
		e.pos_i = colorIntrinsicsInv * (k_i.depth * make_float3(k_i.pos.x, k_i.pos.y, 1.0f));
		e.pos_j = colorIntrinsicsInv * (k_j.depth * make_float3(k_j.pos.x, k_j.pos.y, 1.0f));

		d_globMatches[basePtr + tidx] = e;
		d_globMatchesKeyPointIndices[basePtr + tidx] = currFilteredMachtKeyPointIndices;
	}
}

void SIFTImageManager::AddCurrToResidualsCU(unsigned int numCurrImagePairs) {
	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	const unsigned int threadsPerBlock = ((MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + 31) / 32) * 32;
	dim3 block(threadsPerBlock);

	//CUDATimer timer;
	//timer.startEvent(__FUNCTION__);

	AddCurrToResidualsCU_Kernel << <grid, block >> >(
		d_globMatches,
		d_globMatchesKeyPointIndices,
		d_globNumResiduals,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchKeyPointIndices,
		d_keyPoints,
		m_maxKeyPointsPerImage
		);

	cutilSafeCall(cudaMemcpy(&m_globNumResiduals, d_globNumResiduals, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	//timer.endEvent();
	//timer.evaluate();

	CheckErrorCUDA(__FUNCTION__);
}







#define INVALIDATEIMAGE_TO_IMAGE_KERNEL_THREADS_X 128

void __global__ InvalidateImageToImageCU_Kernel(EntryJ* d_globMatches, unsigned int globNumResiduals, uint2 imageToImageIdx)
{
	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < globNumResiduals) {
		if (d_globMatches[idx].imgIdx_i == imageToImageIdx.x &&
			d_globMatches[idx].imgIdx_j == imageToImageIdx.y) {
			d_globMatches[idx].setInvalid();
		}
		
	}

}

void SIFTImageManager::InvalidateImageToImageCU(const uint2& imageToImageIdx) {

	const unsigned int threadsPerBlock = INVALIDATEIMAGE_TO_IMAGE_KERNEL_THREADS_X;
	dim3 grid((m_globNumResiduals + threadsPerBlock - 1) / threadsPerBlock);
	dim3 block(threadsPerBlock);

	//CUDATimer timer;
	//timer.startEvent(__FUNCTION__);

	InvalidateImageToImageCU_Kernel << <grid, block >> >(d_globMatches, m_globNumResiduals, imageToImageIdx);

	//timer.endEvent();
	//timer.evaluate();

	CheckErrorCUDA(__FUNCTION__);
}


//#define CHECK_FOR_INVALID_FRAMES_THREADS_X 32
//
//void __global__ CheckForInvalidFramesCU_Kernel(const int* d_varToCorrNumEntriesPerRow, unsigned int* d_invalidFramesList, int* d_numInvalidFrames, unsigned int numVars)
//{
//	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
//
//	if (idx < numVars) {
//		if (d_varToCorrNumEntriesPerRow[idx] == 0) { // no connections!
//			int addr = atomicAdd(d_numInvalidFrames, 1);
//
//		}
//	}
//
//}
//
//void SIFTImageManager::CheckForInvalidFramesCU(const int* d_varToCorrNumEntriesPerRow, unsigned int* d_invalidFramesList, int* d_numInvalidFrames, unsigned int numVars)
//{
//	const unsigned int threadsPerBlock = CHECK_FOR_INVALID_FRAMES_THREADS_X;
//	dim3 grid((numVars + threadsPerBlock - 1) / threadsPerBlock);
//	dim3 block(threadsPerBlock);
//
//	//CUDATimer timer;
//	//timer.startEvent(__FUNCTION__);
//
//	cutilSafeCall(cudaMemset(d_numInvalidFrames, 0, sizeof(int));
//	CheckForInvalidFramesCU_Kernel << <grid, block >> >(d_varToCorrNumEntriesPerRow, d_invalidFramesList, d_numInvalidFrames, numVars);
//
//	//timer.endEvent();
//	//timer.evaluate();
//
//	CheckErrorCUDA(__FUNCTION__);
//}






void __global__ TestSVDDebugCU_Kernel(float3x3* d_m, float3x3* d_u, float3x3* d_s, float3x3* d_v) {

	float3x3 m = d_m[0];

	//printf("InKernel\n");
	//m.print();


	float3x3 u, s, v;
	s.setZero();
	v.setZero();
	//SVD::decompose3x3((float*)&m, (float*)&s, (float*)&v);
	for (unsigned int i = 0; i < 100; i++) {
		SVD::svd(m, u, s, v);
	}
	//svd(m, u, s, v);

	d_u[0] = u;
	d_s[0] = s;
	d_v[0] = v;
	//printf("\n");

	//u.print();
	//printf("\n");
	//s.print();
	//printf("\n");
	//v.print();

	//printf("\n");

	//float3x3 res = u * s * v.getTranspose();
	//res.print();
}





void SIFTImageManager::TestSVDDebugCU(const float3x3& m) {

	dim3 grid(1);
	dim3 block(1);

	float3x3* d_m, *d_u, *d_s, *d_v;
	cutilSafeCall(cudaMalloc(&d_m, sizeof(float3x3)));
	cutilSafeCall(cudaMalloc(&d_u, sizeof(float3x3)));
	cutilSafeCall(cudaMalloc(&d_s, sizeof(float3x3)));
	cutilSafeCall(cudaMalloc(&d_v, sizeof(float3x3)));

	//std::cout << "before:\n";
	//m.print();
	//std::cout << std::endl;

	cutilSafeCall(cudaMemcpy(d_m, &m, sizeof(float3x3), cudaMemcpyHostToDevice));

	CUDATimer timer;
	timer.startEvent(__FUNCTION__);

	TestSVDDebugCU_Kernel << <grid, block >> >(d_m, d_u, d_s, d_v);

	timer.endEvent();
	timer.evaluate();

	float3x3 u, s, v;
	cutilSafeCall(cudaMemcpy(&u, d_u, sizeof(float3x3), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&s, d_s, sizeof(float3x3), cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(&v, d_v, sizeof(float3x3), cudaMemcpyDeviceToHost));

	//printf("\n");
	//u.print(); 
	//printf("\n");
	//s.print();
	//printf("\n");
	//v.print();
	//printf("\n");
	float3x3 res = u * s * v.getTranspose();
	res.print();
	printf("\n\n");

	CheckErrorCUDA(__FUNCTION__);

}
