
#include "SIFTImageManager.h"
#include "cudaUtil.h"
#include "CUDATimer.h"
#include "cuda_kabsch.h"
#include "cuda_svd3.h"
#include "cuda_surfaceArea.h"

#define SORT_NUM_BLOCK_THREADS_X (MAX_MATCHES_PER_IMAGE_PAIR_RAW / 2)

int CheckErrorCUDA(const char* location)
{
#if (defined(_DEBUG) || defined(DEBUG))
	cudaDeviceSynchronize();
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

__device__ bool cmpAndSawp(
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

	if (2 * tidx < numMatches) {
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
	}
	else {
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

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	SortKeyPointMatchesCU_Kernel << <grid, block >> >(d_currNumMatchesPerImagePair, d_currMatchDistances, d_currMatchKeyPointIndices);

	if (m_timer) m_timer->endEvent();

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
	float4x4* d_filteredTransforms,
	float4x4* d_filteredTransformsInv,
	float4x4 siftIntrinsicsInv,
	unsigned int minNumMatches)
{
	const unsigned int imagePairIdx = blockIdx.x;
	//const unsigned int imagePairIdx = 1;

	const unsigned int tidx = threadIdx.x;

	const float* d_matchDistances = &d_matchDistancesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	const uint2* d_matchKeyPointIndices = &d_matchKeyPointIndicesGlobal[imagePairIdx*MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	unsigned int numMatches = d_numMatchesPerImagePair[imagePairIdx];

	if (numMatches == 0) {
		if (tidx == 0) {
			d_numFilteredMatchesPerImagePair[imagePairIdx] = 0;
		}
		return;
	}
	if (numMatches > MAX_MATCHES_PER_IMAGE_PAIR_RAW) numMatches = MAX_MATCHES_PER_IMAGE_PAIR_RAW;

	__shared__ float matchDistances[MAX_MATCHES_PER_IMAGE_PAIR_RAW];
	__shared__ uint2 matchKeyPointIndices[MAX_MATCHES_PER_IMAGE_PAIR_RAW];

	if (tidx < numMatches) {
		matchDistances[tidx] = d_matchDistances[tidx];
		matchKeyPointIndices[tidx] = d_matchKeyPointIndices[tidx];
	}
	else {
		matchDistances[tidx] = 999.0f;
		matchKeyPointIndices[tidx] = make_uint2((unsigned int)-1, (unsigned int)-1);
	}


#if !(FILTER_NUM_BLOCK_THREADS_X == 32)
	__syncthreads();
#endif

	__shared__ unsigned int numFilteredMatches;

	if (tidx == 0) 	{
		float4x4 trans;
		unsigned int curr = filterKeyPointMatches(d_keyPointsGlobal, matchKeyPointIndices, matchDistances, numMatches, trans, siftIntrinsicsInv, minNumMatches);
		//if (tidx == 0) {
		numFilteredMatches = curr;
		d_filteredTransforms[imagePairIdx] = trans;
		d_filteredTransformsInv[imagePairIdx] = trans.getInverse();
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

void SIFTImageManager::FilterKeyPointMatchesCU(unsigned int numCurrImagePairs, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches) {

	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	dim3 block(FILTER_NUM_BLOCK_THREADS_X);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	FilterKeyPointMatchesCU_Kernel << <grid, block >> >(
		d_keyPoints,
		d_currNumMatchesPerImagePair,
		d_currMatchDistances,
		d_currMatchKeyPointIndices,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchDistances,
		d_currFilteredMatchKeyPointIndices,
		d_currFilteredTransforms,
		d_currFilteredTransformsInv,
		siftIntrinsicsInv,
		minNumMatches);

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);

	//DEBUG
	//{
	//	std::vector<int> numMatches(numCurrImagePairs);
	//	std::vector<float> matchDistancesGlob(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED*numCurrImagePairs);
	//	std::vector<float4x4> transforms(numCurrImagePairs);
	//	cutilSafeCall(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(int)*numCurrImagePairs, cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(matchDistancesGlob.data(), d_currFilteredMatchDistances, sizeof(float)*numCurrImagePairs*MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(transforms.data(), d_currFilteredTransforms, sizeof(float4x4)*numCurrImagePairs, cudaMemcpyDeviceToHost));
	//	for (unsigned int i = 0; i < numCurrImagePairs; i++) {
	//		unsigned int newNumMatches = numMatches[i];
	//		float checkSum = 0.0f;
	//		for (unsigned int k = 0; k < newNumMatches; k++) {
	//			checkSum += matchDistancesGlob[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED*i + k];
	//		}
	//		std::cout << "checkSum: " << checkSum << std::endl;
	//	}
	//	for (unsigned int i = 0; i < numCurrImagePairs; i++) {
	//		transforms[i].print();
	//	}
	//}
}


//we launch 1 thread for two array entries
void __global__ FilterMatchesBySurfaceAreaCU_Kernel(
	const SIFTKeyPoint* d_keyPointsGlobal,
	int* d_numFilteredMatchesPerImagePair,
	const uint2* d_filteredMatchKeyPointIndicesGlobal,
	const float4x4 colorIntrinsicsInv,
	float areaThresh)
{
	const unsigned int imagePairIdx = blockIdx.x;
	//const unsigned int tidx = threadIdx.x;

	const unsigned int numMatches = d_numFilteredMatchesPerImagePair[imagePairIdx];
	if (numMatches == 0)	return;
	const uint2* d_keyPointMatchIndices = d_filteredMatchKeyPointIndicesGlobal + imagePairIdx * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED;

	float area0 = 0.0f;
	float area1 = 0.0f;


	// compute area image 0
	unsigned int which = 0;
	computeKeyPointMatchesCovariance(d_keyPointsGlobal, d_keyPointMatchIndices, numMatches, colorIntrinsicsInv, which);

	float3 evs, ev0, ev1, ev2;
	bool res;

	res = MYEIGEN::eigenSystem(V, evs, ev0, ev1, ev2);
	__shared__ float2 pointsProj[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
	if (res) { // project
		projectKeysToPlane(pointsProj, d_keyPointsGlobal, d_keyPointMatchIndices, numMatches, colorIntrinsicsInv, which, ev0, ev1, ev2, mean);
		area0 = computeAreaOrientedBoundingBox2(pointsProj, numMatches);
	}

	////if (threadIdx.x == 0)
	//{
	//	float3x3 V_ = V;
	//	float3 mean_ = mean;
	//	__syncthreads();
	//	printf("matches: %d\n", numMatches);
	//	V_.print();
	//	printf("mean: [%f %f %f]\n", mean_.x, mean_.y, mean_.z);
	//	printf("evs [%f | %f | %f]\n", evs.x, evs.y, evs.z);
	//	printf("ev0 [%f | %f | %f]\n", ev0.x, ev0.y, ev0.z);
	//	printf("ev1 [%f | %f | %f]\n", ev1.x, ev1.y, ev1.z);
	//	printf("ev2 [%f | %f | %f]\n", ev2.x, ev2.y, ev2.z);
	//	printf("res %d\n", (int)res);
	//	printf("area0: %f\n", area0);
	//}
	//if (threadIdx.x == 0) {
	//	colorIntrinsicsInv.print();
	//	for (unsigned int i = 0; i < numMatches; i++) {
	//		printf("pointsProj[%d] = [%f | %f] \t\t idx: [%d | %d]\n", i, pointsProj[i].x, pointsProj[i].y, d_keyPointMatchIndices[i].x, d_keyPointMatchIndices[i].y);
	//	}
	//}

	// compute area image 1
	which = 1;
	computeKeyPointMatchesCovariance(d_keyPointsGlobal, d_keyPointMatchIndices, numMatches, colorIntrinsicsInv, which);

	res = MYEIGEN::eigenSystem(V, evs, ev0, ev1, ev2);
	if (res) {// project
		projectKeysToPlane(pointsProj, d_keyPointsGlobal, d_keyPointMatchIndices, numMatches, colorIntrinsicsInv, which, ev0, ev1, ev2, mean);
		area1 = computeAreaOrientedBoundingBox2(pointsProj, numMatches);
	}

	if (threadIdx.x == 0) {
		//printf("[%d] areas %f %f\n", imagePairIdx, area0, area1);
		if (area0 < areaThresh && area1 < areaThresh) {
			d_numFilteredMatchesPerImagePair[imagePairIdx] = 0;
		}
	}
}

void SIFTImageManager::FilterMatchesBySurfaceAreaCU(unsigned int numCurrImagePairs, const float4x4& colorIntrinsicsInv, float areaThresh) {
	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	const unsigned int threadsPerBlock = ((MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + 31) / 32) * 32;
	dim3 block(threadsPerBlock);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	FilterMatchesBySurfaceAreaCU_Kernel << <grid, block >> >(
		d_keyPoints,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchKeyPointIndices,
		colorIntrinsicsInv,
		areaThresh);

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);
}


#define FILTER_DENSE_VERIFY_THREAD_SPLIT 32

__device__ float3 computeProjError(unsigned int idx, unsigned int imageWidth, unsigned int imageHeight,
	float distThresh, float normalThresh, float colorThresh, const float4x4& transform, const float4x4& intrinsics,
	const float* d_inputDepth, const float4* d_inputCamPos, const float4* d_inputNormal, const uchar4* d_inputColor,
	const float* d_modelDepth, const float4* d_modelCamPos, const float4* d_modelNormal, const uchar4* d_modelColor,
	float sensorDepthMin, float sensorDepthMax)
{
	float3 out = make_float3(0.0f);

	float4 pInput = d_inputCamPos[idx]; // point
	float4 nInput = d_inputNormal[idx]; nInput.w = 0.0f; // vector
	//uchar4 cInput = d_inputColor[idx];

	if (pInput.x != MINF && nInput.x != MINF) {
		const float4 pTransInput = transform * pInput;
		const float4 nTransInput = transform * nInput;

		float3 tmp = intrinsics * make_float3(pTransInput.x, pTransInput.y, pTransInput.z);
		int2 screenPos = make_int2((int)roundf(tmp.x / tmp.z), (int)roundf(tmp.y / tmp.z)); // subsampled space
		//float3 tmp = intrinsics * make_float3(pTransInput.x, pTransInput.y, pTransInput.z);
		//int2 screenPos = make_int2((int)roundf(tmp.x / tmp.z) / 4, (int)roundf(tmp.y / tmp.z) / 4); // subsampled space

		if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < (int)imageWidth && screenPos.y < (int)imageHeight) {
			float4 pTarget = d_modelCamPos[screenPos.y * imageWidth + screenPos.x]; //getBestCorrespondence1x1
			float4 nTarget = d_modelNormal[screenPos.y * imageWidth + screenPos.x];
			//uchar4 cTarget = d_modelColor[screenPos.y * imageWidth + screenPos.x];

			if (pTarget.x != MINF && nTarget.x != MINF) {
				float d = length(pTransInput - pTarget);
				float dNormal = dot(make_float3(nTransInput.x, nTransInput.y, nTransInput.z), make_float3(nTarget.x, nTarget.y, nTarget.z)); // should be able to do dot(nTransInput, nTarget)
				//float c = length(make_float3((cInput.x - cTarget.x) / 255.0f, (cInput.y - cTarget.y) / 255.0f, (cInput.z - cTarget.z) / 255.0f));

				float projInputDepth = (intrinsics * make_float3(pTransInput.x, pTransInput.y, pTransInput.z)).z;
				float tgtDepth = d_modelDepth[screenPos.y * imageWidth + screenPos.x];

				bool b = ((tgtDepth != MINF && projInputDepth < tgtDepth) && d > distThresh); // bad matches that are known
				if ((dNormal >= normalThresh && d <= distThresh /*&& c <= colorThresh*/) || b) { // if normal/pos/color correspond or known bad match

					const float cameraToKinectProjZ = (pTransInput.z - 0.1f) / (3.0f - 0.1f); //TODO PARAMS HERE
					const float weight = max(0.0f, 0.5f*((1.0f - d / distThresh) + (1.0f - cameraToKinectProjZ))); // for weighted ICP;

					out.x = length(pTransInput - pTarget);	//residual
					out.y = weight;							//corr weight
					out.z = 1.0f;
				}
			} // projected to valid depth
		} // inside image
	}

	return out;
}


//we launch 1 thread for two array entries
void __global__ FilterMatchesByDenseVerifyCU_Kernel(unsigned int imageWidth, unsigned int imageHeight, const float4x4 intrinsics,
	int* d_currNumFilteredMatchesPerImagePair, const float4x4* d_currFilteredTransforms, const float4x4* d_currFilteredTransformsInv, const CUDACachedFrame* d_cachedFrames,
	float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh, float sensorDepthMin, float sensorDepthMax)
{
	const unsigned int curImageIdx = gridDim.x;
	const unsigned int imagePairIdx = blockIdx.x; // prev image idx
	const unsigned int numMatches = d_currNumFilteredMatchesPerImagePair[imagePairIdx];
	if (numMatches == 0)	return;

	const float*  d_inputDepth = d_cachedFrames[imagePairIdx].d_depthDownsampled;
	const float4* d_inputCamPos = d_cachedFrames[imagePairIdx].d_cameraposDownsampled;
	const float4* d_inputNormal = d_cachedFrames[imagePairIdx].d_normalsDownsampled;
	const uchar4* d_inputColor = d_cachedFrames[imagePairIdx].d_colorDownsampled;

	const float*  d_modelDepth = d_cachedFrames[curImageIdx].d_depthDownsampled;
	const float4* d_modelCamPos = d_cachedFrames[curImageIdx].d_cameraposDownsampled;
	const float4* d_modelNormal = d_cachedFrames[curImageIdx].d_normalsDownsampled;
	const uchar4* d_modelColor = d_cachedFrames[curImageIdx].d_colorDownsampled;

	const float4x4 transform = d_currFilteredTransforms[imagePairIdx];


	float local_sumResidual = 0.0f;
	float local_sumWeight = 0.0f;
	float local_numCorr = 0.0f;

	for (unsigned int i = 0; i < FILTER_DENSE_VERIFY_THREAD_SPLIT; i++) {
		const unsigned int idxX = threadIdx.x;
		const unsigned int idxY = threadIdx.y*FILTER_DENSE_VERIFY_THREAD_SPLIT + i;
		if (idxY < imageHeight) {
			const unsigned int idx = idxY * imageWidth + idxX;

			float3 inputToModel = computeProjError(idx, imageWidth, imageHeight, distThresh, normalThresh, colorThresh, transform, intrinsics,
				d_inputDepth, d_inputCamPos, d_inputNormal, d_inputColor,
				d_modelDepth, d_modelCamPos, d_modelNormal, d_modelColor, sensorDepthMin, sensorDepthMax);
			float3 modelToInput = computeProjError(idx, imageWidth, imageHeight, distThresh, normalThresh, colorThresh, transform.getInverse(), intrinsics,
				d_modelDepth, d_modelCamPos, d_modelNormal, d_modelColor,
				d_inputDepth, d_inputCamPos, d_inputNormal, d_inputColor, sensorDepthMin, sensorDepthMax);

			local_sumResidual += inputToModel.x + modelToInput.x;	//residual
			local_sumWeight += inputToModel.y + modelToInput.y;		//corr weight
			local_numCorr += inputToModel.z + modelToInput.z;		//corr number
		}
	}

	__shared__ float sumResidual;
	__shared__ float sumWeight;
	__shared__ float numCorr;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		sumResidual = 0.0f;
		sumWeight = 0.0f;
		numCorr = 0;
	}
	__syncthreads();

	//atomicAdd(&sumResidual, local_sumResidual);
	//atomicAdd(&sumWeight, local_sumWeight);
	//atomicAdd(&numCorr, local_numCorr);

	local_sumResidual = warpReduceSum(local_sumResidual);
	local_sumWeight = warpReduceSum(local_sumWeight);
	local_numCorr = warpReduceSum(local_numCorr);

	if (threadIdx.x % warpSize == 0) {
		atomicAdd(&sumResidual, local_sumResidual);
		atomicAdd(&sumWeight, local_sumWeight);
		atomicAdd(&numCorr, local_numCorr);
	}

	__syncthreads();

	//write results back
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float err = sumResidual / sumWeight;
		float corr = 0.5f * numCorr / (float)(imageWidth * imageHeight);
		//printf("[%d-%d]: %f %f\n", imagePairIdx, curImageIdx, err, corr);

		if (corr < corrThresh || err > errThresh || isnan(err)) { // invalid!
			d_currNumFilteredMatchesPerImagePair[imagePairIdx] = 0;
		}
	}
}

//TODO camera stuff here
void SIFTImageManager::FilterMatchesByDenseVerifyCU(unsigned int numCurrImagePairs, unsigned int imageWidth, unsigned int imageHeight,
	const float4x4 intrinsics, const CUDACachedFrame* d_cachedFrames,
	float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh, float sensorDepthMin, float sensorDepthMax)
{
	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	dim3 block(imageWidth, (imageHeight + FILTER_DENSE_VERIFY_THREAD_SPLIT - 1) / FILTER_DENSE_VERIFY_THREAD_SPLIT);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	FilterMatchesByDenseVerifyCU_Kernel << <grid, block >> >(
		imageWidth, imageHeight, intrinsics,
		d_currNumFilteredMatchesPerImagePair, d_currFilteredTransforms, d_currFilteredTransformsInv, d_cachedFrames,
		distThresh, normalThresh, colorThresh, errThresh, corrThresh,
		sensorDepthMin, sensorDepthMax);

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);
}


void __global__ AddCurrToResidualsCU_Kernel(
	EntryJ* d_globMatches,
	uint2* d_globMatchesKeyPointIndices,
	int* d_globNumImagePairs,
	const int* d_currNumFilteredMatchesPerImagePair,
	const uint2* d_currFilteredMatchKeyPointIndices,
	const SIFTKeyPoint* d_keyPoints,
	const unsigned int maxKeyPointsPerImage,
	const float4x4 colorIntrinsicsInv
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

void SIFTImageManager::AddCurrToResidualsCU(unsigned int numCurrImagePairs, const float4x4& colorIntrinsicsInv) {
	if (numCurrImagePairs == 0) return;

	dim3 grid(numCurrImagePairs);
	const unsigned int threadsPerBlock = ((MAX_MATCHES_PER_IMAGE_PAIR_FILTERED + 31) / 32) * 32;
	dim3 block(threadsPerBlock);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	AddCurrToResidualsCU_Kernel << <grid, block >> >(
		d_globMatches,
		d_globMatchesKeyPointIndices,
		d_globNumResiduals,
		d_currNumFilteredMatchesPerImagePair,
		d_currFilteredMatchKeyPointIndices,
		d_keyPoints,
		m_maxKeyPointsPerImage,
		colorIntrinsicsInv
		);

	cutilSafeCall(cudaMemcpy(&m_globNumResiduals, d_globNumResiduals, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);
}


//#define FILTER_FRAMES_NUM_BLOCK_THREADS_X 8
//void __global__ FilterFramesCU_Kernel(
//	int* d_numFilteredMatchesPerImagePair,
//	int* d_validImages, unsigned int curImageIndex)
//{
//	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (idx < curImageIndex) {
//		if (d_numFilteredMatchesPerImagePair[idx] > 0)
//			d_validImages[curImageIndex] = 1;
//	}
//}
//
//void SIFTImageManager::FilterFramesCU(unsigned int numCurrImagePairs) {
//
//	if (numCurrImagePairs == 0) return;
//	dim3 grid((numCurrImagePairs + FILTER_FRAMES_NUM_BLOCK_THREADS_X - 1) / FILTER_FRAMES_NUM_BLOCK_THREADS_X);
//	dim3 block(FILTER_FRAMES_NUM_BLOCK_THREADS_X);
//
//	FilterFramesCU_Kernel <<< grid, block >>>(d_currNumFilteredMatchesPerImagePair, d_validImages, numCurrImagePairs);
//
//	CheckErrorCUDA(__FUNCTION__);
//
//	////debug
//	//unsigned int valid;
//	//cutilSafeCall(cudaMemcpy(&valid, d_validImages + numCurrImagePairs, sizeof(unsigned int), cudaMemcpyDeviceToHost));
//	//if (!valid) printf("frame %d not connected to previous!\n", numCurrImagePairs);
//}



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

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	InvalidateImageToImageCU_Kernel << <grid, block >> >(d_globMatches, m_globNumResiduals, imageToImageIdx);

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);
}


#define CHECK_FOR_INVALID_FRAMES_X 128
#define CHECK_FOR_INVALID_FRAMES_THREADS_X 16

void __global__ CheckForInvalidFramesCU_Kernel(const int* d_varToCorrNumEntriesPerRow, int* d_validImages, unsigned int numVars,
	EntryJ* d_globMatches, unsigned int numGlobResiduals)
{
	//const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	//if (idx < numVars) {
	//	if (d_varToCorrNumEntriesPerRow[idx] == 0) { // no connections!
	//		//printf("[CheckForInvalidFramesCU] invalidating frame %d\n", idx); //TODO remove debug print
	//		d_validImages[idx] = 0;
	//	}
	//}

	//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
	//	printf("(vars) blockDim = %d %d, (res)gridDim = %d %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
	//}
	const unsigned int resIdx = blockDim.x*blockIdx.x + blockIdx.y;
	const unsigned int varIdx = gridDim.x*threadIdx.x + threadIdx.y;

	//printf("%d,%d | %d,%d -> (v %d, r %d) (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, varIdx, resIdx, numVars, numGlobResiduals);

	if (varIdx < numVars && resIdx < numGlobResiduals) {
		if (d_varToCorrNumEntriesPerRow[varIdx] == 0) { // no connections!
			if (d_globMatches[resIdx].isValid() && (d_globMatches[resIdx].imgIdx_i == varIdx || d_globMatches[resIdx].imgIdx_j == varIdx)) { // invalidate residuals
				d_globMatches[resIdx].setInvalid();
			}
			//printf("[CheckForInvalidFramesCU] invalidating frame %d\n", idx); //TODO remove debug print
			if (d_validImages[varIdx] != 0) d_validImages[varIdx] = 0;
		}
	}

}
//TODO CHECK grid/block dim (too many threads?)
void SIFTImageManager::CheckForInvalidFramesCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars)
{
	dim3 block((m_globNumResiduals + CHECK_FOR_INVALID_FRAMES_X - 1) / CHECK_FOR_INVALID_FRAMES_X, CHECK_FOR_INVALID_FRAMES_X);
	dim3 threadsPerBlock((numVars + CHECK_FOR_INVALID_FRAMES_THREADS_X - 1) / CHECK_FOR_INVALID_FRAMES_THREADS_X, CHECK_FOR_INVALID_FRAMES_THREADS_X);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	//std::cout << __FUNCTION__ << ", #vars = " << numVars << ", #res = " << m_globNumResiduals << std::endl;

	cutilSafeCall(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int) * numVars, cudaMemcpyHostToDevice));

	CheckForInvalidFramesCU_Kernel << <block, threadsPerBlock >> >(d_varToCorrNumEntriesPerRow, d_validImages, numVars, d_globMatches, m_globNumResiduals);

	cutilSafeCall(cudaMemcpy(m_validImages.data(), d_validImages, sizeof(int) * numVars, cudaMemcpyDeviceToHost));

	//std::cout << "done (press key to continue)" << std::endl;
	//getchar();

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);
}


#define MARK_FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X 128
#define FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X 512

void __global__ FuseToGlobalKeyCU_Kernel(unsigned int maxNumKeysAll, int* d_fuseGlobalKeyMarker,
	const SIFTKeyPoint* d_allLocalKeyPoints, const SIFTKeyPointDesc* d_allLocalKeyPointDescs,
	SIFTKeyPoint* d_curGlobalKeyPoints, SIFTKeyPointDesc* d_curGlobalKeyPointsDescs,
	const float4x4* transforms, float4x4 colorIntrinsics, float4x4 colorIntrinsicsInv, int* d_fuseGlobalKeyCount,
	unsigned int maxNumKeysPerImage)
{
	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < maxNumKeysAll) {
		if (d_fuseGlobalKeyMarker[idx] > 0) {
			int addr = atomicAdd(d_fuseGlobalKeyCount, 1);
			if (addr < maxNumKeysPerImage) {
				const unsigned int imgIdx = d_fuseGlobalKeyMarker[idx] - 1;
				const SIFTKeyPoint key = d_allLocalKeyPoints[idx];
				float3 pos = colorIntrinsicsInv * (key.depth * make_float3(key.pos.x, key.pos.y, 1.0f));
				float3 projPos = colorIntrinsics * (transforms[imgIdx] * pos);
				float2 loc = make_float2(projPos.x / projPos.z, projPos.y / projPos.z);

				SIFTKeyPoint newKey;
				newKey.pos = loc;
				newKey.scale = key.scale;
				newKey.depth = projPos.z;
				d_curGlobalKeyPoints[addr] = newKey;
				d_curGlobalKeyPointsDescs[addr] = d_allLocalKeyPointDescs[idx];
			}
			// unmark for next time
			d_fuseGlobalKeyMarker[idx] = 0;
		} // marked
	}
}
void __global__ MarkKeysToFuseToGlobalKeyCU_Kernel(unsigned int globNumResiduals, const EntryJ* d_correspondences, uint2* d_correspondenceKeyIndices,
	int* d_fuseGlobalKeyMarker)
{
	const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

	if (idx < globNumResiduals) {
		const EntryJ& corr = d_correspondences[idx];
		if (corr.isValid()) {
			const uint2 keyIndices = d_correspondenceKeyIndices[idx];
			d_fuseGlobalKeyMarker[keyIndices.x] = corr.imgIdx_i + 1; // just pick the first one (offset by 1 since 0 invalid)
		} // valid corr
	} // residual/correspondence
}

unsigned int SIFTImageManager::FuseToGlobalKeyCU(SIFTImageGPU& globalImage, const float4x4* transforms,
	const float4x4& colorIntrinsics, const float4x4& colorIntrinsicsInv)
{
	dim3 gridMark((m_globNumResiduals + MARK_FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X - 1) / MARK_FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X);
	dim3 blockMark(MARK_FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	MarkKeysToFuseToGlobalKeyCU_Kernel << <gridMark, blockMark >> >(m_globNumResiduals, d_globMatches,
		d_globMatchesKeyPointIndices, d_fuseGlobalKeyMarker);

	CheckErrorCUDA(__FUNCTION__);

	const unsigned int maxNumKeysAll = m_submapSize * m_maxKeyPointsPerImage;
	dim3 gridFuse((maxNumKeysAll + FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X - 1) / FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X);
	dim3 blockFuse(FUSE_TO_GLOBAL_KEY_KERNEL_THREADS_X);

	cutilSafeCall(cudaMemset(d_fuseGlobalKeyCount, 0, sizeof(int)));
	FuseToGlobalKeyCU_Kernel << <gridFuse, blockFuse >> >(maxNumKeysAll, d_fuseGlobalKeyMarker,
		d_keyPoints, d_keyPointDescs, globalImage.d_keyPoints, globalImage.d_keyPointDescs,
		transforms, colorIntrinsics, colorIntrinsicsInv, d_fuseGlobalKeyCount, m_maxKeyPointsPerImage);

	CheckErrorCUDA(__FUNCTION__);

	unsigned int numKeys;
	cutilSafeCall(cudaMemcpy(&numKeys, d_fuseGlobalKeyCount, sizeof(int), cudaMemcpyDeviceToHost));
	if (numKeys > m_maxKeyPointsPerImage) numKeys = m_maxKeyPointsPerImage;

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);

	return numKeys;
}



__global__ void getSiftTransformCU_Kernel(unsigned int curFrameIndex,
	const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform,
	float4x4* d_siftTrajectory, unsigned int curFrameIndexAll,
	const int* d_numFilteredMatchesPerImagePair,
	const float4x4* d_filteredTransformsInv, float4x4* d_currIntegrateTrans)
{

	for (int i = (int)curFrameIndex - 1; i >= 0; i--) {
		if (d_numFilteredMatchesPerImagePair[i] > 0) {
			float4x4 transform;
			const unsigned int idxPrevSiftKnown = curFrameIndexAll - (curFrameIndex - i);
			d_siftTrajectory[curFrameIndexAll] = d_siftTrajectory[idxPrevSiftKnown] * d_filteredTransformsInv[i];

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

			break;
		}
	}
}

void SIFTImageManager::computeSiftTransformCU(const float4x4* d_completeTrajectory, unsigned int lastValidCompleteTransform,
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


//we launch 1 thread for two array entries
void __global__ VerifyTrajectoryCU_Kernel(unsigned int numImages, float4x4* d_trajectory,
	unsigned int imageWidth, unsigned int imageHeight,
	const float4x4 intrinsics, const CUDACachedFrame* d_cachedFrames,
	float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh,
	int* d_validOpt, float sensorDepthMin, float sensorDepthMax)
{
	const unsigned int img0 = blockIdx.x / numImages;
	const unsigned int img1 = blockIdx.x % numImages;

	if (img0 >= img1) return;
	//if (d_validOpt[0] == 0) return; //TODO 

	const float*  d_inputDepth = d_cachedFrames[img0].d_depthDownsampled;
	const float4* d_inputCamPos = d_cachedFrames[img0].d_cameraposDownsampled;
	const float4* d_inputNormal = d_cachedFrames[img0].d_normalsDownsampled;
	const uchar4* d_inputColor = d_cachedFrames[img0].d_colorDownsampled;

	const float*  d_modelDepth = d_cachedFrames[img1].d_depthDownsampled;
	const float4* d_modelCamPos = d_cachedFrames[img1].d_cameraposDownsampled;
	const float4* d_modelNormal = d_cachedFrames[img1].d_normalsDownsampled;
	const uchar4* d_modelColor = d_cachedFrames[img1].d_colorDownsampled;

	const float4x4 transform = d_trajectory[img1].getInverse() * d_trajectory[img0];

	float local_sumResidual = 0.0f;
	float local_sumWeight = 0.0f;
	float local_numCorr = 0.0f;

	for (unsigned int i = 0; i < FILTER_DENSE_VERIFY_THREAD_SPLIT; i++) {
		const unsigned int idxX = threadIdx.x;
		const unsigned int idxY = threadIdx.y*FILTER_DENSE_VERIFY_THREAD_SPLIT + i;
		if (idxY < imageHeight) {
			const unsigned int idx = idxY * imageWidth + idxX;

			float3 inputToModel = computeProjError(idx, imageWidth, imageHeight, distThresh, normalThresh, colorThresh, transform, intrinsics,
				d_inputDepth, d_inputCamPos, d_inputNormal, d_inputColor,
				d_modelDepth, d_modelCamPos, d_modelNormal, d_modelColor, sensorDepthMin, sensorDepthMax);
			float3 modelToInput = computeProjError(idx, imageWidth, imageHeight, distThresh, normalThresh, colorThresh, transform.getInverse(), intrinsics,
				d_modelDepth, d_modelCamPos, d_modelNormal, d_modelColor,
				d_inputDepth, d_inputCamPos, d_inputNormal, d_inputColor, sensorDepthMin, sensorDepthMax);

			local_sumResidual += inputToModel.x + modelToInput.x;	//residual
			local_sumWeight += inputToModel.y + modelToInput.y;		//corr weight
			local_numCorr += inputToModel.z + modelToInput.z;		//corr number
		}
	}

	__shared__ float sumResidual;
	__shared__ float sumWeight;
	__shared__ float numCorr;

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		sumResidual = 0.0f;
		sumWeight = 0.0f;
		numCorr = 0;
	}
	__syncthreads();

	//atomicAdd(&sumResidual, local_sumResidual);
	//atomicAdd(&sumWeight, local_sumWeight);
	//atomicAdd(&numCorr, local_numCorr);

	local_sumResidual = warpReduceSum(local_sumResidual);
	local_sumWeight = warpReduceSum(local_sumWeight);
	local_numCorr = warpReduceSum(local_numCorr);

	if (threadIdx.x % warpSize == 0) {
		atomicAdd(&sumResidual, local_sumResidual);
		atomicAdd(&sumWeight, local_sumWeight);
		atomicAdd(&numCorr, local_numCorr);
	}

	__syncthreads();

	//write results back
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		float err = sumResidual / sumWeight;
		float corr = 0.5f * numCorr / (float)(imageWidth * imageHeight);
		//printf("VERIFY LOCAL SUBMAP[%d-%d]: %f %f\n", img0, img1, err, corr);

		if (corr < corrThresh || err > errThresh || isnan(err)) { // invalid!
			d_validOpt[0] = 0;
		}
	}
}

int SIFTImageManager::VerifyTrajectoryCU(unsigned int numImages, float4x4* d_trajectory,
	unsigned int imageWidth, unsigned int imageHeight,
	const float4x4 intrinsics, const CUDACachedFrame* d_cachedFrames,
	float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh,
	float sensorDepthMin, float sensorDepthMax)
{
	if (numImages < 2) return 0;
	const unsigned int numPairs = (numImages * (numImages - 1)) / 2;

	dim3 grid(numPairs);
	dim3 block(imageWidth, (imageHeight + FILTER_DENSE_VERIFY_THREAD_SPLIT - 1) / FILTER_DENSE_VERIFY_THREAD_SPLIT);

	if (m_timer) m_timer->startEvent(__FUNCTION__);

	int valid = 1;
	cutilSafeCall(cudaMemcpy(d_validOpt, &valid, sizeof(int), cudaMemcpyHostToDevice));

	VerifyTrajectoryCU_Kernel << <grid, block >> >(
		numImages, d_trajectory, imageWidth, imageHeight, intrinsics,
		d_cachedFrames, distThresh, normalThresh, colorThresh, errThresh, corrThresh,
		d_validOpt, sensorDepthMin, sensorDepthMax);

	cutilSafeCall(cudaMemcpy(&valid, d_validOpt, sizeof(int), cudaMemcpyDeviceToHost));

	if (m_timer) m_timer->endEvent();

	CheckErrorCUDA(__FUNCTION__);

	return valid;
}
