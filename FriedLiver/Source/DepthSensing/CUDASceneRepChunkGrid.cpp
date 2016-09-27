
#include "stdafx.h"

#include "CUDASceneRepChunkGrid.h"

LONG WINAPI StreamingFunc(LPVOID lParam) 
{
	CUDASceneRepChunkGrid* chunkGrid = (CUDASceneRepChunkGrid*)lParam;

	while (true)	{
		//std::cout <<" Shouldnt run" << std::endl;
		HRESULT hr = S_OK;

		chunkGrid->streamOutToCPUPass1CPU(true);
		chunkGrid->streamInToGPUPass0CPU(chunkGrid->getPosCamera(), chunkGrid->getRadius(), true);


		if (chunkGrid->getTerminatedThread()) {
			return 0;
		}
	}

	return 0;
}

void CUDASceneRepChunkGrid::streamOutToCPUAll()
{
	unsigned int nStreamedBlocksSum = 1;
	while(nStreamedBlocksSum != 0) {
		nStreamedBlocksSum = 0;
		for (unsigned int i = 0; i < m_streamOutParts; i++) {
			unsigned int nStreamedBlocks = 0;
			streamOutToCPU(worldToChunks(m_minGridPos-vec3i(1, 1, 1)), 0.0f, true, nStreamedBlocks);

			nStreamedBlocksSum += nStreamedBlocks;
		}
	}
}

void CUDASceneRepChunkGrid::streamOutToCPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks )
{
	s_posCamera = posCamera;
	s_radius = radius;

	streamOutToCPUPass0GPU(posCamera, radius, useParts, false);
	streamOutToCPUPass1CPU(false);

	nStreamedBlocks = s_nStreamdOutBlocks;
}

void CUDASceneRepChunkGrid::streamOutToCPUPass0GPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded /*= true*/ )
{
	if (multiThreaded) {
		WaitForSingleObject(hEventOutProduce, INFINITE);
		WaitForSingleObject(hMutexOut, INFINITE);
	}

	s_posCamera = posCamera;
	s_radius = radius;

	resetHashBucketMutexCUDA(m_sceneRepHashSDF->getHashData(), m_sceneRepHashSDF->getHashParams());
	clearSDFBlockCounter();

	const unsigned int hashNumBuckets = m_sceneRepHashSDF->getHashParams().m_hashNumBuckets;
	const unsigned int hashBucketSize = m_sceneRepHashSDF->getHashParams().m_hashBucketSize;

	//-------------------------------------------------------
	// Pass 1: Find all SDFBlocks that have to be transfered
	//-------------------------------------------------------

	unsigned int threadsPerPart = (hashNumBuckets*hashBucketSize + m_streamOutParts - 1) / m_streamOutParts;
	if (!useParts) threadsPerPart = hashNumBuckets*hashBucketSize;

	uint start = m_currentPart*threadsPerPart;
	integrateFromGlobalHashPass1CUDA(m_sceneRepHashSDF->getHashParams(), m_sceneRepHashSDF->getHashData(), threadsPerPart, start, radius, MatrixConversion::toCUDA(posCamera), d_SDFBlockCounter, d_SDFBlockDescOutput);

	const unsigned int nSDFBlockDescs = getSDFBlockCounter();

	if (useParts) m_currentPart = (m_currentPart+1) % m_streamOutParts;

	if (nSDFBlockDescs != 0) {
		//std::cout << "SDFBlocks streamed out: " << nSDFBlockDescs << std::endl;

		//-------------------------------------------------------
		// Pass 2: Copy SDFBlocks to output buffer
		//-------------------------------------------------------

		integrateFromGlobalHashPass2CUDA(m_sceneRepHashSDF->getHashParams(), m_sceneRepHashSDF->getHashData(), threadsPerPart, d_SDFBlockDescOutput, (Voxel*)d_SDFBlockOutput, nSDFBlockDescs);


		MLIB_CUDA_SAFE_CALL(cudaMemcpy(h_SDFBlockDescOutput, d_SDFBlockDescOutput, sizeof(SDFBlockDesc)*nSDFBlockDescs, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(h_SDFBlockOutput, d_SDFBlockOutput, sizeof(SDFBlock)*nSDFBlockDescs, cudaMemcpyDeviceToHost));
	}

	s_nStreamdOutBlocks = nSDFBlockDescs;

	if (multiThreaded) {
		SetEvent(hEventOutConsume);
		ReleaseMutex(hMutexOut);
	}
}

void CUDASceneRepChunkGrid::streamOutToCPUPass1CPU(bool multiThreaded /*= true*/ )
{
	if (multiThreaded) {
		WaitForSingleObject(hEventOutConsume, INFINITE);
		WaitForSingleObject(hMutexOut, INFINITE);

		if (s_terminateThread)	return;		//avoid duplicate insertions when stop multithreading is called
	}

	if (s_nStreamdOutBlocks != 0) {
		integrateInChunkGrid((int*)h_SDFBlockDescOutput, (int*)h_SDFBlockOutput, s_nStreamdOutBlocks);
	}

	if (multiThreaded) {
		SetEvent(hEventOutProduce);
		ReleaseMutex(hMutexOut);
	}
}

void CUDASceneRepChunkGrid::integrateInChunkGrid(const int* desc, const int* block, unsigned int nSDFBlocks)
{
	const HashParams& hashParams = m_sceneRepHashSDF->getHashParams();
	const unsigned int descSize = 4;

	for (unsigned int i = 0; i < nSDFBlocks; i++) {
		vec3i pos(&desc[i*descSize]);
		//vec3f posWorld = VoxelUtilHelper::SDFBlockToWorld(pos);
		vec3f posWorld = vec3f(pos*SDF_BLOCK_SIZE)*hashParams.m_virtualVoxelSize;
		vec3i chunk = worldToChunks(posWorld);

		if (!isValidChunk(chunk)) {
			std::cout << "Chunk out of bounds" << std::endl;
			continue;
		}

		unsigned int index = linearizeChunkPos(chunk);

		if (m_grid[index] == NULL) // Allocate memory for chunk
		{
			m_grid[index] = new ChunkDesc(m_initialChunkDescListSize);
		}

		// Add element
		m_grid[index]->addSDFBlock(((const SDFBlockDesc*)desc)[i], ((const SDFBlock*)block)[i]);
		m_bitMask.setBit(index);
	}
}

void CUDASceneRepChunkGrid::streamInToGPUAll()
{
	unsigned int nStreamedBlocks = 1;
	while (nStreamedBlocks != 0) {
		streamInToGPU(getChunkCenter(vec3i(0, 0, 0)), 1.1f*getGridRadiusInMeter(), true, nStreamedBlocks);
	}
}

void CUDASceneRepChunkGrid::streamInToGPUAll(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks )
{
	unsigned int nStreamedBlocksSum = 0;
	unsigned int nBlock = 1;
	while (nBlock != 0) // Should not be necessary
	{
		streamInToGPU(posCamera, radius, useParts, nBlock);
		nStreamedBlocksSum += nBlock;
	}
	nStreamedBlocks = nStreamedBlocksSum;
}

void CUDASceneRepChunkGrid::streamInToGPUChunk(const vec3i& chunkPos )
{
	unsigned int nStreamedBlocks = 1;
	while (nStreamedBlocks != 0) // Should not be necessary
	{
		streamInToGPU(getChunkCenter(chunkPos), 1.1f*getChunkRadiusInMeter(), true, nStreamedBlocks);
	}
}

void CUDASceneRepChunkGrid::streamInToGPUChunkNeighborhood(const vec3i& chunkPos, int kernelRadius )
{
	vec3i startChunk = vec3i(std::max(chunkPos.x-kernelRadius, m_minGridPos.x), std::max(chunkPos.y-kernelRadius, m_minGridPos.y), std::max(chunkPos.z-kernelRadius, m_minGridPos.z));
	vec3i endChunk = vec3i(std::min(chunkPos.x+kernelRadius, m_maxGridPos.x), std::min(chunkPos.y+kernelRadius, m_maxGridPos.y), std::min(chunkPos.z+kernelRadius, m_maxGridPos.z));

	for (int x = startChunk.x; x<endChunk.x; x++) {
		for (int y = startChunk.y; y<endChunk.y; y++) {
			for (int z = startChunk.z; z<endChunk.z; z++) {
				streamInToGPUChunk(vec3i(x, y, z));
			}
		}
	}
}

void CUDASceneRepChunkGrid::streamInToGPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks )
{
	s_posCamera = posCamera;
	s_radius = radius;

	streamInToGPUPass0CPU(posCamera, radius, useParts, false);
	streamInToGPUPass1GPU(false);

	nStreamedBlocks = s_nStreamdInBlocks;
}

void CUDASceneRepChunkGrid::streamInToGPUPass0CPU( const vec3f& posCamera, float radius, bool useParts, bool multiThreaded /*= true*/ )
{
	if (multiThreaded) {
		WaitForSingleObject(hEventInProduce, INFINITE);
		WaitForSingleObject(hMutexIn, INFINITE);
		if (s_terminateThread)	return;	//avoid duplicate insertions when stop multithreading is called
	}

	unsigned int nSDFBlockDescs = integrateInHash(posCamera, radius, useParts);

	s_nStreamdInBlocks = nSDFBlockDescs;


	if (multiThreaded) {
		SetEvent(hEventInConsume);
		ReleaseMutex(hMutexIn);
	}
}

void CUDASceneRepChunkGrid::streamInToGPUPass1GPU( bool multiThreaded /*= true*/ )
{
	if (multiThreaded) {
		WaitForSingleObject(hEventInConsume, INFINITE);
		WaitForSingleObject(hMutexIn, INFINITE);
	}

	if (s_nStreamdInBlocks != 0) {
		//std::cout << "SDFBlocks streamed in: " << s_nStreamdInBlocks << std::endl;

		//-------------------------------------------------------
		// Pass 1: Alloc memory for chunks
		//-------------------------------------------------------

		unsigned int heapFreeCountPrev = m_sceneRepHashSDF->getHeapFreeCount();

		unsigned int heapCountPrev;	//pointer to the first free block
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&heapCountPrev, m_sceneRepHashSDF->getHashData().d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		chunkToGlobalHashPass1CUDA(m_sceneRepHashSDF->getHashParams(), m_sceneRepHashSDF->getHashData(), s_nStreamdInBlocks, heapCountPrev, d_SDFBlockDescInput, (Voxel*)d_SDFBlockInput);


		//-------------------------------------------------------
		// Pass 2: Initialize corresponding SDFBlocks
		//-------------------------------------------------------

		chunkToGlobalHashPass2CUDA(m_sceneRepHashSDF->getHashParams(), m_sceneRepHashSDF->getHashData(), s_nStreamdInBlocks, heapCountPrev, d_SDFBlockDescInput, (Voxel*)d_SDFBlockInput);

		//Update heap counter
		unsigned int initialCountNew = heapCountPrev-s_nStreamdInBlocks;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_sceneRepHashSDF->getHashData().d_heapCounter, &initialCountNew, sizeof(unsigned int), cudaMemcpyHostToDevice));

	}

	if (multiThreaded) {
		SetEvent(hEventInProduce);
		ReleaseMutex(hMutexIn);
	}
}

unsigned int CUDASceneRepChunkGrid::integrateInHash( const vec3f& posCamera, float radius, bool useParts )
{
	const unsigned int blockSize = sizeof(SDFBlock)/sizeof(int);
	const unsigned int descSize = sizeof(SDFBlockDesc)/sizeof(int);

	vec3i camChunk = worldToChunks(posCamera);
	vec3i chunkRadius = meterToNumberOfChunksCeil(radius);
	vec3i startChunk = vec3i(std::max(camChunk.x-chunkRadius.x, m_minGridPos.x), std::max(camChunk.y-chunkRadius.y, m_minGridPos.y), std::max(camChunk.z-chunkRadius.z, m_minGridPos.z));
	vec3i endChunk = vec3i(std::min(camChunk.x+chunkRadius.x, m_maxGridPos.x), std::min(camChunk.y+chunkRadius.y, m_maxGridPos.y), std::min(camChunk.z+chunkRadius.z, m_maxGridPos.z));

	unsigned int nSDFBlocks = 0;
	for (int x = startChunk.x; x <= endChunk.x; x++) {
		for (int y = startChunk.y; y <= endChunk.y; y++) {
			for (int z = startChunk.z; z <= endChunk.z; z++) {

				unsigned int index = linearizeChunkPos(vec3i(x, y, z));
				if (m_grid[index] != NULL && m_grid[index]->isStreamedOut()) // As been allocated and has streamed out blocks
				{
					if (isChunkInSphere(delinearizeChunkIndex(index), posCamera, radius)) // Is in camera range
					{
						unsigned int nBlock = m_grid[index]->getNElements();
						if (nBlock > m_maxNumberOfSDFBlocksIntegrateFromGlobalHash) {
							throw MLIB_EXCEPTION("not enough memory allocated for intermediate GPU buffer");
						}
						// Copy data to GPU
						MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_SDFBlockDescInput, &(m_grid[index]->getSDFBlockDescs()[0]), sizeof(SDFBlockDesc)*nBlock, cudaMemcpyHostToDevice));
						MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_SDFBlockInput, &(m_grid[index]->getSDFBlocks()[0]), sizeof(SDFBlock)*nBlock, cudaMemcpyHostToDevice));

						// Remove data from CPU
						m_grid[index]->clear();
						m_bitMask.resetBit(index);

						nSDFBlocks += nBlock;

						if (useParts) return nSDFBlocks; // only in one chunk per frame
					}
				}
			}
		}
	}
	return nSDFBlocks;
}

void CUDASceneRepChunkGrid::debugCheckForDuplicates() const
{
	std::unordered_set<SDFBlockDesc> descHash;

	unsigned int numHashEntries = m_sceneRepHashSDF->getHashParams().m_hashBucketSize * m_sceneRepHashSDF->getHashParams().m_hashNumBuckets;
	HashEntry* hashCPU = new HashEntry[numHashEntries];
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(hashCPU, m_sceneRepHashSDF->getHashData().d_hash, sizeof(HashEntry)*numHashEntries, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i < numHashEntries; i++) {
		if (hashCPU[i].ptr != FREE_ENTRY) {
			SDFBlockDesc curr(hashCPU[i]);
			if (descHash.find(curr) == descHash.end()) descHash.insert(curr);
			else throw MLIB_EXCEPTION("Duplicate found in streaming hash data (in hash)");
		}

	}
	SAFE_DELETE_ARRAY(hashCPU);

	for (unsigned int i = 0; i < m_grid.size(); i++) {
		if (m_grid[i] != NULL)	{
			std::vector<SDFBlockDesc>& descsCopy = m_grid[i]->getSDFBlockDescs();

			for (unsigned int k = 0; k < descsCopy.size(); k++) {	
				if (descHash.find(descsCopy[k]) == descHash.end()) descHash.insert(descsCopy[k]);
				else throw MLIB_EXCEPTION("Duplicate found in streaming hash data (in grid)");
			}
		}
	}
	std::cout << __FUNCTION__ " : OK!" << std::endl;
}
