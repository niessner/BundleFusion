
#pragma once

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "CUDASceneRepHashSDF.h"

#include "BitArray.h"


struct SDFBlock : public BinaryDataSerialize<SDFBlock>
{
	Voxel data[SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];
	//int data[2*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];

	static vec3ui delinearizeVoxelIndex(uint idx) {
		uint x = idx % SDF_BLOCK_SIZE;
		uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
		uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);	
		return vec3ui(x,y,z);
	}
};


class SDFBlockDesc : public BinaryDataSerialize<SDFBlockDesc> {
public:

	SDFBlockDesc() : pos(0,0,0), ptr(-1) {}

	SDFBlockDesc(const HashEntry& hashEntry) {
		pos = MatrixConversion::toMlib(hashEntry.pos);
		ptr = hashEntry.ptr;
	}

	bool operator<(const SDFBlockDesc& other) const	{
		if(pos.x == other.pos.x) {
			if (pos.y == other.pos.y) {
				return pos.z < other.pos.z;
			}
			return pos.y < other.pos.y;
		}
		return pos.x < other.pos.x;
	}

	bool operator==(const SDFBlockDesc& other) const {
		return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
	}

	vec3i pos;
	int ptr;
};

template<>
struct std::hash<SDFBlockDesc> : public std::unary_function<SDFBlockDesc, size_t> {
	size_t operator()(const SDFBlockDesc& other) const {
		const ml::vec3i& v = other.pos;

		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		return res;
	}
};



class ChunkDesc {
public:
	ChunkDesc(unsigned int initialChunkListSize) {
		m_SDFBlocks = std::vector<SDFBlock>(); m_SDFBlocks.reserve(initialChunkListSize);
		m_ChunkDesc = std::vector<SDFBlockDesc>(); m_ChunkDesc.reserve(initialChunkListSize);
	}

	void addSDFBlock(const SDFBlockDesc& desc, const SDFBlock& data) {
		m_ChunkDesc.push_back(desc);
		m_SDFBlocks.push_back(data);
	}

	unsigned int getNElements()	{
		return (unsigned int)m_SDFBlocks.size();
	}

	SDFBlockDesc& getSDFBlockDesc(unsigned int i) {
		return m_ChunkDesc[i];
	}

	SDFBlock& getSDFBlock(unsigned int i) {
		return m_SDFBlocks[i];
	}

	void clear() {
		m_ChunkDesc.clear();
		m_SDFBlocks.clear();
	}

	bool isStreamedOut() {
		return m_SDFBlocks.size() > 0;
	}

	std::vector<SDFBlockDesc>& getSDFBlockDescs() {
		return m_ChunkDesc;
	}

	std::vector<SDFBlock>& getSDFBlocks() {
		return m_SDFBlocks;
	}

	const std::vector<SDFBlockDesc>& getSDFBlockDescs() const {
		return m_ChunkDesc;
	}

	const std::vector<SDFBlock>& getSDFBlocks() const {
		return m_SDFBlocks;
	}
	
	private:
		std::vector<SDFBlock>		m_SDFBlocks;
		std::vector<SDFBlockDesc>	m_ChunkDesc;
};


//! write to binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const ChunkDesc& chunkDesc) {
	s << chunkDesc.getSDFBlocks();
	s << chunkDesc.getSDFBlockDescs();
	return s;
}

//! read from binary stream overload
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, ChunkDesc& chunkDesc) {
	std::vector<SDFBlock>& blocks = chunkDesc.getSDFBlocks();
	std::vector<SDFBlockDesc>& descs = chunkDesc.getSDFBlockDescs();
	s >> blocks;
	s >> descs;
	return s;
}

extern "C" void integrateFromGlobalHashPass1CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint threadsPerPart, uint start, float radius, const float3& cameraPosition, uint* d_outputCounter, SDFBlockDesc* d_output);
extern "C" void integrateFromGlobalHashPass2CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint threadsPerPart, const SDFBlockDesc* d_SDFBlockDescs, Voxel* d_output, unsigned int nSDFBlocks);

extern "C" void chunkToGlobalHashPass1CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint numSDFBlockDescs, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks);
extern "C" void chunkToGlobalHashPass2CUDA(const HashParams& hashParams, const HashDataStruct& hashData, uint numSDFBlockDescs, uint heapCountPrev, const SDFBlockDesc* d_SDFBlockDescs, const Voxel* d_SDFBlocks);


LONG WINAPI StreamingFunc(LPVOID lParam);


class CUDASceneRepChunkGrid {

public:
	CUDASceneRepChunkGrid(CUDASceneRepHashSDF* sceneRepHashSDF, const vec3f& voxelExtends, const vec3i& gridDimensions, const vec3i& minGridPos, unsigned int initialChunkListSize, bool streamingEnabled, unsigned int streamOutParts)	{

		m_sceneRepHashSDF = sceneRepHashSDF;

		m_currentPart = 0;
		m_streamOutParts = streamOutParts;

		m_maxNumberOfSDFBlocksIntegrateFromGlobalHash = 100000;

		h_SDFBlockDescOutput = NULL;
		h_SDFBlockOutput = NULL;
		d_SDFBlockDescOutput = NULL;
		d_SDFBlockDescInput = NULL;
		d_SDFBlockOutput = NULL;
		d_SDFBlockInput = NULL;
		d_SDFBlockCounter = NULL;

		d_bitMask = NULL;

		s_terminateThread = true;	//by default the thread is disabled

		create(voxelExtends, gridDimensions, minGridPos, initialChunkListSize, streamingEnabled);
	}

	~CUDASceneRepChunkGrid() {
		destroy();
	}

	
	const HashDataStruct& getHashData() const {
		return m_sceneRepHashSDF->getHashData();
	}
	const HashParams& getHashParams() const {
		return m_sceneRepHashSDF->getHashParams();
	}


	// Stream Out
	void streamOutToCPUAll();
	void streamOutToCPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamOutToCPUPass0GPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
	void streamOutToCPUPass1CPU(bool multiThreaded = true);
	void integrateInChunkGrid(const int* desc, const int* block, unsigned int nSDFBlocks);

	// Stream In
	void streamInToGPUAll();
	void streamInToGPUAll(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamInToGPUChunk(const vec3i& chunkPos);
	void streamInToGPUChunkNeighborhood(const vec3i& chunkPos, int kernelRadius);
	void streamInToGPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamInToGPUPass0CPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
	void streamInToGPUPass1GPU(bool multiThreaded = true);

	unsigned int integrateInHash(const vec3f& posCamera, float radius, bool useParts);

	void debugCheckForDuplicates() const;
	void debugDump() const {
		const HashParams& hashParams = m_sceneRepHashSDF->getHashParams();
		float thresh = hashParams.m_virtualVoxelSize;

		std::vector<vec3f> hashPoints;
		std::vector<vec3f> voxelPoints;
		for (unsigned int i = 0; i < m_grid.size(); i++) {
			if (m_grid[i] != NULL)	{
				std::vector<SDFBlockDesc>& descs = m_grid[i]->getSDFBlockDescs();
				std::vector<SDFBlock>& blocks = m_grid[i]->getSDFBlocks();
				unsigned int linearBlockSize = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

				for (unsigned int k = 0; k < descs.size(); k++) {
					vec3i pos(descs[k].pos);
					vec3f posWorld = vec3f(pos*SDF_BLOCK_SIZE)*hashParams.m_virtualVoxelSize;
					hashPoints.push_back(posWorld);
					for (unsigned int l = 0; l < linearBlockSize; l++) {
						if (blocks[k].data[l].weight > 0 && std::fabsf(blocks[k].data[l].sdf) <= thresh) {
							vec3i posUI = SDFBlock::delinearizeVoxelIndex(l) + pos;
							vec3f posWorld = vec3f(posUI*SDF_BLOCK_SIZE)*hashParams.m_virtualVoxelSize;
							voxelPoints.push_back(posWorld);
						}
					}
				}				
			}
		}

		std::cout << "hashPoints: " << hashPoints.size() << std::endl;
		std::cout << "voxelPoints: " << voxelPoints.size() << std::endl; 
		PointCloudIOf::saveToFile("hashPoints.ply", hashPoints);
		PointCloudIOf::saveToFile("voxelPoints.ply", voxelPoints);
	}

	void startMultiThreading() {

		initializeCriticalSection();

		s_terminateThread = false;

		hStreamingThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)StreamingFunc, (LPVOID)this, 0, &dwStreamingThreadID);
		 
		if (hStreamingThread == NULL) {
			std::cout << "Thread GPU-CPU could not be created" << std::endl;
		}
	}

	void stopMultiThreading() {

		if (!s_terminateThread) {
			//WaitForSingleObject(hEventOutProduce, INFINITE);
			//WaitForSingleObject(hMutexOut, INFINITE);

			//WaitForSingleObject(hEventInConsume, INFINITE);
			//WaitForSingleObject(hMutexIn, INFINITE);


			s_terminateThread = true;

			SetEvent(hEventOutProduce);
			SetEvent(hEventOutConsume);

			SetEvent(hEventInProduce);
			SetEvent(hEventInConsume);

			WaitForSingleObject(hStreamingThread, INFINITE);

			if (CloseHandle(hStreamingThread) == 0) {
				std::cout << "Thread Handle GPU CPU could not be closed" << std::endl;
			}

			// Mutex
			deleteCritialSection();
		}
	}

	void clearGrid() {
		for (unsigned int i = 0; i<m_grid.size(); i++) {
			SAFE_DELETE(m_grid[i]);
		}
	}

	void reset() {

		stopMultiThreading();
		clearGrid();
		m_bitMask.reset();
		startMultiThreading();

	}

	//! Caution maps the buffer and performs the copy
	unsigned int* getBitMaskGPU()	{
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_bitMask, m_bitMask.getRawData(), m_bitMask.getByteWidth(), cudaMemcpyHostToDevice));
		return d_bitMask;
	}

	bool containsSDFBlocksChunk(const vec3i& chunk) const {
		unsigned int index = linearizeChunkPos(chunk);
		return ((m_grid[index] != 0) && m_grid[index]->isStreamedOut());
	}

	bool isChunkInSphere(const vec3i& chunk, const vec3f& center, float radius) const {
		vec3f posWorld = chunkToWorld(chunk);
		vec3f offset = m_voxelExtends/2.0f;

		for (int x = -1; x<=1; x+=2)	{
			for (int y = -1; y<=1; y+=2)	{
				for (int z = -1; z<=1; z+=2)	{
					vec3f p = vec3f(posWorld.x+x*offset.x, posWorld.y+y*offset.y, posWorld.z+z*offset.z);
					float d = (p-center).length();

					if (d > radius) {
						return false;
					}
				}
			}
		}

		return true;
	}

	bool containsSDFBlocksChunkInRadius(const vec3i& chunk, int chunkRadius) {
		vec3i startChunk = vec3i(std::max(chunk.x-chunkRadius, m_minGridPos.x), std::max(chunk.y-chunkRadius, m_minGridPos.y), std::max(chunk.z-chunkRadius, m_minGridPos.z));
		vec3i endChunk = vec3i(std::min(chunk.x+chunkRadius, m_maxGridPos.x), std::min(chunk.y+chunkRadius, m_maxGridPos.y), std::min(chunk.z+chunkRadius, m_maxGridPos.z));

		for (int x = startChunk.x; x<=endChunk.x; x++) {
			for (int y = startChunk.y; y<=endChunk.y; y++) {
				for (int z = startChunk.z; z<=endChunk.z; z++) {
					unsigned int index = linearizeChunkPos(vec3i(x, y, z));
					if ((m_grid[index] != 0) && m_grid[index]->isStreamedOut()) {
						return true;
					}
				}
			}
		}

		return false;
	}

	void create(const vec3f& voxelExtends, const vec3i& gridDimensions, const vec3i& minGridPos, unsigned int initialChunkListSize, bool streamingEnabled) {

		m_voxelExtends = voxelExtends;
		m_gridDimensions = gridDimensions;
		m_initialChunkDescListSize = initialChunkListSize;

		m_minGridPos = minGridPos;
		m_maxGridPos = -m_minGridPos;

		m_grid.resize(m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z, NULL);

		m_bitMask = BitArray<unsigned int>(m_gridDimensions.x*m_gridDimensions.y*m_gridDimensions.z);


		MLIB_CUDA_SAFE_CALL(cudaHostAlloc(&h_SDFBlockDescOutput, sizeof(SDFBlockDesc)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash, cudaHostAllocDefault));
		MLIB_CUDA_SAFE_CALL(cudaHostAlloc(&h_SDFBlockOutput, sizeof(SDFBlock)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash, cudaHostAllocDefault));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_SDFBlockDescOutput, sizeof(SDFBlockDesc)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_SDFBlockDescInput, sizeof(SDFBlockDesc)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_SDFBlockOutput, sizeof(SDFBlock)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_SDFBlockInput, sizeof(SDFBlock)*m_maxNumberOfSDFBlocksIntegrateFromGlobalHash));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_SDFBlockCounter, sizeof(unsigned int)));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_bitMask, m_bitMask.getByteWidth()));

		if (streamingEnabled) startMultiThreading();

	}

	void destroy() 
	{
		stopMultiThreading();

		clearGrid();

		MLIB_CUDA_SAFE_CALL(cudaFreeHost(h_SDFBlockDescOutput));
		MLIB_CUDA_SAFE_CALL(cudaFreeHost(h_SDFBlockOutput));

		MLIB_CUDA_SAFE_CALL(cudaFree(d_SDFBlockDescOutput));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_SDFBlockDescInput));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_SDFBlockOutput));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_SDFBlockInput));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_SDFBlockCounter));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_bitMask));
	}

	const vec3i& getMinGridPos() const {
		return m_minGridPos;
	}

	const vec3i& getMaxGridPos() const {
		return m_maxGridPos;
	}

	const vec3f& getVoxelExtends() const {
		return m_voxelExtends;
	}

	vec3f getWorldPosChunk(const vec3i& chunk) const {
		return chunkToWorld(chunk);
	}		

	void printStatistics() const {
		unsigned int nChunks = (unsigned int)m_grid.size();

		unsigned int nSDFBlocks = 0;
		for (unsigned int i = 0; i < nChunks; i++) {
			if (m_grid[i] != NULL) {
				nSDFBlocks+=m_grid[i]->getNElements();
			}
		}

		std::cout << "Total number of Blocks on the CPU: " << nSDFBlocks << std::endl;
	}

	//void setPositionAndRadius(const vec3f& position, float radius, bool multiThreaded) {
	//	if (multiThreaded) {
	//		WaitForSingleObject(hEventSetTransformProduce, INFINITE);
	//		WaitForSingleObject(hMutexSetTransform, INFINITE);
	//	}

	//	s_posCamera = position;
	//	s_radius = radius;

	//	if (multiThreaded) {
	//		SetEvent(hEventSetTransformConsume);
	//		ReleaseMutex(hMutexSetTransform);
	//	}
	//}


	//! saves the entire state of the mesh to disc (including the GPU part)
	void saveToFile(const std::string& filename, const RayCastData& rayCastData, const vec3f& camPos, float radius) {
		
		stopMultiThreading();

		streamOutToCPUAll();

		BinaryDataStreamFile outStream(filename, true);

		unsigned int numOccupiedChunks = 0;
		for (unsigned int i = 0; i < m_grid.size(); i++) {
			if (m_grid[i]) numOccupiedChunks++;
		}
		outStream << numOccupiedChunks;

		for (unsigned int i = 0; i < m_grid.size(); i++) {
			ChunkDesc* desc = m_grid[i];
			if (desc) {
				outStream << i << *desc;
			}
		}

		outStream.close();


		unsigned int nStreamedBlocks;
		streamInToGPUAll(camPos, radius, true, nStreamedBlocks);

		startMultiThreading();
	}


	void loadFromFile(const std::string& filename, const RayCastData& rayCastData, const vec3f& camPos, float radius) {
		stopMultiThreading();


		streamOutToCPUAll();
		clearGrid();
		m_bitMask.reset();

		BinaryDataStreamFile inStream(filename, false);
		unsigned int numOccupiedChunks = 0;
		inStream >> numOccupiedChunks;

		for (unsigned int i = 0; i < numOccupiedChunks; i++) {
			unsigned int index = 0;
			inStream >> index;
			m_grid[index] = new ChunkDesc(m_initialChunkDescListSize);
			inStream >> *m_grid[index];
		}
		inStream.close();

		startMultiThreading();
	}


	const vec3f& getPosCamera() const {
		return s_posCamera;
	}

	float getRadius() const {
		return s_radius;
	}

	bool getTerminatedThread() const {
		return s_terminateThread;
	}

	private:

	//-------------------------------------------------------
	// Helper
	//-------------------------------------------------------

	bool isValidChunk(const vec3i& chunk) const	{
		if(chunk.x < m_minGridPos.x || chunk.y < m_minGridPos.y || chunk.z < m_minGridPos.z) return false;
		if(chunk.x > m_maxGridPos.x || chunk.y > m_maxGridPos.y || chunk.z > m_maxGridPos.z) return false;

		return true;
	}

	float getChunkRadiusInMeter() const {
		return m_voxelExtends.length()/2.0f;
	}

	float getGridRadiusInMeter() const {
		vec3f minPos = chunkToWorld(m_minGridPos)-m_voxelExtends/2.0f;
		vec3f maxPos = chunkToWorld(m_maxGridPos)+m_voxelExtends/2.0f;

		return (minPos-maxPos).length()/2.0f;
	}

	vec3f numberOfChunksToMeter(const vec3i& c) const {
		return vec3f(c.x*m_voxelExtends.x, c.y*m_voxelExtends.y, c.z*m_voxelExtends.z);
	}

	vec3f meterToNumberOfChunks(float f) const {
		return vec3f(f/m_voxelExtends.x, f/m_voxelExtends.y, f/m_voxelExtends.z);
	}

	vec3i meterToNumberOfChunksCeil(float f) const {
		return vec3i((int)ceil(f/m_voxelExtends.x), (int)ceil(f/m_voxelExtends.y), (int)ceil(f/m_voxelExtends.z));
	}

	vec3i worldToChunks(const vec3f& posWorld) const {
		vec3f p;
		p.x = posWorld.x/m_voxelExtends.x;
		p.y = posWorld.y/m_voxelExtends.y;
		p.z = posWorld.z/m_voxelExtends.z;

		vec3f s;
		s.x = (float)math::sign(p.x);
		s.y = (float)math::sign(p.y);
		s.z = (float)math::sign(p.z);

		return vec3i(p+s*0.5f);
	}

	vec3f getChunkCenter(const vec3i& chunk) const {
		return chunkToWorld(chunk);
	}

	vec3f chunkToWorld(const vec3i& posChunk) const	{
		vec3f res;
		res.x = posChunk.x*m_voxelExtends.x;
		res.y = posChunk.y*m_voxelExtends.y;
		res.z = posChunk.z*m_voxelExtends.z;

		return res;
	}

	vec3i delinearizeChunkIndex(unsigned int idx)	{
		unsigned int x = idx % m_gridDimensions.x;
		unsigned int y = (idx % (m_gridDimensions.x * m_gridDimensions.y)) / m_gridDimensions.x;
		unsigned int z = idx / (m_gridDimensions.x * m_gridDimensions.y);

		return m_minGridPos+vec3i(x,y,z);
	}

	unsigned int linearizeChunkPos(const vec3i& chunkPos) const {
		vec3ui p = chunkPos-m_minGridPos;

		return  p.z * m_gridDimensions.x * m_gridDimensions.y +
			p.y * m_gridDimensions.x +
			p.x;
	}

	// Mutex
	void deleteCritialSection() {
		CloseHandle(hMutexOut);
		CloseHandle(hEventOutProduce);
		CloseHandle(hEventOutConsume);

		CloseHandle(hMutexIn);
		CloseHandle(hEventInProduce);
		CloseHandle(hEventInConsume);

		CloseHandle(hMutexSetTransform);
		CloseHandle(hEventSetTransformProduce);
		CloseHandle(hEventSetTransformConsume);
	}

	void initializeCriticalSection() {
		hMutexOut = CreateMutex(NULL, FALSE, NULL);
		hEventOutProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
		hEventOutConsume = CreateEvent(NULL, FALSE, FALSE, NULL);

		hMutexIn = CreateMutex(NULL, FALSE, NULL);
		hEventInProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
		hEventInConsume = CreateEvent(NULL, FALSE, FALSE, NULL); //! has to be TRUE if stream out and in calls are split !!!

		hMutexSetTransform = CreateMutex(NULL, FALSE, NULL);
		hEventSetTransformProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
		hEventSetTransformConsume = CreateEvent(NULL, FALSE, FALSE, NULL);
	}


	void clearSDFBlockCounter() {
		unsigned int src = 0;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_SDFBlockCounter, &src, sizeof(unsigned int), cudaMemcpyHostToDevice));
	}

	unsigned int getSDFBlockCounter() const {
		unsigned int dest;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&dest, d_SDFBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return dest;
	}



	unsigned int m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;

	SDFBlockDesc*	h_SDFBlockDescOutput;
	SDFBlock*		h_SDFBlockOutput;
	
	SDFBlockDesc*	d_SDFBlockDescOutput;
	SDFBlockDesc*	d_SDFBlockDescInput;
	SDFBlock*		d_SDFBlockOutput;
	SDFBlock*		d_SDFBlockInput;
	unsigned int*	d_SDFBlockCounter;


	unsigned int*	d_bitMask;

	//-------------------------------------------------------
	// Chunk Grid
	//-------------------------------------------------------

	vec3f m_voxelExtends;		// extend of the voxels in meters
	vec3i m_gridDimensions;	    // number of voxels in each dimension

	vec3i m_minGridPos;
	vec3i m_maxGridPos;

	unsigned int m_initialChunkDescListSize;	 // initial size for vectors in the ChunkDesc

	std::vector<ChunkDesc*>	m_grid; // Grid data
	BitArray<unsigned int>	m_bitMask;

	unsigned int m_currentPart;
	unsigned int m_streamOutParts;

	// Multi-threading
	HANDLE hStreamingThread;
	DWORD dwStreamingThreadID;

	// Mutex
	HANDLE hMutexOut;
	HANDLE hEventOutProduce;
	HANDLE hEventOutConsume;

	HANDLE hMutexIn;
	HANDLE hEventInProduce;
	HANDLE hEventInConsume;

	HANDLE hMutexSetTransform;
	HANDLE hEventSetTransformProduce;
	HANDLE hEventSetTransformConsume;

	Timer m_timer;
	

	///////////////////////
	// Runtime variables //
	///////////////////////

	vec3f			s_posCamera;
	float			s_radius;
	unsigned int	s_nStreamdInBlocks;
	unsigned int	s_nStreamdOutBlocks;
	bool			s_terminateThread;

	CUDASceneRepHashSDF*	m_sceneRepHashSDF;
};

