#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>

#include "MatrixConversion.h"
#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"
#include "CUDAScan.h"
#include "CUDATimer.h"

#include "GlobalAppState.h"
#include "TimingLogDepthSensing.h"

extern "C" void resetCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void resetHashBucketMutexCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void allocCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask);
extern "C" void fillDecisionArrayCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void compactifyHashCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" unsigned int compactifyHashAllInOneCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void integrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern "C" void deIntegrateDepthMapCUDA(HashDataStruct& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern "C" void bindInputDepthColorTextures(const DepthCameraData& depthCameraData, unsigned int width, unsigned int height);

extern "C" void starveVoxelsKernelCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void garbageCollectIdentifyCUDA(HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void garbageCollectFreeCUDA(HashDataStruct& hashData, const HashParams& hashParams);

class CUDASceneRepHashSDF
{
public:
	CUDASceneRepHashSDF(const HashParams& params) {
		create(params);
	}
	~CUDASceneRepHashSDF() {
		destroy();
	}

	static HashParams parametersFromGlobalAppState(const GlobalAppState& gas) {
		HashParams params;
		params.m_rigidTransform.setIdentity();
		params.m_rigidTransformInverse.setIdentity();
		params.m_hashNumBuckets = gas.s_hashNumBuckets;
		params.m_hashBucketSize = HASH_BUCKET_SIZE;
		params.m_hashMaxCollisionLinkedListSize = gas.s_hashMaxCollisionLinkedListSize;
		params.m_SDFBlockSize = SDF_BLOCK_SIZE;
		params.m_numSDFBlocks = gas.s_hashNumSDFBlocks;
		params.m_virtualVoxelSize = gas.s_SDFVoxelSize;
		params.m_maxIntegrationDistance = gas.s_SDFMaxIntegrationDistance;
		params.m_truncation = gas.s_SDFTruncation;
		params.m_truncScale = gas.s_SDFTruncationScale;
		params.m_integrationWeightSample = gas.s_SDFIntegrationWeightSample;
		params.m_integrationWeightMax = gas.s_SDFIntegrationWeightMax;
		params.m_streamingVoxelExtents = MatrixConversion::toCUDA(gas.s_streamingVoxelExtents);
		params.m_streamingGridDimensions = MatrixConversion::toCUDA(gas.s_streamingGridDimensions);
		params.m_streamingMinGridPos = MatrixConversion::toCUDA(gas.s_streamingMinGridPos);
		params.m_streamingInitialChunkListSize = gas.s_streamingInitialChunkListSize;
		return params;
	}

	void bindDepthCameraTextures(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		bindInputDepthColorTextures(depthCameraData, depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
	}

	void integrate(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {
		
		bindDepthCameraTextures(depthCameraData, depthCameraParams);

		setLastRigidTransform(lastRigidTransform);

		//allocate all hash blocks which are corresponding to depth map entries
		alloc(depthCameraData, depthCameraParams, d_bitMask);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries();

		//volumetrically integrate the depth data into the depth SDFBlocks
		integrateDepthMap(depthCameraData, depthCameraParams);

		//garbageCollect();

		m_numIntegratedFrames++;
	}

	void deIntegrate(const mat4f& lastRigidTransform, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask) {

		bindDepthCameraTextures(depthCameraData, depthCameraParams);

		if (GlobalAppState::get().s_streamingEnabled == true) {
			MLIB_WARNING("s_streamingEnabled is no compatible with deintegration");
		}

		setLastRigidTransform(lastRigidTransform);

		//generate a linear hash array with only occupied entries
		compactifyHashEntries();

		//volumetrically integrate the depth data into the depth SDFBlocks
		deIntegrateDepthMap(depthCameraData, depthCameraParams);

		//garbageCollect();

		//DepthImage32 test(depthCameraParams.m_imageWidth, depthCameraParams.m_imageHeight);
		//cudaMemcpyFromArray(test.getData(), depthCameraData.d_depthArray, 0, 0, sizeof(float)*depthCameraParams.m_imageWidth *depthCameraParams.m_imageHeight, cudaMemcpyDeviceToHost);
		//FreeImageWrapper::saveImage("test_deint_depth" + std::to_string(m_numIntegratedFrames) + " .png", ColorImageR32G32B32(test), true);

		m_numIntegratedFrames--;
	}

	void garbageCollect() {
		//only perform if enabled by global app state
		if (GlobalAppState::get().s_garbageCollectionEnabled) {

			//if (m_numIntegratedFrames > 0 && m_numIntegratedFrames % GlobalAppState::get().s_garbageCollectionStarve == 0) {
			//	starveVoxelsKernelCUDA(m_hashData, m_hashParams);

			//	MLIB_WARNING("starving voxel weights is incompatible with bundling");
			//}

			if (m_hashParams.m_numOccupiedBlocks > 0) {
				garbageCollectIdentifyCUDA(m_hashData, m_hashParams);
				resetHashBucketMutexCUDA(m_hashData, m_hashParams);	//needed if linked lists are enabled -> for memeory deletion
				garbageCollectFreeCUDA(m_hashData, m_hashParams);
			}
		}
	}

	void setLastRigidTransform(const mat4f& lastRigidTransform) {
		m_hashParams.m_rigidTransform = MatrixConversion::toCUDA(lastRigidTransform);
		m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();

		//make the rigid transform available on the GPU
		m_hashData.updateParams(m_hashParams);
	}

	void setLastRigidTransformAndCompactify(const mat4f& lastRigidTransform) {
		setLastRigidTransform(lastRigidTransform);
		compactifyHashEntries();
	}


	const mat4f getLastRigidTransform() const {
		return MatrixConversion::toMlib(m_hashParams.m_rigidTransform);
	}

	//! resets the hash to the initial state (i.e., clears all data)
	void reset() {
		m_numIntegratedFrames = 0;

		m_hashParams.m_rigidTransform.setIdentity();
		m_hashParams.m_rigidTransformInverse.setIdentity();
		m_hashParams.m_numOccupiedBlocks = 0;
		m_hashData.updateParams(m_hashParams);
		resetCUDA(m_hashData, m_hashParams);
	}


	HashDataStruct& getHashData() {
		return m_hashData;
	} 

	const HashParams& getHashParams() const {
		return m_hashParams;
	}


	//! debug only!
	unsigned int getHeapFreeCount() {
		unsigned int count;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&count, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		return count+1;	//there is one more free than the address suggests (0 would be also a valid address)
	}

	unsigned int getNumIntegratedFrames() const {
		return m_numIntegratedFrames;
	}

	//! debug only!
	void debugHash() {
		HashEntry* hashCPU = new HashEntry[m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets];
		unsigned int* heapCPU = new unsigned int[m_hashParams.m_numSDFBlocks];
		unsigned int heapCounterCPU;

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&heapCounterCPU, m_hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		heapCounterCPU++;	//points to the first free entry: number of blocks is one more

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(heapCPU, m_hashData.d_heap, sizeof(unsigned int)*m_hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(hashCPU, m_hashData.d_hash, sizeof(HashEntry)*m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets, cudaMemcpyDeviceToHost));

		Voxel* sdfBlocksCPU = new Voxel[m_hashParams.m_numSDFBlocks*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE];
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(sdfBlocksCPU, m_hashData.d_SDFBlocks, sizeof(Voxel)*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*m_hashParams.m_numSDFBlocks, cudaMemcpyDeviceToHost));


		//Check for duplicates
		class myint3Voxel {
		public:
			myint3Voxel() {}
			~myint3Voxel() {}
			bool operator<(const myint3Voxel& other) const {
				if (x == other.x) {
					if (y == other.y) {
						return z < other.z;
					}
					return y < other.y;
				}
				return x < other.x;
			}

			bool operator==(const myint3Voxel& other) const {
				return x == other.x && y == other.y && z == other.z;
			}

			int x,y,z, i;
			int offset;
			int ptr;
		}; 


		std::unordered_set<unsigned int> pointersFreeHash;
		std::vector<unsigned int> pointersFreeVec(m_hashParams.m_numSDFBlocks, 0);
		for (unsigned int i = 0; i < heapCounterCPU; i++) {
			pointersFreeHash.insert(heapCPU[i]);
			pointersFreeVec[heapCPU[i]] = FREE_ENTRY;
		}
		if (pointersFreeHash.size() != heapCounterCPU) {
			throw MLIB_EXCEPTION("ERROR: duplicate free pointers in heap array");
		}
		 

		unsigned int numOccupied = 0;
		unsigned int numMinusOne = 0;
		unsigned int listOverallFound = 0;

		PointCloudf voxelBlocksPC;
		PointCloudf voxelPC;
		std::list<myint3Voxel> l;
		BoundingBox3<int> bboxBlocks;
		//std::vector<myint3Voxel> v;
		
		for (unsigned int i = 0; i < m_hashParams.m_hashBucketSize*m_hashParams.m_hashNumBuckets; i++) {
			if (hashCPU[i].ptr == -1) {
				numMinusOne++;
			}

			if (hashCPU[i].ptr != -2) {
				numOccupied++;	// != FREE_ENTRY
				myint3Voxel a;	
				a.x = hashCPU[i].pos.x;
				a.y = hashCPU[i].pos.y;
				a.z = hashCPU[i].pos.z;
				l.push_back(a);
				//v.push_back(a);

				unsigned int linearBlockSize = m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize*m_hashParams.m_SDFBlockSize;
				if (pointersFreeHash.find(hashCPU[i].ptr / linearBlockSize) != pointersFreeHash.end()) {
					throw MLIB_EXCEPTION("ERROR: ptr is on free heap, but also marked as an allocated entry");
				}
				pointersFreeVec[hashCPU[i].ptr / linearBlockSize] = LOCK_ENTRY;

				voxelBlocksPC.m_points.push_back(vec3f((float)a.x, (float)a.y, (float)a.z));
				bboxBlocks.include(vec3i(a.x, a.y, a.z));

				for (unsigned int z = 0; z < SDF_BLOCK_SIZE; z++) {
					for (unsigned int y = 0; y < SDF_BLOCK_SIZE; y++) {
						for (unsigned int x = 0; x < SDF_BLOCK_SIZE; x++) {
							unsigned int linearOffset = z*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE + y*SDF_BLOCK_SIZE + x;
							const Voxel& v = sdfBlocksCPU[hashCPU[i].ptr + linearOffset];
							if (v.weight > 0 && std::abs(v.sdf) <= m_hashParams.m_virtualVoxelSize) {
								vec3f pos = vec3f(vec3i(hashCPU[i].pos.x, hashCPU[i].pos.y, hashCPU[i].pos.z) * SDF_BLOCK_SIZE + vec3i(x, y, z));
								pos = pos * m_hashParams.m_virtualVoxelSize;
								voxelPC.m_points.push_back(pos);

								std::cout << "voxel weight " << v.weight << std::endl;
								std::cout << "voxel sdf " << v.sdf << std::endl;
							}
						}
					}
				}
			}
		}

		std::cout << "valid blocks found " << voxelBlocksPC.m_points.size() << std::endl;
		std::cout << "valid voxel found " << voxelPC.m_points.size() << std::endl;

		unsigned int numHeapFree = 0;
		unsigned int numHeapOccupied = 0;
		for (unsigned int i = 0; i < m_hashParams.m_numSDFBlocks; i++) {
			if		(pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
			else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
			else {
				throw MLIB_EXCEPTION("memory leak detected: neither free nor allocated");
			}
		}
		if (numHeapFree + numHeapOccupied == m_hashParams.m_numSDFBlocks) std::cout << "HEAP OK!" << std::endl;
		else throw MLIB_EXCEPTION("HEAP CORRUPTED");

		l.sort();
		size_t sizeBefore = l.size();
		l.unique();
		size_t sizeAfter = l.size();


		std::cout << "diff: " << sizeBefore - sizeAfter << std::endl;
		std::cout << "minOne: " << numMinusOne << std::endl;
		std::cout << "numOccupied: " << numOccupied << "\t numFree: " << getHeapFreeCount() << std::endl;
		std::cout << "numOccupied + free: " << numOccupied + getHeapFreeCount() << std::endl;
		std::cout << "numInFrustum: " << m_hashParams.m_numOccupiedBlocks << std::endl;

		SAFE_DELETE_ARRAY(heapCPU);
		SAFE_DELETE_ARRAY(hashCPU);

		SAFE_DELETE_ARRAY(sdfBlocksCPU);
		//getchar();
	}
private:

	void create(const HashParams& params) {
		m_hashParams = params;
		m_hashData.allocate(m_hashParams);

		reset();
	}

	void destroy() {
		m_hashData.free();
	}

	void alloc(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//resetHashBucketMutexCUDA(m_hashData, m_hashParams);
		//allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);
		 
		unsigned int prevFree = getHeapFreeCount();
		while (1) {
			resetHashBucketMutexCUDA(m_hashData, m_hashParams);
			allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);

			unsigned int currFree = getHeapFreeCount();

			if (prevFree != currFree) {
				prevFree = currFree;
			}
			else {
				break;
			}
		}

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeAlloc += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeAlloc++; }
	}


	void compactifyHashEntries() {
		//Start Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		//CUDATimer t;
		
		////t.startEvent("fillDecisionArray");
		//fillDecisionArrayCUDA(m_hashData, m_hashParams);
		////t.endEvent();

		////t.startEvent("prefixSum");
		//m_hashParams.m_numOccupiedBlocks = 
		//	m_cudaScan.prefixSum(
		//		m_hashParams.m_hashNumBuckets*m_hashParams.m_hashBucketSize,
		//		m_hashData.d_hashDecision,
		//		m_hashData.d_hashDecisionPrefix);
		////t.endEvent();

		////t.startEvent("compactifyHash");
		//m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU
		//compactifyHashCUDA(m_hashData, m_hashParams);
		////t.endEvent();

		 

		//t.startEvent("compactifyAllInOne");
		m_hashParams.m_numOccupiedBlocks = compactifyHashAllInOneCUDA(m_hashData, m_hashParams);
		m_hashData.updateParams(m_hashParams);	//make sure numOccupiedBlocks is updated on the GPU
		//t.endEvent();
		//t.evaluate();


		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeCompactifyHash += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeCompactifyHash++; }

		//std::cout << "numOccupiedBlocks: " << m_hashParams.m_numOccupiedBlocks << std::endl;
	}

	void integrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		//Start Timing
		if(GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		integrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeIntegrate += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeIntegrate++; }
	}

	void deIntegrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams) {
		//Start Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

		deIntegrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);

		// Stop Timing
		if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLogDepthSensing::totalTimeDeIntegrate += m_timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeDeIntegrate++; }
	}



	HashParams		m_hashParams;
	HashDataStruct		m_hashData;

	CUDAScan		m_cudaScan;

	unsigned int	m_numIntegratedFrames;	//used for garbage collect

	static Timer m_timer;
};
