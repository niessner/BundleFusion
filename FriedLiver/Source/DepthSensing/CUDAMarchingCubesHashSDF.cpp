#include "stdafx.h"

#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "CUDAMarchingCubesHashSDF.h"

extern "C" void resetMarchingCubesCUDA(MarchingCubesData& data);
extern "C" void extractIsoSurfaceCUDA(const HashDataStruct& hashData,
										 const RayCastData& rayCastData,
										 const MarchingCubesParams& params,
										 MarchingCubesData& data);

void CUDAMarchingCubesHashSDF::create(const MarchingCubesParams& params)
{
	m_params = params;
	m_data.allocate(m_params);

	resetMarchingCubesCUDA(m_data);
}

void CUDAMarchingCubesHashSDF::destroy(void)
{
	m_data.free();
}

void CUDAMarchingCubesHashSDF::copyTrianglesToCPU() {

	MarchingCubesData cpuData = m_data.copyToCPU();

	unsigned int nTriangles = *cpuData.d_numTriangles;

	//std::cout << "Marching Cubes: #triangles = " << nTriangles << std::endl;

	if (nTriangles != 0) {
		unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
		m_meshData.m_Vertices.resize(baseIdx + 3 * nTriangles);
		m_meshData.m_Colors.resize(baseIdx + 3 * nTriangles);

		vec3f* vc = (vec3f*)cpuData.d_triangles;
		for (unsigned int i = 0; i < 3 * nTriangles; i++) {
			m_meshData.m_Vertices[baseIdx + i] = vc[2 * i + 0];
			m_meshData.m_Colors[baseIdx + i] = vec4f(vc[2 * i + 1]);
		}
	}
	cpuData.free();
}

void CUDAMarchingCubesHashSDF::saveMesh(const std::string& filename, const mat4f *transform /*= NULL*/, bool overwriteExistingFile /*= false*/)
{
	std::string folder = util::directoryFromPath(filename);
	if (!util::directoryExists(folder)) {
		util::makeDirectory(folder);
	}

	std::string actualFilename = filename;
	if (!overwriteExistingFile) {
		while (util::fileExists(actualFilename)) {
			std::string path = util::directoryFromPath(actualFilename);
			std::string curr = util::fileNameFromPath(actualFilename);
			std::string ext = util::getFileExtension(curr);
			curr = util::removeExtensions(curr);
			std::string base = util::getBaseBeforeNumericSuffix(curr);
			unsigned int num = util::getNumericSuffix(curr);
			if (num == (unsigned int)-1) {
				num = 0;
			}
			actualFilename = path + base + std::to_string(num + 1) + "." + ext;
		}
	}

	//create index buffer (required for merging the triangle soup)
	m_meshData.m_FaceIndicesVertices.resize(m_meshData.m_Vertices.size());
	for (unsigned int i = 0; i < (unsigned int)m_meshData.m_Vertices.size()/3; i++) {
		m_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
		m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
		m_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
	}
	std::cout << "size before:\t" << m_meshData.m_Vertices.size() << std::endl;

	//std::cout << "saving initial mesh...";
	//MeshIOf::saveToFile("./Scans/scan_initial.ply", m_meshData);
	//std::cout << "done!" << std::endl;
	
	//m_meshData.removeDuplicateVertices();
	//m_meshData.mergeCloseVertices(0.00001f);
	std::cout << "merging close vertices... ";
	m_meshData.mergeCloseVertices(0.00001f, true);
	std::cout << "done!" << std::endl;
	std::cout << "removing duplicate faces... ";
	m_meshData.removeDuplicateFaces();
	std::cout << "done!" << std::endl;

	std::cout << "size after:\t" << m_meshData.m_Vertices.size() << std::endl;

	if (transform) {
		m_meshData.applyTransform(*transform);
	}

	std::cout << "saving mesh (" << actualFilename << ") ...";
	MeshIOf::saveToFile(actualFilename, m_meshData);
	std::cout << "done!" << std::endl;

	clearMeshBuffer();
	
}

void CUDAMarchingCubesHashSDF::extractIsoSurface(const HashDataStruct& hashData, const HashParams& hashParams, const RayCastData& rayCastData,  const vec3f& minCorner, const vec3f& maxCorner, bool boxEnabled)
{
	resetMarchingCubesCUDA(m_data);

	m_params.m_maxCorner = MatrixConversion::toCUDA(maxCorner);
	m_params.m_minCorner = MatrixConversion::toCUDA(minCorner);
	m_params.m_boxEnabled = boxEnabled;
	m_data.updateParams(m_params);

	extractIsoSurfaceCUDA(hashData, rayCastData, m_params, m_data);
	copyTrianglesToCPU();
}

void CUDAMarchingCubesHashSDF::extractIsoSurface( CUDASceneRepChunkGrid& chunkGrid, const RayCastData& rayCastData, const vec3f& camPos, float radius)
{

	chunkGrid.stopMultiThreading();

	const vec3i& minGridPos = chunkGrid.getMinGridPos();
	const vec3i& maxGridPos = chunkGrid.getMaxGridPos();

	clearMeshBuffer();

	chunkGrid.streamOutToCPUAll();

	for (int x = minGridPos.x; x < maxGridPos.x; x++)	{
		for (int y = minGridPos.y; y < maxGridPos.y; y++) {
			for (int z = minGridPos.z; z < maxGridPos.z; z++) {

				vec3i chunk(x, y, z);
				if (chunkGrid.containsSDFBlocksChunk(chunk)) {
					std::cout << "Marching Cubes on chunk (" << x << ", " << y << ", " << z << ") " << std::endl;

					chunkGrid.streamInToGPUChunkNeighborhood(chunk, 1);

					const vec3f& chunkCenter = chunkGrid.getWorldPosChunk(chunk);
					const vec3f& voxelExtends = chunkGrid.getVoxelExtends();
					float virtualVoxelSize = chunkGrid.getHashParams().m_virtualVoxelSize;

					vec3f minCorner = chunkCenter-voxelExtends/2.0f-vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;
					vec3f maxCorner = chunkCenter+voxelExtends/2.0f+vec3f(virtualVoxelSize, virtualVoxelSize, virtualVoxelSize)*(float)chunkGrid.getHashParams().m_SDFBlockSize;

					extractIsoSurface(chunkGrid.getHashData(), chunkGrid.getHashParams(), rayCastData, minCorner, maxCorner, true);

					chunkGrid.streamOutToCPUAll();
				}
			}
		}
	}

	unsigned int nStreamedBlocks;
	chunkGrid.streamInToGPUAll(camPos, radius, true, nStreamedBlocks);

	chunkGrid.startMultiThreading();
}


/*
void CUDAMarchingCubesHashSDF::extractIsoSurfaceCPU(const HashData& hashData, const HashParams& hashParams, const RayCastData& rayCastData)
{
	reset();
	m_params.m_numOccupiedSDFBlocks = hashParams.m_numOccupiedBlocks;
	m_data.updateParams(m_params);

	MarchingCubesData cpuData = m_data.copyToCPU();
	HashData		  cpuHashData = hashData.copyToCPU();

	for (unsigned int sdfBlockId = 0; sdfBlockId < m_params.m_numOccupiedSDFBlocks; sdfBlockId++) {
		for (int x = 0; x < hashParams.m_SDFBlockSize; x++) {
			for (int y = 0; y < hashParams.m_SDFBlockSize; y++) {
				for (int z = 0; z < hashParams.m_SDFBlockSize; z++) {

					const HashEntry& entry = cpuHashData.d_hashCompactified[sdfBlockId];
					if (entry.ptr != FREE_ENTRY) {
						int3 pi_base = cpuHashData.SDFBlockToVirtualVoxelPos(entry.pos);
						int3 pi = pi_base + make_int3(x,y,z);
						float3 worldPos = cpuHashData.virtualVoxelPosToWorld(pi);

						cpuData.extractIsoSurfaceAtPosition(worldPos, cpuHashData, rayCastData);
					}

				} // z
			} // y
		} // x
	} // sdf block id

	// save mesh
	{
		std::cout << "saving mesh..." << std::endl;
		std::string filename = "Scans/scan.ply";
		unsigned int nTriangles = *cpuData.d_numTriangles;

		std::cout << "marching cubes: #triangles = " << nTriangles << std::endl;

		if (nTriangles == 0) return;

		unsigned int baseIdx = (unsigned int)m_meshData.m_Vertices.size();
		m_meshData.m_Vertices.resize(baseIdx + 3*nTriangles);
		m_meshData.m_Colors.resize(baseIdx + 3*nTriangles);

		vec3f* vc = (vec3f*)cpuData.d_triangles;
		for (unsigned int i = 0; i < 3*nTriangles; i++) {
			m_meshData.m_Vertices[baseIdx + i] = vc[2*i+0];
			m_meshData.m_Colors[baseIdx + i] = vc[2*i+1];
		}

		//create index buffer (required for merging the triangle soup)
		m_meshData.m_FaceIndicesVertices.resize(nTriangles);
		for (unsigned int i = 0; i < nTriangles; i++) {
			m_meshData.m_FaceIndicesVertices[i][0] = 3*i+0;
			m_meshData.m_FaceIndicesVertices[i][1] = 3*i+1;
			m_meshData.m_FaceIndicesVertices[i][2] = 3*i+2;
		}

		//m_meshData.removeDuplicateVertices();
		//m_meshData.mergeCloseVertices(0.00001f);
		std::cout << "merging close vertices... ";
		m_meshData.mergeCloseVertices(0.00001f, true);
		std::cout << "done!" << std::endl;
		std::cout << "removing duplicate faces... ";
		m_meshData.removeDuplicateFaces();
		std::cout << "done!" << std::endl;

		std::cout << "saving mesh (" << filename << ") ...";
		MeshIOf::saveToFile(filename, m_meshData);
		std::cout << "done!" << std::endl;

		clearMeshBuffer();
	}

	cpuData.free();
}
*/
