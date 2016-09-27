#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "Tables.h"

struct MarchingCubesParams {
	bool m_boxEnabled;
	float3 m_minCorner;

	unsigned int m_maxNumTriangles;
	float3 m_maxCorner;

	unsigned int m_sdfBlockSize;
	unsigned int m_hashNumBuckets;
	unsigned int m_hashBucketSize;
	float m_threshMarchingCubes;
	float m_threshMarchingCubes2;
	float3 dummy;
};



struct MarchingCubesData {

	///////////////
	// Host part //
	///////////////

	struct Vertex
	{
		float3 p;
		float3 c;
	};

	struct Triangle
	{
		Vertex v0;
		Vertex v1;
		Vertex v2;
	};

	__device__ __host__
	MarchingCubesData() {
		d_params = NULL;
		d_triangles = NULL;
		d_numTriangles = NULL;
		m_bIsOnGPU = false;
	}

	__host__
	void allocate(const MarchingCubesParams& params, bool dataOnGPU = true) {
		m_bIsOnGPU = dataOnGPU;
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaMalloc(&d_params, sizeof(MarchingCubesParams)));
			cutilSafeCall(cudaMalloc(&d_triangles, sizeof(Triangle)* params.m_maxNumTriangles));
			cutilSafeCall(cudaMalloc(&d_numTriangles, sizeof(uint)));
		}
		else {
			d_params = new MarchingCubesParams;
			d_triangles = new Triangle[params.m_maxNumTriangles];
			d_numTriangles = new uint;
		}
	}

	__host__
	void updateParams(const MarchingCubesParams& params) {
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaMemcpy(d_params, &params, sizeof(MarchingCubesParams),cudaMemcpyHostToDevice));
		} 
		else {
			*d_params = params;
		}
	}

	__host__
	void free() {
		if (m_bIsOnGPU) {
			cutilSafeCall(cudaFree(d_params));
			cutilSafeCall(cudaFree(d_triangles));
			cutilSafeCall(cudaFree(d_numTriangles));
		}
		else {
			if (d_params) delete d_params;
			if (d_triangles) delete[] d_triangles;
			if (d_numTriangles) delete d_numTriangles;
		}

		d_params = NULL;
		d_triangles = NULL;
		d_numTriangles = NULL;
	}

	__host__
	MarchingCubesData copyToCPU() const {
		MarchingCubesParams params;
		cutilSafeCall(cudaMemcpy(&params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));

		MarchingCubesData data;
		data.allocate(params, false);	// allocate the data on the CPU
		cutilSafeCall(cudaMemcpy(data.d_params, d_params, sizeof(MarchingCubesParams), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(data.d_numTriangles, d_numTriangles, sizeof(uint), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(data.d_triangles, d_triangles, sizeof(Triangle) * (params.m_maxNumTriangles), cudaMemcpyDeviceToHost));
		return data;	//TODO MATTHIAS look at this (i.e,. when does memory get destroyed ; if it's in the destructor it would kill everything here 
	}

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__

//#ifdef __CUDACC__
	__device__
//#else
//	__host__
//#endif
	void extractIsoSurfaceAtPosition(const float3& worldPos, const HashDataStruct& hashData, const RayCastData& rayCastData)
	{
		const HashParams& hashParams = c_hashParams;
		const MarchingCubesParams& params = *d_params;

		if(params.m_boxEnabled == 1) {
			if(!isInBoxAA(params.m_minCorner, params.m_maxCorner, worldPos)) return;
		}

		const float isolevel = 0.0f;

		const float P = hashParams.m_virtualVoxelSize/2.0f;
		const float M = -P;

		float3 p000 = worldPos+make_float3(M, M, M); float dist000; uchar3 color000; bool valid000 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p000, dist000, color000);
		float3 p100 = worldPos+make_float3(P, M, M); float dist100; uchar3 color100; bool valid100 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p100, dist100, color100);
		float3 p010 = worldPos+make_float3(M, P, M); float dist010; uchar3 color010; bool valid010 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p010, dist010, color010);
		float3 p001 = worldPos+make_float3(M, M, P); float dist001; uchar3 color001; bool valid001 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p001, dist001, color001);
		float3 p110 = worldPos+make_float3(P, P, M); float dist110; uchar3 color110; bool valid110 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p110, dist110, color110);
		float3 p011 = worldPos+make_float3(M, P, P); float dist011; uchar3 color011; bool valid011 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p011, dist011, color011);
		float3 p101 = worldPos+make_float3(P, M, P); float dist101; uchar3 color101; bool valid101 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p101, dist101, color101);
		float3 p111 = worldPos+make_float3(P, P, P); float dist111; uchar3 color111; bool valid111 = rayCastData.trilinearInterpolationSimpleFastFast(hashData, p111, dist111, color111);

		if(!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;

		uint cubeindex = 0;
		if(dist010 < isolevel) cubeindex += 1;
		if(dist110 < isolevel) cubeindex += 2;
		if(dist100 < isolevel) cubeindex += 4;
		if(dist000 < isolevel) cubeindex += 8;
		if(dist011 < isolevel) cubeindex += 16;
		if(dist111 < isolevel) cubeindex += 32;
		if(dist101 < isolevel) cubeindex += 64;
		if(dist001 < isolevel) cubeindex += 128;

		const float thres = params.m_threshMarchingCubes;
		float distArray[] = {dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111};
		for(uint k = 0; k < 8; k++) {
			for(uint l = 0; l < 8; l++) {
				if(distArray[k]*distArray[l] < 0.0f) {
					if(abs(distArray[k]) + abs(distArray[l]) > thres) return;
				}
				else {
					if(abs(distArray[k]-distArray[l]) > thres) return;
				}
			}
		}

		if(abs(dist000) > params.m_threshMarchingCubes2) return;
		if(abs(dist100) > params.m_threshMarchingCubes2) return;
		if(abs(dist010) > params.m_threshMarchingCubes2) return;
		if(abs(dist001) > params.m_threshMarchingCubes2) return;
		if(abs(dist110) > params.m_threshMarchingCubes2) return;
		if(abs(dist011) > params.m_threshMarchingCubes2) return;
		if(abs(dist101) > params.m_threshMarchingCubes2) return;
		if(abs(dist111) > params.m_threshMarchingCubes2) return;

		if(edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255

		Voxel v = hashData.getVoxel(worldPos);

		Vertex vertlist[12];
		if(edgeTable[cubeindex] & 1)	vertlist[0]  = vertexInterp(isolevel, p010, p110, dist010, dist110, v.color, v.color);
		if(edgeTable[cubeindex] & 2)	vertlist[1]  = vertexInterp(isolevel, p110, p100, dist110, dist100, v.color, v.color);
		if(edgeTable[cubeindex] & 4)	vertlist[2]  = vertexInterp(isolevel, p100, p000, dist100, dist000, v.color, v.color);
		if(edgeTable[cubeindex] & 8)	vertlist[3]  = vertexInterp(isolevel, p000, p010, dist000, dist010, v.color, v.color);
		if(edgeTable[cubeindex] & 16)	vertlist[4]  = vertexInterp(isolevel, p011, p111, dist011, dist111, v.color, v.color);
		if(edgeTable[cubeindex] & 32)	vertlist[5]  = vertexInterp(isolevel, p111, p101, dist111, dist101, v.color, v.color);
		if(edgeTable[cubeindex] & 64)	vertlist[6]  = vertexInterp(isolevel, p101, p001, dist101, dist001, v.color, v.color);
		if(edgeTable[cubeindex] & 128)	vertlist[7]  = vertexInterp(isolevel, p001, p011, dist001, dist011, v.color, v.color);
		if(edgeTable[cubeindex] & 256)	vertlist[8]	 = vertexInterp(isolevel, p010, p011, dist010, dist011, v.color, v.color);
		if(edgeTable[cubeindex] & 512)	vertlist[9]  = vertexInterp(isolevel, p110, p111, dist110, dist111, v.color, v.color);
		if(edgeTable[cubeindex] & 1024) vertlist[10] = vertexInterp(isolevel, p100, p101, dist100, dist101, v.color, v.color);
		if(edgeTable[cubeindex] & 2048) vertlist[11] = vertexInterp(isolevel, p000, p001, dist000, dist001, v.color, v.color);

		for(int i=0; triTable[cubeindex][i] != -1; i+=3)
		{
			Triangle t;
			t.v0 = vertlist[triTable[cubeindex][i+0]];
			t.v1 = vertlist[triTable[cubeindex][i+1]];
			t.v2 = vertlist[triTable[cubeindex][i+2]];

			appendTriangle(t);
		}
	}

	__device__
	Vertex vertexInterp(float isolevel, const float3& p1, const float3& p2, float d1, float d2, const uchar4& c1, const uchar4& c2) const
	{
		Vertex r1; r1.p = p1; r1.c = make_float3(c1.x, c1.y, c1.z) / 255.f;
		Vertex r2; r2.p = p2; r2.c = make_float3(c2.x, c2.y, c2.z) / 255.f;

		if(abs(isolevel-d1) < 0.00001f)		return r1;
		if(abs(isolevel-d2) < 0.00001f)		return r2;
		if(abs(d1-d2) < 0.00001f)			return r1;

		float mu = (isolevel - d1) / (d2 - d1);

		Vertex res;
		res.p.x = p1.x + mu * (p2.x - p1.x); // Positions
		res.p.y = p1.y + mu * (p2.y - p1.y);
		res.p.z = p1.z + mu * (p2.z - p1.z);

		res.c.x = (float)(c1.x + mu * (float)(c2.x - c1.x)) / 255.f; // Color
		res.c.y = (float)(c1.y + mu * (float)(c2.y - c1.y)) / 255.f;
		res.c.z = (float)(c1.z + mu * (float)(c2.z - c1.z)) / 255.f;

		return res;
	}

	__device__
	bool isInBoxAA(const float3& minCorner, const float3& maxCorner, const float3& pos) const
	{
		if(pos.x < minCorner.x || pos.x > maxCorner.x) return false;
		if(pos.y < minCorner.y || pos.y > maxCorner.y) return false;
		if(pos.z < minCorner.z || pos.z > maxCorner.z) return false;

		return true;
	}

//#ifdef __CUDACC__
	__device__
	uint append() {
		uint addr = atomicAdd(d_numTriangles, 1);
		//TODO check
		return addr;
	}
//#endif // __CUDACC__

//#ifdef __CUDACC__
	__device__
//#else
//	__host__
//#endif
		void appendTriangle(const Triangle& t) {
		if (*d_numTriangles >= d_params->m_maxNumTriangles) {
			*d_numTriangles = d_params->m_maxNumTriangles;
			return; // todo
		}

//#ifdef __CUDACC__
		uint addr = append();
//#else		
//		uint addr = *d_numTriangles;
//		(*d_numTriangles)++;
//#endif

		if (addr >= d_params->m_maxNumTriangles) {
			printf("marching cubes exceeded max number of triangles (addr, #tri, max#tri): (%d, %d, %d)\n", addr, *d_numTriangles, d_params->m_maxNumTriangles);
			*d_numTriangles = d_params->m_maxNumTriangles;
			return; // todo
		}

		Triangle& triangle = d_triangles[addr];
		triangle.v0 = t.v0;
		triangle.v1 = t.v1;
		triangle.v2 = t.v2;
		return;
	}
#endif // __CUDACC__

	MarchingCubesParams*	d_params;

	uint*			d_numTriangles;
	Triangle*		d_triangles;

	bool			m_bIsOnGPU;				// the class be be used on both cpu and gpu
};

