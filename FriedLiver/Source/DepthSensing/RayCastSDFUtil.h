#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"
//#include "DepthCameraUtil.h"
#include "VoxelUtilHashSDF.h"

#include "CUDARayCastParams.h"

#ifndef __CUDACC__
#include "mLib.h"
#endif



struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
	//uint3 color;
};

#ifndef MINF
#define MINF asfloat(0xff800000)
#endif

extern __constant__ RayCastParams c_rayCastParams;
extern "C" void updateConstantRayCastParams(const RayCastParams& params);


struct RayCastData {

	///////////////
	// Host part //
	///////////////

	__device__ __host__
	RayCastData() {
		d_depth = NULL;
		d_depth4 = NULL;
		d_normals = NULL;
		d_colors = NULL;

		d_vertexBuffer = NULL;

		d_rayIntervalSplatMinArray = NULL;
		d_rayIntervalSplatMaxArray = NULL;
	}

#ifndef __CUDACC__
	__host__
	void allocate(const RayCastParams& params) {
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth4, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normals, sizeof(float4) * params.m_width * params.m_height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colors, sizeof(float4) * params.m_width * params.m_height));
	}

	__host__
	void updateParams(const RayCastParams& params) {
		updateConstantRayCastParams(params);
	}

	__host__
		void free() {
			MLIB_CUDA_SAFE_FREE(d_depth);
			MLIB_CUDA_SAFE_FREE(d_depth4);
			MLIB_CUDA_SAFE_FREE(d_normals);
			MLIB_CUDA_SAFE_FREE(d_colors);
	}
#endif

	/////////////////
	// Device part //
	/////////////////
#ifdef __CUDACC__

	__device__
		const RayCastParams& params() const {
			return c_rayCastParams;
	}

	__device__
	float frac(float val) const {
		return (val - floorf(val));
	}
	__device__
	float3 frac(const float3& val) const {
			return make_float3(frac(val.x), frac(val.y), frac(val.z));
	}
	
	__device__
	bool trilinearInterpolationSimpleFastFast(const HashDataStruct& hash, const float3& pos, float& dist, uchar3& color) const {
		const float oSet = c_hashParams.m_virtualVoxelSize;
		const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
		float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

		dist = 0.0f;
		float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
		Voxel v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, 0.0f)); if(v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*v.sdf; colorFloat+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*vColor;
		      v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*v.sdf; colorFloat+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*vColor;
		      v = hash.getVoxel(posDual+make_float3(0.0f, oSet, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*v.sdf; colorFloat+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*vColor;
		      v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *v.sdf; colorFloat+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *vColor;
		      v = hash.getVoxel(posDual+make_float3(oSet, oSet, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *	   weight.y *(1.0f-weight.z)*v.sdf; colorFloat+=	   weight.x *	   weight.y *(1.0f-weight.z)*vColor;
		      v = hash.getVoxel(posDual+make_float3(0.0f, oSet, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat+= (1.0f-weight.x)*	   weight.y *	   weight.z *vColor;
		      v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *(1.0f-weight.y)*	   weight.z *v.sdf; colorFloat+=	   weight.x *(1.0f-weight.y)*	   weight.z *vColor;
		      v = hash.getVoxel(posDual+make_float3(oSet, oSet, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat+=	   weight.x *	   weight.y *	   weight.z *vColor;

		color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;
		
		return true;
	}
	//__device__
	//bool trilinearInterpolationSimpleFastFast(const HashData& hash, const float3& pos, float& dist, uchar3& color) const {
	//	const float oSet = c_hashParams.m_virtualVoxelSize;
	//	const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
	//	float3 weight = frac(hash.worldToVirtualVoxelPosFloat(pos));

	//	dist = 0.0f;
	//	Voxel v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, 0.0f)); if(v.weight == 0) return false; dist+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, 0.0f)); if(v.weight == 0) return false;		dist+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, oSet, 0.0f)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, 0.0f, oSet)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, oSet, 0.0f)); if(v.weight == 0) return false;		dist+=	   weight.x *	   weight.y *(1.0f-weight.z)*v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(0.0f, oSet, oSet)); if(v.weight == 0) return false;		dist+= (1.0f-weight.x)*	   weight.y *	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, 0.0f, oSet)); if(v.weight == 0) return false;		dist+=	   weight.x *(1.0f-weight.y)*	   weight.z *v.sdf;
	//	v = hash.getVoxel(posDual+make_float3(oSet, oSet, oSet)); if(v.weight == 0) return false;		dist+=	   weight.x *	   weight.y *	   weight.z *v.sdf;

	//	color = v.color;

	//	return true;
	//}


	__device__
	float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const
	{
		return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
	}
	
	static const unsigned int nIterationsBisection = 3;
	
	// d0 near, d1 far
	__device__
		bool findIntersectionBisection(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) const
	{
		float a = r0; float aDist = d0;
		float b = r1; float bDist = d1;
		float c = 0.0f;

#pragma unroll 1
		for(uint i = 0; i<nIterationsBisection; i++)
		{
			c = findIntersectionLinear(a, b, aDist, bDist);

			float cDist;
			if(!trilinearInterpolationSimpleFastFast(hash, worldCamPos+c*worldDir, cDist, color)) return false;

			if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
			else { b = c; bDist = cDist; }
		}

		alpha = c;

		return true;
	}
	
	
	__device__
	float3 gradientForPoint(const HashDataStruct& hash, const float3& pos) const
	{
		const float voxelSize = c_hashParams.m_virtualVoxelSize;
		float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

		float distp00; uchar3 colorp00; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
		float dist0p0; uchar3 color0p0; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
		float dist00p; uchar3 color00p; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

		float dist100; uchar3 color100; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100, color100);
		float dist010; uchar3 color010; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010, color010);
		float dist001; uchar3 color001; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001, color001);

		float3 grad = make_float3((distp00-dist100)/offset.x, (dist0p0-dist010)/offset.y, (dist00p-dist001)/offset.z);

		float l = length(grad);
		if(l == 0.0f) {
			return make_float3(0.0f, 0.0f, 0.0f);
		}

		return -grad/l;
	}

	static __inline__ __device__
	float depthProjToCameraZ(float z)	{
		return z * (c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth) + c_rayCastParams.m_minDepth;
	}
	static __inline__ __device__
	float3 depthToCamera(unsigned int ux, unsigned int uy, float depth) 
	{
		const float x = ((float)ux-c_rayCastParams.mx) / c_rayCastParams.fx;
		const float y = ((float)uy-c_rayCastParams.my) / c_rayCastParams.fy;
		return make_float3(depth*x, depth*y, depth);
	}
	static __inline__ __device__
	float3 cameraToDepthProj(const float3& pos)	{
		float2 proj = make_float2(
			pos.x*c_rayCastParams.fx/pos.z + c_rayCastParams.mx,			
			pos.y*c_rayCastParams.fy/pos.z + c_rayCastParams.my);

		float3 pImage = make_float3(proj.x, proj.y, pos.z);

		pImage.x = (2.0f*pImage.x - (c_rayCastParams.m_width- 1.0f))/(c_rayCastParams.m_width- 1.0f);
		//pImage.y = (2.0f*pImage.y - (c_rayCastParams.m_height-1.0f))/(c_rayCastParams.m_height-1.0f);
		pImage.y = ((c_rayCastParams.m_height-1.0f) - 2.0f*pImage.y)/(c_rayCastParams.m_height-1.0f);
		pImage.z = (pImage.z - c_rayCastParams.m_minDepth)/(c_rayCastParams.m_maxDepth - c_rayCastParams.m_minDepth);

		return pImage;
	}

	__device__
	void traverseCoarseGridSimpleSampleAll(const HashDataStruct& hash, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const
	{
		const RayCastParams& rayCastParams = c_rayCastParams;

		// Last Sample
		RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
		const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
		
		float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
		float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
		//float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
		//float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength

#pragma unroll 1
		while(rayCurrent < rayEnd)
		{
			float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
			float dist;	uchar3 color;

			if(trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))
			{
				if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f)// current sample is always valid here 
				//if(lastSample.weight > 0 && ((lastSample.sdf > 0.0f && dist < 0.0f) || (lastSample.sdf < 0.0f && dist > 0.0f))) //hack for top down video
				{

					float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
					uchar3 color2;
					bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);
					
					float3 currentIso = worldCamPos+alpha*worldDir;
					if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
					{
						if(abs(dist) < rayCastParams.m_thresDist)
						{
							float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

							d_depth[dTid.y*rayCastParams.m_width+dTid.x] = depth;
							d_depth4[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(depthToCamera(dTid.x, dTid.y, depth), 1.0f);
							d_colors[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(color2.x/255.f, color2.y/255.f, color2.z/255.f, 1.0f);

							if(rayCastParams.m_useGradients)
							{
								float3 normal = -gradientForPoint(hash, currentIso);
								float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
								d_normals[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
							}

							return;
						}
					}
				}

				lastSample.sdf = dist;
				lastSample.alpha = rayCurrent;
				// lastSample.color = color;
				lastSample.weight = 1;
				rayCurrent += rayCastParams.m_rayIncrement;
			} else {
				lastSample.weight = 0;
				rayCurrent += rayCastParams.m_rayIncrement;
			}

			
		}
		
	}

#endif // __CUDACC__

	float*  d_depth;
	float4* d_depth4;
	float4* d_normals;
	float4* d_colors;

	float4* d_vertexBuffer; // ray interval splatting triangles, mapped from directx (memory lives there)

	cudaArray* d_rayIntervalSplatMinArray;
	cudaArray* d_rayIntervalSplatMaxArray;
};
