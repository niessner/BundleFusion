#pragma once

#include "CUDACacheUtil.h"
#include "CUDAImageUtil.h"

class CUDACache {
public:

	CUDACache(unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& intrinsics);
	~CUDACache() { 
		free(); 
	}

	void storeFrame(const float* d_depth, unsigned int inputDepthWidth, unsigned int inputDepthHeight,
		const uchar4* d_color, unsigned int inputColorWidth, unsigned int inputColorHeight);

	void reset() {
		m_currentFrame = 0;
	}

	const std::vector<CUDACachedFrame>& getCacheFrames() const { return m_cache; }
	const CUDACachedFrame* getCacheFramesGPU() const { return d_cache; }

	void copyCacheFrameFrom(CUDACache* other, unsigned int frameFrom) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_depthDownsampled, other->m_cache[frameFrom].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_colorDownsampled, other->m_cache[frameFrom].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_cameraposDownsampled, other->m_cache[frameFrom].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampled, other->m_cache[frameFrom].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

		m_currentFrame++;
	}

	//! for invalid (global) frames don't need to copy
	void incrementCache() {
		m_currentFrame++;
	}

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }

	const mat4f& getIntrinsics() const { return m_intrinsics; }
	const mat4f& getIntrinsicsInv() const { return m_intrinsicsInv; }

	//! warning: untested!
	void saveToFile(const std::string& filename) const {
		BinaryDataStreamFile s(filename, true);
		s << m_width;
		s << m_height;
		s << m_intrinsics;
		s << m_intrinsicsInv;
		s << m_currentFrame;
		s << m_maxNumImages;
		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		ColorImageR8G8B8A8 color(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getPointer(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPos.getPointer(), f.d_cameraposDownsampled, sizeof(float4)*camPos.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(normals.getPointer(), f.d_normalsDownsampled, sizeof(float4)*normals.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(color.getPointer(), f.d_colorDownsampled, sizeof(uchar4)*color.getNumPixels(), cudaMemcpyDeviceToHost));
			s << depth;
			s << camPos;
			s << normals;
			s << color;
		}
		s.closeStream();
	}
	//! warning: untested!
	void loadFromFile(const std::string& filename) {
		BinaryDataStreamFile s(filename, false);
		s >> m_width;
		s >> m_height;
		s >> m_intrinsics;
		s >> m_intrinsicsInv;
		s >> m_currentFrame;
		unsigned int maxNumImages;
		s >> maxNumImages;
		MLIB_ASSERT(maxNumImages <= m_maxNumImages);

		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		ColorImageR8G8B8A8 color(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			s >> depth;
			s >> camPos;
			s >> normals;
			s >> color;
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_depthDownsampled, depth.getPointer(), sizeof(float)*depth.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_cameraposDownsampled, camPos.getPointer(), sizeof(float4)*camPos.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampled, normals.getPointer(), sizeof(float4)*normals.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_colorDownsampled, color.getPointer(), sizeof(uchar4)*color.getNumPixels(), cudaMemcpyHostToDevice));
		}
		s.closeStream();
	}

	//!debugging only
	void setCachedFrames(const std::vector<CUDACachedFrame> cachedFrames) {
		MLIB_ASSERT(cachedFrames.size() <= m_cache.size());
		for (unsigned int i = 0; i < cachedFrames.size(); i++) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_depthDownsampled, cachedFrames[i].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_colorDownsampled, cachedFrames[i].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_cameraposDownsampled, cachedFrames[i].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampled, cachedFrames[i].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		}
	}

private:

	void alloc() {
		m_cache.resize(m_maxNumImages);
		for (CUDACachedFrame& f : m_cache) {
			f.alloc(m_width, m_height);
		}
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cache, sizeof(CUDACachedFrame)*m_maxNumImages));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_cache, m_cache.data(), sizeof(CUDACachedFrame)*m_maxNumImages, cudaMemcpyHostToDevice));
	}

	void free() {
		for (CUDACachedFrame& f : m_cache) {
			f.free();
		}
		m_cache.clear();
		MLIB_CUDA_SAFE_FREE(d_cache);
	}

	unsigned int m_width;
	unsigned int m_height;
	mat4f		 m_intrinsics;
	mat4f		 m_intrinsicsInv;

	unsigned int m_currentFrame;
	unsigned int m_maxNumImages;

	std::vector < CUDACachedFrame > m_cache;
	CUDACachedFrame*				d_cache;

};