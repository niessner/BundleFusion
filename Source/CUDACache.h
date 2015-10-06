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