#pragma once

#include "CUDAImageManager.h"
#include "CUDAImageUtil.h"

class CUDACache {
public:
	struct CUDACachedFrame {
		void alloc(unsigned int width, unsigned int height) {
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorDownsampled, sizeof(uchar4) * width * height));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cameraposDownsampled, sizeof(float4) * width * height));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normalsDownsampled, sizeof(float4) * width * height));
		}
		void free() {
			MLIB_CUDA_SAFE_FREE(d_depthDownsampled);
			MLIB_CUDA_SAFE_FREE(d_colorDownsampled);
			MLIB_CUDA_SAFE_FREE(d_cameraposDownsampled);
			MLIB_CUDA_SAFE_FREE(d_normalsDownsampled);
		}

		float* d_depthDownsampled;
		uchar4* d_colorDownsampled;
		float4* d_cameraposDownsampled;
		float4* d_normalsDownsampled;
	};

	CUDACache(unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& intrinsics) 
	{
		m_width = widthDownSampled;
		m_height = heightDownSampled;
		m_maxNumImages = maxNumImages;

		m_intrinsics = intrinsics;
		m_intrinsicsInv = m_intrinsics.getInverse();

		alloc();
		m_currentFrame = 0;

	}
	~CUDACache() { 
		free(); 
	}

	void storeFrame(const float* d_depth, const uchar4* d_color, unsigned int inputWidth, unsigned int inputHeight);

	void reset() {
		m_currentFrame = 0;
	}

	const std::vector<CUDACachedFrame>& getCacheFrames() const { return m_cache; }

	const void copyCacheFrameFrom(CUDACache* other, unsigned int frameFrom) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_depthDownsampled, other->m_cache[frameFrom].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_colorDownsampled, other->m_cache[frameFrom].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_cameraposDownsampled, other->m_cache[frameFrom].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampled, other->m_cache[frameFrom].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

		m_currentFrame++;
	}

private:

	void alloc() {
		m_cache.resize(m_maxNumImages);
		for (CUDACachedFrame& f : m_cache) {
			f.alloc(m_width, m_height);
		}
	}

	void free() {
		for (CUDACachedFrame& f : m_cache) {
			f.free();
		}
		m_cache.clear();
	}

	unsigned int m_width;
	unsigned int m_height;
	mat4f		 m_intrinsics;
	mat4f		 m_intrinsicsInv;

	unsigned int m_currentFrame;
	unsigned int m_maxNumImages;

	std::vector < CUDACachedFrame > m_cache;

};