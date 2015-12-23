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
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_colorDownsampled, other->m_cache[frameFrom].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_cameraposDownsampled, other->m_cache[frameFrom].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampled, other->m_cache[frameFrom].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDownsampled, other->m_cache[frameFrom].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDerivsDownsampled, other->m_cache[frameFrom].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));

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
		//ColorImageR8G8B8A8 color(m_width, m_height);
		ColorImageR32 intensity(m_width, m_height);
		BaseImage<vec2f> intensityDerivative(m_width, m_height);
		ColorImageR32 intensityOrig(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getPointer(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPos.getPointer(), f.d_cameraposDownsampled, sizeof(float4)*camPos.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(normals.getPointer(), f.d_normalsDownsampled, sizeof(float4)*normals.getNumPixels(), cudaMemcpyDeviceToHost));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(color.getPointer(), f.d_colorDownsampled, sizeof(uchar4)*color.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getPointer(), f.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityDerivative.getPointer(), f.d_intensityDerivsDownsampled, sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityOrig.getPointer(), f.d_intensityOrigDown, sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyDeviceToHost));
			s << depth;
			s << camPos;
			s << normals;
			//s << color;
			s << intensity;
			s << intensityDerivative;
			s << intensityOrig;
		}
		s.closeStream();
	}
	//! warning: untested!
	void loadFromFile(const std::string& filename) {
		unsigned int oldMaxNumImages = m_maxNumImages;
		unsigned int oldWidth = m_width;
		unsigned int oldHeight = m_height;
		BinaryDataStreamFile s(filename, false);
		s >> m_width;
		s >> m_height;
		s >> m_intrinsics;
		s >> m_intrinsicsInv;
		s >> m_currentFrame;
		s >> m_maxNumImages;
		if (m_maxNumImages > oldMaxNumImages || m_width > oldWidth || m_height > oldHeight) {
			free();
			alloc();
		}

		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		//ColorImageR8G8B8A8 color(m_width, m_height);
		ColorImageR32 intensity(m_width, m_height);
		BaseImage<vec2f> intensityDerivative(m_width, m_height);
		ColorImageR32 intensityOrig(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			s >> depth;
			s >> camPos;
			s >> normals;
			//s >> color;
			s >> intensity;
			s >> intensityDerivative;
			s >> intensityOrig;
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_depthDownsampled, depth.getPointer(), sizeof(float)*depth.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_cameraposDownsampled, camPos.getPointer(), sizeof(float4)*camPos.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampled, normals.getPointer(), sizeof(float4)*normals.getNumPixels(), cudaMemcpyHostToDevice));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_colorDownsampled, color.getPointer(), sizeof(uchar4)*color.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDownsampled, intensity.getPointer(), sizeof(float)*intensity.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDerivsDownsampled, intensityDerivative.getPointer(), sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityOrigDown, intensityOrig.getPointer(), sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyHostToDevice));
		}
		s.closeStream();
	}

	//!debugging only
	void setCachedFrames(const std::vector<CUDACachedFrame> cachedFrames) {
		MLIB_ASSERT(cachedFrames.size() <= m_cache.size());
		for (unsigned int i = 0; i < cachedFrames.size(); i++) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_depthDownsampled, cachedFrames[i].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_colorDownsampled, cachedFrames[i].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_cameraposDownsampled, cachedFrames[i].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampled, cachedFrames[i].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDownsampled, cachedFrames[i].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDerivsDownsampled, cachedFrames[i].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityOrigDown, cachedFrames[i].d_intensityOrigDown, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		}
	}
	//!debugging only
	void reFilterCachedIntensityFrames(const float sigma) {
		if (sigma == m_filterIntensitySigma) return;
		std::cout << "re-filtering intensity (sigma = " << sigma << ")" << std::endl;
		m_filterIntensitySigma = sigma;
		for (unsigned int i = 0; i < m_cache.size(); i++) {
			if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(m_cache[i].d_intensityDownsampled, m_cache[i].d_intensityOrigDown, m_filterIntensitySigma, m_width, m_height);
			else CUDAImageUtil::copy(m_cache[i].d_intensityDownsampled, m_cache[i].d_intensityOrigDown, m_width, m_height);
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

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityHelper, sizeof(float)*m_width*m_height));
	}

	void free() {
		for (CUDACachedFrame& f : m_cache) {
			f.free();
		}
		m_cache.clear();
		MLIB_CUDA_SAFE_FREE(d_cache);
		MLIB_CUDA_SAFE_FREE(d_intensityHelper);

		m_currentFrame = 0;
	}

	unsigned int m_width;
	unsigned int m_height;
	mat4f		 m_intrinsics;
	mat4f		 m_intrinsicsInv;

	unsigned int m_currentFrame;
	unsigned int m_maxNumImages;

	std::vector < CUDACachedFrame > m_cache;
	CUDACachedFrame*				d_cache;

	float* d_intensityHelper; //for intensity filtering
	float m_filterIntensitySigma;
};