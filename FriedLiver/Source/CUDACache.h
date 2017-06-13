#pragma once

#include "CUDACacheUtil.h"
#include "CUDAImageUtil.h"

class CUDACache {
public:

	CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& inputIntrinsics);
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

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDownsampled, other->m_cache[frameFrom].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDerivsDownsampled, other->m_cache[frameFrom].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampledUCHAR4, other->m_cache[frameFrom].d_normalsDownsampledUCHAR4, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampled, other->m_cache[frameFrom].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
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

	unsigned int getNumFrames() const { return m_currentFrame; }

	//! warning: untested!
	void saveToFile(const std::string& filename) const {
		BinaryDataStreamFile s(filename, true);
		s << m_width;
		s << m_height;
		s << m_intrinsics;
		s << m_intrinsicsInv;
		s << m_currentFrame;
		s << m_maxNumImages;
		//s << m_filterIntensitySigma;
		//s << m_filterDepthSigmaD;
		//s << m_filterDepthSigmaR;
		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		//ColorImageR8G8B8A8 color(m_width, m_height);
		ColorImageR32 intensity(m_width, m_height);
		BaseImage<vec2f> intensityDerivative(m_width, m_height);
		ColorImageR32 intensityOrig(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getData(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPos.getData(), f.d_cameraposDownsampled, sizeof(float4)*camPos.getNumPixels(), cudaMemcpyDeviceToHost));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(color.getData(), f.d_colorDownsampled, sizeof(uchar4)*color.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), f.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityDerivative.getData(), f.d_intensityDerivsDownsampled, sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyDeviceToHost));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityOrig.getData(), f.d_normalsDownsampledUCHAR4, sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyDeviceToHost));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(normals.getData(), f.d_normalsDownsampled, sizeof(float4)*normals.getNumPixels(), cudaMemcpyDeviceToHost));
#endif
			s << depth;
			s << camPos;
			s << normals;
			//s << color;
			s << intensity;
			s << intensityDerivative;
			s << intensityOrig;
		}
		s.close();
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
		//s >> m_filterIntensitySigma;
		//s >> m_filterDepthSigmaD;
		//s >> m_filterDepthSigmaR;
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
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_depthDownsampled, depth.getData(), sizeof(float)*depth.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_cameraposDownsampled, camPos.getData(), sizeof(float4)*camPos.getNumPixels(), cudaMemcpyHostToDevice));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_colorDownsampled, color.getData(), sizeof(uchar4)*color.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDownsampled, intensity.getData(), sizeof(float)*intensity.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDerivsDownsampled, intensityDerivative.getData(), sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyHostToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampledUCHAR4, intensityOrig.getData(), sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyHostToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampled, normals.getData(), sizeof(float4)*normals.getNumPixels(), cudaMemcpyHostToDevice));
#endif
		}
		s.close();
	}

	void printCacheImages(std::string outDir) const {
		if (m_cache.empty()) return;
		if (!(outDir.back() == '/' || outDir.back() == '\\')) outDir.push_back('/');
		if (!util::directoryExists(outDir)) util::makeDirectory(outDir);

		ColorImageR32 intensity(m_width, m_height); DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 image(m_width, m_height); ColorImageR8G8B8A8 image8(m_width, m_height);
		image.setInvalidValue(vec4f(-std::numeric_limits<float>::infinity()));
		for (unsigned int i = 0; i < m_cache.size(); i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getData(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), f.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(image8.getData(), f.d_normalsDownsampledUCHAR4, sizeof(uchar4)*image8.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(image.getData(), f.d_normalsDownsampled, sizeof(float4)*image.getNumPixels(), cudaMemcpyDeviceToHost));
			for (auto& p : image) {
				if (p.value.x != -std::numeric_limits<float>::infinity()) {
					p.value.w = 1.0f;
					image8(p.x, p.y).w = 255;
				}
			}
			FreeImageWrapper::saveImage(outDir + std::to_string(i) + "_cache-depth.png", ColorImageR32G32B32(depth));
			FreeImageWrapper::saveImage(outDir + std::to_string(i) + "_cache-intensity.png", intensity);
			FreeImageWrapper::saveImage(outDir + std::to_string(i) + "_cache-normal-uchar4.png", image8);
			FreeImageWrapper::saveImage(outDir + std::to_string(i) + "_cache-normal.png", image);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(image.getData(), f.d_cameraposDownsampled, sizeof(float4)*image.getNumPixels(), cudaMemcpyDeviceToHost));
			FreeImageWrapper::saveImage(outDir + std::to_string(i) + "_cache-campos.png", image);
		}
	}

	//!debugging only
	std::vector<CUDACachedFrame>& getCachedFramesDEBUG() { return m_cache; }
	void setCurrentFrame(unsigned int c) { m_currentFrame = c; }
	void setIntrinsics(const mat4f& inputIntrinsics, const mat4f& intrinsics) { 
		m_inputIntrinsics = inputIntrinsics; m_inputIntrinsicsInv = inputIntrinsics.getInverse();
		m_intrinsics = intrinsics; m_intrinsicsInv = intrinsics.getInverse();
	}
	//!debugging only
	void setCachedFrames(const std::vector<CUDACachedFrame>& cachedFrames) {
		MLIB_ASSERT(cachedFrames.size() <= m_cache.size());
		for (unsigned int i = 0; i < cachedFrames.size(); i++) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_depthDownsampled, cachedFrames[i].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			//MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_colorDownsampled, cachedFrames[i].d_colorDownsampled, sizeof(uchar4) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_cameraposDownsampled, cachedFrames[i].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDownsampled, cachedFrames[i].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDerivsDownsampled, cachedFrames[i].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampledUCHAR4, cachedFrames[i].d_normalsDownsampledUCHAR4, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampled, cachedFrames[i].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
		}
	}

	void fuseDepthFrames(CUDACache* globalCache, const int* d_validImages, const float4x4* d_transforms) const;

private:

	void alloc() {
		m_cache.resize(m_maxNumImages);
		for (CUDACachedFrame& f : m_cache) {
			f.alloc(m_width, m_height);
		}
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cache, sizeof(CUDACachedFrame)*m_maxNumImages));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_cache, m_cache.data(), sizeof(CUDACachedFrame)*m_maxNumImages, cudaMemcpyHostToDevice));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityHelper, sizeof(float)*m_width*m_height));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_filterHelper, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_helperCamPos, sizeof(float4)*m_inputDepthWidth*m_inputDepthHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_helperNormals, sizeof(float4)*m_inputDepthWidth*m_inputDepthHeight));
	}

	void free() {
		for (CUDACachedFrame& f : m_cache) {
			f.free();
		}
		m_cache.clear();
		MLIB_CUDA_SAFE_FREE(d_cache);
		MLIB_CUDA_SAFE_FREE(d_intensityHelper);
		MLIB_CUDA_SAFE_FREE(d_filterHelper);
		MLIB_CUDA_SAFE_FREE(d_helperCamPos);
		MLIB_CUDA_SAFE_FREE(d_helperNormals);

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

	//for hi-res compute
	float* d_filterHelper;
	float4* d_helperCamPos, *d_helperNormals; //TODO ANGIE
	unsigned int m_inputDepthWidth;
	unsigned int m_inputDepthHeight;
	mat4f		 m_inputIntrinsics;
	mat4f		 m_inputIntrinsicsInv;

	float* d_intensityHelper;
	float m_filterIntensitySigma;
	float m_filterDepthSigmaD;
	float m_filterDepthSigmaR;
};