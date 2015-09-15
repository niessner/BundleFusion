#pragma once
#include "RGBDSensor.h"
#include "CUDAImageUtil.h"

#include <cuda_runtime.h>

class CUDAImageManager {
public:
	struct CUDARGBDInputFrame {
		void alloc(unsigned int widthIntegration, unsigned int heightIntegration) {
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthIntegration, sizeof(float)*widthIntegration*heightIntegration));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorIntegration, sizeof(uchar4)*widthIntegration*heightIntegration));
		}
		void free() {
			MLIB_CUDA_SAFE_FREE(d_depthIntegration);
			MLIB_CUDA_SAFE_FREE(d_colorIntegration);
		}

		float*	d_depthIntegration;
		uchar4*	d_colorIntegration;
	};

	CUDAImageManager(unsigned int widthIntegration, unsigned int heightIntegration, unsigned int widthSIFT, unsigned int heightSIFT, RGBDSensor* sensor) {
		m_RGBDSensor = sensor;

		m_widthSIFT = widthSIFT;
		m_heightSIFT = heightSIFT;
		m_widthIntegration = widthIntegration;
		m_heightIntegration = heightIntegration;

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInput, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorInput, sizeof(uchar4)*bufferDimColorInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensitySIFT, sizeof(float)*m_widthSIFT*m_heightSIFT));

		m_currFrame = 0;
	}

	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInput);
		MLIB_CUDA_SAFE_FREE(d_colorInput);
		MLIB_CUDA_SAFE_FREE(d_intensitySIFT);
	}

	void reset() {
		for (CUDARGBDInputFrame& f : m_data) {
			f.free();
		}
		m_data.clear();
	}

	bool process() {

		if (!m_RGBDSensor->processDepth()) return false;	// Order is important!
		if (!m_RGBDSensor->processColor()) return false;

		m_data.push_back(CUDARGBDInputFrame());
		CUDARGBDInputFrame& frame = m_data.back();
		frame.alloc(m_widthIntegration, m_heightIntegration);	//could be done offline for a max number of frames?

		////////////////////////////////////////////////////////////////////////////////////
		// Process Color
		////////////////////////////////////////////////////////////////////////////////////

		const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_colorInput, m_RGBDSensor->getColorRGBX(), sizeof(uchar4)*bufferDimColorInput, cudaMemcpyHostToDevice));
		if ((m_RGBDSensor->getColorWidth() == m_widthIntegration) && (m_RGBDSensor->getColorHeight() == m_heightIntegration)) {
			CUDAImageUtil::copy<uchar4>(frame.d_colorIntegration, d_colorInput, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.d_colorIntegration, d_colorInput);
		}
		else {
			CUDAImageUtil::resampleUCHAR4(frame.d_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
			//CUDAImageUtil::resample<uchar4>(frame.d_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
		}

		////////////////////////////////////////////////////////////////////////////////////
		// Process Depth
		////////////////////////////////////////////////////////////////////////////////////

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthInput, m_RGBDSensor->getDepthFloat(), sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice));
		if ((m_RGBDSensor->getDepthWidth() == m_widthIntegration) && (m_RGBDSensor->getDepthHeight() == m_heightIntegration)) {
			CUDAImageUtil::copy<float>(frame.d_depthIntegration, d_depthInput, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.d_depthIntegration, d_depthInput);
		}
		else {
			CUDAImageUtil::resampleFloat(frame.d_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInput, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
			//CUDAImageUtil::resample<float>(frame.d_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInput, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
		}

		////////////////////////////////////////////////////////////////////////////////////
		// SIFT Intensity Image
		////////////////////////////////////////////////////////////////////////////////////
		CUDAImageUtil::resampleToIntensity(d_intensitySIFT, m_widthSIFT, m_heightSIFT, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());

		m_currFrame++;
		return true;
	}


	//TODO not const because direct assignment in SiftGPU
	float* getIntensityImage() {
		return d_intensitySIFT;
	}

	const float* getDepthInput() const {
		return d_depthInput;
	}

	const uchar4* getColorInput() const {
		return d_colorInput;
	}

	const float* getLastIntegrateDepth() const {
		return m_data.back().d_depthIntegration;
	}

	const uchar4* getLastIntegrateColor() const {
		return m_data.back().d_colorIntegration;
	}

	const uchar4* getIntegrateColor(unsigned int frame) const {
		return m_data[frame].d_colorIntegration;
	}

	// called after process
	unsigned int getCurrFrameNumber() const {
		MLIB_ASSERT(m_currFrame > 0);
		return m_currFrame - 1;
	}

	unsigned int getIntegrationWidth() const {
		return m_widthIntegration;
	}
	unsigned int getIntegrationHeight() const {
		return m_heightIntegration;
	}
	unsigned int getSIFTWidth() const {
		return m_widthSIFT;
	}
	unsigned int getSIFTHeight() const {
		return m_heightSIFT;
	}

private:
	RGBDSensor* m_RGBDSensor;	

	//! resolution for sift key point detection
	unsigned int m_widthSIFT;
	unsigned int m_heightSIFT;

	//! resolution for integration both depth and color data
	unsigned int m_widthIntegration;
	unsigned int m_heightIntegration;

	//! temporary GPU storage for inputting the current frame
	float*	d_depthInput;
	uchar4*	d_colorInput;

	float* d_intensitySIFT;

	//! all image data on the GPU
	std::vector<CUDARGBDInputFrame>	m_data;

	unsigned int m_currFrame;

};