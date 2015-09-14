
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

	CUDAImageManager() {}
	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInput);
		MLIB_CUDA_SAFE_FREE(d_colorInput);
		MLIB_CUDA_SAFE_FREE(d_intensitySIFT);
	}

	void init(unsigned int widthIntegration, unsigned int heightIntegration, unsigned int widthSIFT, unsigned int heightSIFT, RGBDSensor* sensor) {

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
		} else {
			CUDAImageUtil::resample<uchar4>(frame.d_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
		}

		////////////////////////////////////////////////////////////////////////////////////
		// Process Depth
		////////////////////////////////////////////////////////////////////////////////////

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthInput, m_RGBDSensor->getDepthFloat(), sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice));
		if ((m_RGBDSensor->getDepthWidth() == m_widthIntegration) && (m_RGBDSensor->getDepthHeight() == m_heightIntegration)) {
			CUDAImageUtil::copy<float>(frame.d_depthIntegration, d_depthInput, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.d_depthIntegration, d_depthInput);
		} else {
			CUDAImageUtil::resample<float>(frame.d_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInput, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
		}

		////////////////////////////////////////////////////////////////////////////////////
		// SIFT Intensity Image
		////////////////////////////////////////////////////////////////////////////////////
		CUDAImageUtil::resampleToIntensity(d_intensitySIFT, m_widthSIFT, m_heightSIFT, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
	}


	const float* getIntensityImage() const {
		return d_intensitySIFT;
	}

	const float* getDepthInput() const {
		return d_depthInput;
	}

	const uchar4* getColorInput() const {
		return d_colorInput;
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

};