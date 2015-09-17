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

	class ManagedRGBDInputFrame {
	public:
		static void init(unsigned int width, unsigned int height, bool isOnGPU) 
		{
			m_width = width;
			m_height = height;
			m_bIsOnGPU = isOnGPU;

			if (!m_bIsOnGPU) {
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthIntegrationGlobal, sizeof(float)*width*height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorIntegrationGlobal, sizeof(uchar4)*width*height));
			}
			else {
				d_depthIntegrationGlobal = NULL;
				d_colorIntegrationGlobal = NULL;
			}
		}
		static void free() {
			if (!m_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(d_depthIntegrationGlobal);
				MLIB_CUDA_SAFE_FREE(d_colorIntegrationGlobal);
			}
		}

		ManagedRGBDInputFrame(float* depthIntegration, uchar4* colorIntegration) {
			m_depthIntegration = depthIntegration;
			m_colorIntegration = colorIntegration;
		}
		~ManagedRGBDInputFrame() {
		}

		float* getDepthFrame() {
			if (m_bIsOnGPU) {
				return m_depthIntegration;
			}
			else {
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*m_width*m_height,cudaMemcpyHostToDevice));
				return d_depthIntegrationGlobal;
			}
		}
		uchar4* getColorFrame() {
			if (m_bIsOnGPU) {
				return m_colorIntegration;
			}
			else {
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*m_width*m_height, cudaMemcpyHostToDevice));
				return d_colorIntegrationGlobal;
			}
		}
	private:
		float*	m_depthIntegration;	//either on the GPU or CPU
		uchar4*	m_colorIntegration;	//either on the GPU or CPU

		static bool			m_bIsOnGPU;
		static float*		d_depthIntegrationGlobal;
		static uchar4*		d_colorIntegrationGlobal;
		static unsigned int m_width;
		static unsigned int m_height;
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


		const float scaleWidthDepth = (float)m_widthIntegration / (float)m_RGBDSensor->getDepthWidth();
		const float scaleHeightDepth = (float)m_heightIntegration / (float)m_RGBDSensor->getDepthHeight();

		// adapt intrinsics
		m_intrinsics = m_RGBDSensor->getDepthIntrinsics();
		m_intrinsics._m00 *= scaleWidthDepth;  m_intrinsics._m02 *= scaleWidthDepth;
		m_intrinsics._m11 *= scaleHeightDepth; m_intrinsics._m12 *= scaleHeightDepth;

		m_intrinsicsInv = m_RGBDSensor->getDepthIntrinsicsInv();
		m_intrinsicsInv._m00 /= scaleWidthDepth; m_intrinsicsInv._m11 /= scaleHeightDepth;

		// adapt extrinsics
		m_extrinsics = m_RGBDSensor->getDepthExtrinsics();
		m_extrinsicsInv = m_RGBDSensor->getDepthExtrinsicsInv();

		const float scaleWidthSIFT = (float)m_widthSIFT / (float)m_RGBDSensor->getColorWidth();
		const float scaleHeightSIFT = (float)m_heightSIFT / (float)m_RGBDSensor->getColorHeight();
		m_SIFTintrinsics = m_RGBDSensor->getColorIntrinsics();
		m_SIFTintrinsics._m00 *= scaleWidthSIFT;  m_SIFTintrinsics._m02 *= scaleWidthSIFT;
		m_SIFTintrinsics._m11 *= scaleHeightSIFT; m_SIFTintrinsics._m12 *= scaleHeightSIFT;
		m_SIFTintrinsicsInv = m_RGBDSensor->getColorIntrinsicsInv();
		m_SIFTintrinsicsInv._m00 /= scaleWidthSIFT; m_SIFTintrinsicsInv._m11 /= scaleHeightSIFT;
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
	float* getIntensityImageSIFT() {
		return d_intensitySIFT;
	}

	//const float* getDepthInput() const {
	//	return d_depthInput;
	//}

	//const uchar4* getColorInput() const {
	//	return d_colorInput;
	//}

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



	const mat4f& getIntrinsics() const	{
		return m_intrinsics;
	}

	const mat4f& getIntrinsicsInv() const {
		return m_intrinsicsInv;
	}

	const mat4f& getExtrinsics() const	{
		return m_extrinsics;
	}

	const mat4f& getExtrinsicsInv() const {
		return m_extrinsicsInv;
	}

	const mat4f& getSIFTIntrinsics() const	{
		return m_SIFTintrinsics;
	}

	const mat4f& getSIFTIntrinsicsInv() const {
		return m_SIFTintrinsicsInv;
	}

private:
	RGBDSensor* m_RGBDSensor;	

	mat4f m_intrinsics;
	mat4f m_intrinsicsInv;
	mat4f m_extrinsics;
	mat4f m_extrinsicsInv;

	//! resolution for sift key point detection
	unsigned int m_widthSIFT;
	unsigned int m_heightSIFT;
	mat4f m_SIFTintrinsics;
	mat4f m_SIFTintrinsicsInv;

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