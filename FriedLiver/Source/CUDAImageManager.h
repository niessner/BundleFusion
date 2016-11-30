#pragma once
#include "RGBDSensor.h"
#include "CUDAImageUtil.h"
#include "CUDAImageCalibrator.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"

#include <cuda_runtime.h>

class CUDAImageManager {
public:

	class ManagedRGBDInputFrame {
	public:
		friend class CUDAImageManager;

		static void globalInit(unsigned int width, unsigned int height, bool isOnGPU)
		{
			globalFree();

			s_width = width;
			s_height = height;
			s_bIsOnGPU = isOnGPU;

			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_depthIntegrationGlobal, sizeof(float)*width*height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&s_colorIntegrationGlobal, sizeof(uchar4)*width*height));
			}
			else {
				s_depthIntegrationGlobal = new float[width*height];
				s_colorIntegrationGlobal = new uchar4[width*height];
			}
		}
		static void globalFree()
		{
			if (!s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(s_depthIntegrationGlobal);
				MLIB_CUDA_SAFE_FREE(s_colorIntegrationGlobal);
			}
			else {
				SAFE_DELETE_ARRAY(s_depthIntegrationGlobal);
				SAFE_DELETE_ARRAY(s_colorIntegrationGlobal);
			}
		}


		void alloc() {
			if (s_bIsOnGPU) {
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_depthIntegration, sizeof(float)*s_width*s_height));
				MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_colorIntegration, sizeof(uchar4)*s_width*s_height));
			}
			else {
				m_depthIntegration = new float[s_width*s_height];
				m_colorIntegration = new uchar4[s_width*s_height];
			}
		}


		void free() {
			if (s_bIsOnGPU) {
				MLIB_CUDA_SAFE_FREE(m_depthIntegration);
				MLIB_CUDA_SAFE_FREE(m_colorIntegration);
			}
			else {
				SAFE_DELETE_ARRAY(m_depthIntegration);
				SAFE_DELETE_ARRAY(m_colorIntegration);
			}
		}


		const float* getDepthFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_depthIntegration;
			}
			else {
				if (this != s_activeDepthGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeDepthGPU = this;
				}
				return s_depthIntegrationGlobal;
			}
		}
		const uchar4* getColorFrameGPU() {	//be aware that only one depth frame is globally valid at a time
			if (s_bIsOnGPU) {
				return m_colorIntegration;
			}
			else {
				if (this != s_activeColorGPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyHostToDevice));
					s_activeColorGPU = this;
				}
				return s_colorIntegrationGlobal;
			}
		}

		const float* getDepthFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeDepthCPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_depthIntegrationGlobal, m_depthIntegration, sizeof(float)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeDepthCPU = this;
				}
				return s_depthIntegrationGlobal;
			}
			else {
				return m_depthIntegration;
			}
		}
		const uchar4* getColorFrameCPU() {
			if (s_bIsOnGPU) {
				if (this != s_activeColorCPU) {
					MLIB_CUDA_SAFE_CALL(cudaMemcpy(s_colorIntegrationGlobal, m_colorIntegration, sizeof(uchar4)*s_width*s_height, cudaMemcpyDeviceToHost));
					s_activeDepthCPU = this;
				}
				return s_colorIntegrationGlobal;
			}
			else {
				return m_colorIntegration;
			}
		}

	private:
		float*	m_depthIntegration;	//either on the GPU or CPU
		uchar4*	m_colorIntegration;	//either on the GPU or CPU

		static bool			s_bIsOnGPU;
		static unsigned int s_width;
		static unsigned int s_height;

		static float*		s_depthIntegrationGlobal;
		static uchar4*		s_colorIntegrationGlobal;
		static ManagedRGBDInputFrame*	s_activeColorGPU;
		static ManagedRGBDInputFrame*	s_activeDepthGPU;

		static float*		s_depthIntegrationGlobalCPU;
		static uchar4*		s_colorIntegrationGlobalCPU;
		static ManagedRGBDInputFrame*	s_activeColorCPU;
		static ManagedRGBDInputFrame*	s_activeDepthCPU;
	};

	CUDAImageManager(unsigned int widthIntegration, unsigned int heightIntegration, unsigned int widthSIFT, unsigned int heightSIFT, RGBDSensor* sensor, bool storeFramesOnGPU = false) {
		m_RGBDSensor = sensor;

		m_widthSIFTdepth = sensor->getDepthWidth();
		m_heightSIFTdepth = sensor->getDepthHeight();
		m_SIFTdepthIntrinsics = sensor->getDepthIntrinsics();
		m_widthIntegration = widthIntegration;
		m_heightIntegration = heightIntegration;

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputRaw, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputFiltered, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorInput, sizeof(uchar4)*bufferDimColorInput));

		m_currFrame = 0;


		const unsigned int rgbdSensorWidthDepth = m_RGBDSensor->getDepthWidth();
		const unsigned int rgbdSensorHeightDepth = m_RGBDSensor->getDepthHeight();

		// adapt intrinsics
		m_depthIntrinsics = m_RGBDSensor->getDepthIntrinsics();
		m_depthIntrinsics._m00 *= (float)m_widthIntegration / (float)rgbdSensorWidthDepth;  //focal 
		m_depthIntrinsics._m11 *= (float)m_heightIntegration/ (float)rgbdSensorHeightDepth; 
		m_depthIntrinsics._m02 *= (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthDepth-1);	//principal point
		m_depthIntrinsics._m12 *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightDepth-1);
		m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();

		const unsigned int rgbdSensorWidthColor = m_RGBDSensor->getColorWidth();
		const unsigned int rgbdSensorHeightColor = m_RGBDSensor->getColorHeight();

		m_colorIntrinsics = m_RGBDSensor->getColorIntrinsics();
		m_colorIntrinsics._m00 *= (float)m_widthIntegration / (float)rgbdSensorWidthColor;  //focal 
		m_colorIntrinsics._m11 *= (float)m_heightIntegration/ (float)rgbdSensorHeightColor; 
		m_colorIntrinsics._m02 *=  (float)(m_widthIntegration-1) / (float)(rgbdSensorWidthColor-1);	//principal point
		m_colorIntrinsics._m12 *= (float)(m_heightIntegration-1) / (float)(rgbdSensorHeightColor-1);
		m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();

		// adapt extrinsics
		m_depthExtrinsics = m_RGBDSensor->getDepthExtrinsics();
		m_depthExtrinsicsInv = m_RGBDSensor->getDepthExtrinsicsInv();

		if (GlobalAppState::get().s_bUseCameraCalibration) {
			m_SIFTdepthIntrinsics = m_RGBDSensor->getColorIntrinsics();
			m_SIFTdepthIntrinsics._m00 *= (float)m_widthSIFTdepth / (float)rgbdSensorWidthColor;  
			m_SIFTdepthIntrinsics._m11 *= (float)m_heightSIFTdepth / (float)rgbdSensorHeightColor; 
			m_SIFTdepthIntrinsics._m02 *= (float)(m_widthSIFTdepth-1) / (float)(m_RGBDSensor->getColorWidth()-1);
			m_SIFTdepthIntrinsics._m12 *= (float)(m_heightSIFTdepth-1) / (float)(m_RGBDSensor->getColorHeight()-1);
		}

		ManagedRGBDInputFrame::globalInit(getIntegrationWidth(), getIntegrationHeight(), storeFramesOnGPU);
		m_bHasBundlingFrameRdy = false;
	}

	HRESULT OnD3D11CreateDevice(ID3D11Device* device) {
		HRESULT hr = S_OK;
		V_RETURN(m_imageCalibrator.OnD3D11CreateDevice(device, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight()));
		return hr;
	}

	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInputRaw);
		MLIB_CUDA_SAFE_FREE(d_depthInputFiltered);
		MLIB_CUDA_SAFE_FREE(d_colorInput);

		//m_imageCalibrator.OnD3D11DestroyDevice();

		ManagedRGBDInputFrame::globalFree();
	}

	void reset() {
		for (auto& f : m_data) {
			f.free();
		}
		m_data.clear();
	}

	bool process();

	void copyToBundling(float* d_depthRaw, float* d_depthFilt, uchar4* d_color) const {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthRaw, d_depthInputRaw, sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthFilt, d_depthInputFiltered, sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_color, d_colorInput, sizeof(uchar4)*m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight(), cudaMemcpyDeviceToDevice));
	}


	//TODO not const because direct assignment in SiftGPU
	//float* getIntensityImageSIFT() {
	//	return d_intensitySIFT;
	//}

	ManagedRGBDInputFrame& getLastIntegrateFrame() {
		return m_data.back();
	}

	ManagedRGBDInputFrame& getIntegrateFrame(unsigned int frame) {
		return m_data[frame];
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



	const mat4f& getDepthIntrinsics() const	{
		return m_depthIntrinsics;
	}

	const mat4f& getDepthIntrinsicsInv() const {
		return m_depthIntrinsicsInv;
	}

	//const mat4f& getColorIntrinsics() const	{
	//	return m_colorIntrinsics;
	//}

	//const mat4f& getColorIntrinsicsInv() const {
	//	return m_colorIntrinsicsInv;
	//}

	const mat4f& getDepthExtrinsics() const	{
		return m_depthExtrinsics;
	}

	const mat4f& getDepthExtrinsicsInv() const {
		return m_depthExtrinsicsInv;
	}

	const unsigned int getSIFTDepthWidth() const {
		return m_widthSIFTdepth;
	}
	const unsigned int getSIFTDepthHeight() const {
		return m_heightSIFTdepth;
	}
	const mat4f& getSIFTDepthIntrinsics() const	{
		return m_SIFTdepthIntrinsics;
	}


	bool hasBundlingFrameRdy() const {
		return m_bHasBundlingFrameRdy;
	}

	//! must be called by depth sensing to signal bundling that a frame is ready
	void setBundlingFrameRdy() {
		m_bHasBundlingFrameRdy = true;
	}

	//! must be called by bundling to signal depth sensing it can read it a new frame
	void confirmRdyBundlingFrame() {
		m_bHasBundlingFrameRdy = false;
	}
private:
	bool m_bHasBundlingFrameRdy;

	RGBDSensor* m_RGBDSensor;
	CUDAImageCalibrator m_imageCalibrator;

	mat4f m_colorIntrinsics;
	mat4f m_colorIntrinsicsInv;
	mat4f m_depthIntrinsics;
	mat4f m_depthIntrinsicsInv;
	mat4f m_depthExtrinsics;
	mat4f m_depthExtrinsicsInv;

	//! resolution for integration both depth and color data
	unsigned int m_widthIntegration;
	unsigned int m_heightIntegration;
	mat4f m_SIFTdepthIntrinsics;

	//! temporary GPU storage for inputting the current frame
	float*	d_depthInputRaw;
	uchar4*	d_colorInput;
	float*	d_depthInputFiltered;

	unsigned int m_widthSIFTdepth;
	unsigned int m_heightSIFTdepth;

	//! all image data on the GPU
	std::vector<ManagedRGBDInputFrame> m_data;

	unsigned int m_currFrame;

	static Timer s_timer;
};