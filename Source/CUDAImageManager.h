#pragma once
#include "RGBDSensor.h"
#include "CUDAImageUtil.h"
#include "GlobalBundlingState.h"
#include "TimingLog.h"

#include <cuda_runtime.h>

class CUDAImageManager {
public:
	//struct CUDARGBDInputFrame {
	//public:


	//	void alloc(unsigned int widthIntegration, unsigned int heightIntegration) {
	//		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthIntegration, sizeof(float)*widthIntegration*heightIntegration));
	//		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorIntegration, sizeof(uchar4)*widthIntegration*heightIntegration));
	//	}
	//	void free() {
	//		MLIB_CUDA_SAFE_FREE(d_depthIntegration);
	//		MLIB_CUDA_SAFE_FREE(d_colorIntegration);
	//	}

	//	float*	d_depthIntegration;
	//	uchar4*	d_colorIntegration;

	//};

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

		//m_widthSIFT = widthSIFT;
		//m_heightSIFT = heightSIFT;
		m_widthIntegration = widthIntegration;
		m_heightIntegration = heightIntegration;

		const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
		const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputRaw, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthInputFiltered, sizeof(float)*bufferDimDepthInput));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorInput, sizeof(uchar4)*bufferDimColorInput));

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

		//const float scaleWidthSIFT = (float)m_widthSIFT / (float)m_RGBDSensor->getColorWidth();
		//const float scaleHeightSIFT = (float)m_heightSIFT / (float)m_RGBDSensor->getColorHeight();
		//m_SIFTintrinsics = m_RGBDSensor->getColorIntrinsics();
		//m_SIFTintrinsics._m00 *= scaleWidthSIFT;  m_SIFTintrinsics._m02 *= scaleWidthSIFT;
		//m_SIFTintrinsics._m11 *= scaleHeightSIFT; m_SIFTintrinsics._m12 *= scaleHeightSIFT;
		//m_SIFTintrinsicsInv = m_RGBDSensor->getColorIntrinsicsInv();
		//m_SIFTintrinsicsInv._m00 /= scaleWidthSIFT; m_SIFTintrinsicsInv._m11 /= scaleHeightSIFT;


		ManagedRGBDInputFrame::globalInit(getIntegrationWidth(), getIntegrationHeight(), storeFramesOnGPU);
		m_bHasBundlingFrameRdy = false;
	}

	~CUDAImageManager() {
		reset();

		MLIB_CUDA_SAFE_FREE(d_depthInputRaw);
		MLIB_CUDA_SAFE_FREE(d_depthInputFiltered);
		MLIB_CUDA_SAFE_FREE(d_colorInput);

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
	//unsigned int getSIFTWidth() const {
	//	return m_widthSIFT;
	//}
	//unsigned int getSIFTHeight() const {
	//	return m_heightSIFT;
	//}



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

	//const mat4f& getSIFTIntrinsics() const	{
	//	return m_SIFTintrinsics;
	//}

	//const mat4f& getSIFTIntrinsicsInv() const {
	//	return m_SIFTintrinsicsInv;
	//}


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

	mat4f m_intrinsics;
	mat4f m_intrinsicsInv;
	mat4f m_extrinsics;
	mat4f m_extrinsicsInv;

	//! resolution for integration both depth and color data
	unsigned int m_widthIntegration;
	unsigned int m_heightIntegration;

	//! temporary GPU storage for inputting the current frame
	float*	d_depthInputRaw;
	uchar4*	d_colorInput;
	float*	d_depthInputFiltered;

	//! all image data on the GPU
	//std::vector<CUDARGBDInputFrame>	m_data;
	std::vector<ManagedRGBDInputFrame> m_data;

	unsigned int m_currFrame;

	static Timer s_timer;
};