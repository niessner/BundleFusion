#pragma once

#include "RGBDSensor.h"
#include "GlobalAppState.h"
#include "GlobalBundlingState.h"

struct BundlerInputData {
	//meta-info
	unsigned int			m_inputDepthWidth, m_inputDepthHeight;
	unsigned int			m_inputColorWidth, m_inputColorHeight;
	unsigned int			m_widthSIFT, m_heightSIFT;
	mat4f					m_SIFTIntrinsics;
	mat4f					m_SIFTIntrinsicsInv;
	//data
	float*					d_inputDepthFilt, *d_inputDepthRaw;
	uchar4*					d_inputColor;
	float*					d_intensitySIFT;
	float*					d_intensityFilterHelper; //TODO check if used
	//filtering  //TODO option for depth filter
	bool m_bFilterIntensity;
	float m_intensitySigmaD, m_intensitySigmaR;

	BundlerInputData() {
		m_inputDepthWidth = 0;	m_inputDepthHeight = 0;
		m_inputColorWidth = 0;	m_inputColorHeight = 0;
		m_widthSIFT = 0;		m_heightSIFT = 0;
		d_inputDepthRaw = NULL; d_inputDepthFilt = NULL;
		d_inputColor = NULL;
		d_intensitySIFT = NULL;
		d_intensityFilterHelper = NULL;
		m_bFilterIntensity = false;
	}
	void alloc(const RGBDSensor* sensor) {
		m_inputDepthWidth = sensor->getDepthWidth();
		m_inputDepthHeight = sensor->getDepthHeight();
		m_inputColorWidth = sensor->getColorWidth();
		m_inputColorHeight = sensor->getColorHeight();
		m_widthSIFT = GlobalBundlingState::get().s_widthSIFT;
		m_heightSIFT = GlobalBundlingState::get().s_heightSIFT;
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputDepthFilt, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputDepthRaw, sizeof(float)*m_inputDepthWidth*m_inputDepthHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityFilterHelper, sizeof(float)*m_widthSIFT*m_heightSIFT));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_inputColor, sizeof(uchar4)*m_inputColorWidth*m_inputColorHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensitySIFT, sizeof(float)*m_widthSIFT*m_heightSIFT));

		m_SIFTIntrinsics = sensor->getColorIntrinsics();
		m_SIFTIntrinsics._m00 *= (float)m_widthSIFT / (float)m_inputColorWidth;
		m_SIFTIntrinsics._m11 *= (float)m_heightSIFT / (float)m_inputColorHeight;
		m_SIFTIntrinsics._m02 *= (float)(m_widthSIFT - 1) / (float)(m_inputColorWidth - 1);
		m_SIFTIntrinsics._m12 *= (float)(m_heightSIFT - 1) / (float)(m_inputColorHeight - 1);
		m_SIFTIntrinsicsInv = m_SIFTIntrinsics.getInverse();

		m_bFilterIntensity = GlobalAppState::get().s_colorFilter;
		m_intensitySigmaR = GlobalAppState::get().s_colorSigmaR;
		m_intensitySigmaD = GlobalAppState::get().s_colorSigmaD;
	}
	~BundlerInputData() {
		MLIB_CUDA_SAFE_CALL(cudaFree(d_inputDepthFilt));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_inputDepthRaw));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_inputColor));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_intensitySIFT));
		MLIB_CUDA_SAFE_CALL(cudaFree(d_intensityFilterHelper));
	}
};


//TODO CHECK WHAT IS NEEDED
struct BundlerState {
	enum PROCESS_STATE {
		DO_NOTHING,
		PROCESS,
		INVALIDATE
	};

	//current frames
	int						m_lastFrameProcessed;
	bool					m_bLastFrameValid;

	int						m_localToSolve;		// index of local submap to solve (-1) if none
	int						m_lastLocalSolved; // to check if can fuse to global

	unsigned int			m_numFramesPastEnd; // past end of sequence

	unsigned int			m_numCompleteTransforms;
	unsigned int			m_lastValidCompleteTransform;
	bool					m_bGlobalTrackingLost;

	PROCESS_STATE			m_processState;
	bool					m_bUseSolve;

	unsigned int			m_totalNumOptLocalFrames;

	BundlerState() {
		m_localToSolve = -1;
		m_lastLocalSolved = -1;

		m_lastFrameProcessed = -1;
		m_bLastFrameValid = false;
		m_numFramesPastEnd = 0;

		m_numCompleteTransforms = 0;
		m_lastValidCompleteTransform = 0;
		m_bGlobalTrackingLost = false;
		m_processState = DO_NOTHING;

		m_bUseSolve = true;
		m_totalNumOptLocalFrames = 0;
	}
};