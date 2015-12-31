#include "stdafx.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"


CUDACache::CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& inputIntrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	const float scaleWidth = (float)widthDownSampled / (float)widthDepthInput;
	const float scaleHeight = (float)heightDownSampled / (float)heightDepthInput;
	m_intrinsics = inputIntrinsics;
	m_intrinsics._m00 *= scaleWidth;  m_intrinsics._m02 *= scaleWidth;
	m_intrinsics._m11 *= scaleHeight; m_intrinsics._m12 *= scaleHeight;
	m_intrinsicsInv = m_intrinsics.getInverse();

	d_intensityHelper = NULL;
	d_filterHelper = NULL;
	d_helperCamPos = NULL;
	d_helperNormals = NULL;
	m_filterIntensitySigma = GlobalBundlingState::get().s_colorDownSigma;
	m_filterDepthSigmaD = GlobalBundlingState::get().s_depthDownSigmaD;
	m_filterDepthSigmaR = GlobalBundlingState::get().s_depthDownSigmaR;

	m_inputDepthWidth = widthDepthInput;
	m_inputDepthHeight = heightDepthInput;
	m_inputIntrinsics = inputIntrinsics;
	m_inputIntrinsicsInv = m_inputIntrinsics.getInverse();

	alloc();
	m_currentFrame = 0;
}


void CUDACache::storeFrame(const float* d_depth, unsigned int inputDepthWidth, unsigned int inputDepthHeight,
	const uchar4* d_color, unsigned int inputColorWidth, unsigned int inputColorHeight)
{
	CUDACachedFrame& frame = m_cache[m_currentFrame];
	//depth
	const float* d_inputDepth = d_depth;
	if (m_filterDepthSigmaD > 0.0f) {
		CUDAImageUtil::gaussFilterDepthMap(d_filterHelper, d_depth, m_filterDepthSigmaD, m_filterDepthSigmaR, inputDepthWidth, inputDepthHeight);
		d_inputDepth = d_filterHelper;
	}
	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(d_helperCamPos, d_inputDepth, *(float4x4*)&m_inputIntrinsicsInv, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_cameraposDownsampled, m_width, m_height, d_helperCamPos, inputDepthWidth, inputDepthHeight);

	CUDAImageUtil::computeNormals(d_helperNormals, d_helperCamPos, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_helperNormals, inputDepthWidth, inputDepthHeight);

	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_inputDepth, inputDepthWidth, inputDepthHeight);

	//CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	//color
	CUDAImageUtil::resampleToIntensity(d_intensityHelper, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//!!!debugging
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.d_intensityOrigDown, d_intensityHelper, sizeof(float)*m_width*m_height, cudaMemcpyDeviceToDevice));
	//!!!debugging
	if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, m_filterIntensitySigma, m_width, m_height);
	else std::swap(frame.d_intensityDownsampled, d_intensityHelper);
	CUDAImageUtil::computeIntensityDerivatives(frame.d_intensityDerivsDownsampled, frame.d_intensityDownsampled, m_width, m_height);

	m_currentFrame++;
}