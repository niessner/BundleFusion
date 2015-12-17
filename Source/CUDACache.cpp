#include "stdafx.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"


CUDACache::CUDACache(unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& intrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	m_intrinsics = intrinsics;
	m_intrinsicsInv = m_intrinsics.getInverse();

	d_intensityHelper = NULL;
	m_filterIntensitySigma = GlobalBundlingState::get().s_colorSigma;

	alloc();
	m_currentFrame = 0;
}


void CUDACache::storeFrame(const float* d_depth, unsigned int inputDepthWidth, unsigned int inputDepthHeight,
	const uchar4* d_color, unsigned int inputColorWidth, unsigned int inputColorHeight)
{
	CUDACachedFrame& frame = m_cache[m_currentFrame];
	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_depth, inputDepthWidth, inputDepthHeight);
	//CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputColorWidth, inputColorHeight);

	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(frame.d_cameraposDownsampled, frame.d_depthDownsampled, *(float4x4*)&m_intrinsicsInv, m_width, m_height);
	CUDAImageUtil::computeNormals(frame.d_normalsDownsampled, frame.d_cameraposDownsampled, m_width, m_height);

	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	CUDAImageUtil::resampleToIntensity(d_intensityHelper, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//!!!debugging
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.d_intensityOrigDown, d_intensityHelper, sizeof(float)*m_width*m_height, cudaMemcpyDeviceToDevice));
	//!!!debugging
	if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, m_filterIntensitySigma, m_width, m_height);
	else std::swap(frame.d_intensityDownsampled, d_intensityHelper);
	CUDAImageUtil::computeIntensityDerivatives(frame.d_intensityDerivsDownsampled, frame.d_intensityDownsampled, m_width, m_height);

	m_currentFrame++;
}