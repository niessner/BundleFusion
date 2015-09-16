#include "stdafx.h"
#include "CUDACache.h"


CUDACache::CUDACache(unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& intrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	m_intrinsics = intrinsics;
	m_intrinsicsInv = m_intrinsics.getInverse();

	alloc();
	m_currentFrame = 0;
}


void CUDACache::storeFrame(const float* d_depth, const uchar4* d_color, unsigned int inputWidth, unsigned int inputHeight)
{
	CUDACachedFrame& frame = m_cache[m_currentFrame];
	//CUDAImageUtil::resample<float>(frame.d_depthDownsampled, m_width, m_height, d_depth, inputWidth, inputHeight);
	//CUDAImageUtil::resample<uchar4>(frame.d_colorDownsampled, m_width, m_height, d_color, inputWidth, inputHeight);
	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_depth, inputWidth, inputHeight);
	CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputWidth, inputHeight);

	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(frame.d_cameraposDownsampled, frame.d_depthDownsampled, *(float4x4*)&m_intrinsicsInv, m_width, m_height);
	CUDAImageUtil::computeNormals(frame.d_normalsDownsampled, frame.d_cameraposDownsampled, m_width, m_height);

	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	m_currentFrame++;
}