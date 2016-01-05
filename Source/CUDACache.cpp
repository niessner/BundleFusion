#include "stdafx.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"
#include "MatrixConversion.h"

extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const uchar4* d_normals);

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

	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(frame.d_normalsDownsampledUCHAR4, frame.d_normalsDownsampled, m_width, m_height);

	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_inputDepth, inputDepthWidth, inputDepthHeight);

	//CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	//color
	CUDAImageUtil::resampleToIntensity(d_intensityHelper, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, m_filterIntensitySigma, m_width, m_height);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::jointBilateralFilterFloat(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_intensityFilterSigma, 0.01f, m_width, m_height);
	if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::adaptiveBilateralFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_filterIntensitySigma, 0.01f, 1.0f, m_width, m_height);
	else std::swap(frame.d_intensityDownsampled, d_intensityHelper);
	CUDAImageUtil::computeIntensityDerivatives(frame.d_intensityDerivsDownsampled, frame.d_intensityDownsampled, m_width, m_height);

	m_currentFrame++;
}

void CUDACache::fuseDepthFrames(CUDACache* globalCache, const int* d_validImages, const float4x4* d_transforms) const
{
	assert(globalCache->m_currentFrame > 0);
	const unsigned int numFrames = m_currentFrame;
	const unsigned int globalFrameIdx = globalCache->m_currentFrame - 1;

	CUDACachedFrame& globalFrame = globalCache->m_cache[globalFrameIdx];
	if (globalFrameIdx + 1 == globalCache->m_maxNumImages) {
		std::cerr << "CUDACache reached max # images!" << std::endl;
		while (1);
	}
	CUDACachedFrame& tmpFrame = globalCache->m_cache[globalFrameIdx + 1];

	////!!!debugging
	//DepthImage32 depthImage(m_width, m_height);
	//ColorImageR32 intensity(m_width, m_height);
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthImage.getPointer(), globalFrame.d_depthDownsampled, sizeof(float)*depthImage.getNumPixels(), cudaMemcpyDeviceToHost));
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getPointer(), globalFrame.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
	//FreeImageWrapper::saveImage("debug/_orig.png", depthImage);
	//FreeImageWrapper::saveImage("debug/_origIntensity.png", intensity);
	////!!!debugging

	float4 intrinsics = make_float4(m_intrinsics(0, 0), m_intrinsics(1, 1), m_intrinsics(0, 2), m_intrinsics(1, 2));
	fuseCacheFramesCU(d_cache, d_validImages, intrinsics, d_transforms, numFrames, m_width, m_height,
		globalFrame.d_depthDownsampled, tmpFrame.d_intensityDerivsDownsampled, globalFrame.d_normalsDownsampledUCHAR4);

	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(globalFrame.d_cameraposDownsampled, globalFrame.d_depthDownsampled, MatrixConversion::toCUDA(m_intrinsicsInv), m_width, m_height);
	CUDAImageUtil::computeNormals(globalFrame.d_normalsDownsampled, globalFrame.d_cameraposDownsampled, m_width, m_height);
	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(globalFrame.d_normalsDownsampledUCHAR4, globalFrame.d_normalsDownsampled, m_width, m_height);

	////!!!debugging
	//PointCloudf pcOrig;
	//ColorImageR32G32B32A32 cpos(m_width, m_height);
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(cpos.getPointer(), m_cache.front().d_cameraposDownsampled, sizeof(float4)*cpos.getNumPixels(), cudaMemcpyDeviceToHost));
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getPointer(), m_cache.front().d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
	//for (unsigned int i = 0; i < cpos.getNumPixels(); i++) {
	//	const vec4f& p = cpos.getPointer()[i];
	//	if (p.x != -std::numeric_limits<float>::infinity()) {
	//		pcOrig.m_points.push_back(p.getVec3());
	//		float c = intensity.getPointer()[i];
	//		pcOrig.m_colors.push_back(vec4f(c, c, c, 1.0f));
	//	}
	//}
	//PointCloudIOf::saveToFile("debug/_orig.ply", pcOrig);
	//
	////fused
	//DepthImage32 depthFused(m_width, m_height);
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthFused.getPointer(), globalFrame.d_depthDownsampled, sizeof(float)*depthFused.getNumPixels(), cudaMemcpyDeviceToHost));
	//unsigned int newCount = 0; unsigned int diffCount = 0; unsigned int numOrigValid = 0;
	//for (unsigned int y = 0; y < m_height; y++) {
	//	for (unsigned int x = 0; x < m_width; x++) {
	//		float fuseDepth = depthFused(x, y);
	//		float origDepth = depthImage(x, y);
	//		if (origDepth != -std::numeric_limits<float>::infinity()) numOrigValid++;
	//		if (fuseDepth != -std::numeric_limits<float>::infinity() && origDepth == -std::numeric_limits<float>::infinity())
	//			newCount++; //should not happen!
	//		if (fuseDepth != -std::numeric_limits<float>::infinity() && origDepth != -std::numeric_limits<float>::infinity()) {
	//			if (fabs(fuseDepth - origDepth) > 0.001f) diffCount++;
	//		}
	//	}
	//}
	//FreeImageWrapper::saveImage("debug/_fuse.png", depthFused);
	//PointCloudf pcFuse;
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(cpos.getPointer(), globalFrame.d_cameraposDownsampled, sizeof(float4)*cpos.getNumPixels(), cudaMemcpyDeviceToHost));
	//MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getPointer(), globalFrame.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
	//for (unsigned int i = 0; i < cpos.getNumPixels(); i++) {
	//	const vec4f& p = cpos.getPointer()[i];
	//	if (p.x != -std::numeric_limits<float>::infinity()) {
	//		pcFuse.m_points.push_back(p.getVec3());
	//		float c = intensity.getPointer()[i];
	//		pcFuse.m_colors.push_back(vec4f(c, c, c, 1.0f));
	//	}
	//}
	//PointCloudIOf::saveToFile("debug/_fuse.ply", pcFuse);

	//int a = 5;
	////!!!debugging
}
