#include "stdafx.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"
#include "MatrixConversion.h"

#ifdef CUDACACHE_UCHAR_NORMALS
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const uchar4* d_normals);
#else
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const float4* d_normals);
#endif

CUDACache::CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& inputIntrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	m_intrinsics = inputIntrinsics;
	m_intrinsics._m00 *= (float)widthDownSampled / (float)widthDepthInput;
	m_intrinsics._m11 *= (float)heightDownSampled / (float)heightDepthInput;
	m_intrinsics._m02 *= (float)(widthDownSampled -1)/ (float)(widthDepthInput-1);
	m_intrinsics._m12 *= (float)(heightDownSampled-1) / (float)(heightDepthInput-1);
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
#if defined(CUDACACHE_FLOAT_NORMALS) && defined(CUDACACHE_UCHAR_NORMALS)
	CUDAImageUtil::computeNormals(d_helperNormals, d_helperCamPos, inputDepthWidth, inputDepthHeight);

	CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_helperNormals, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(frame.d_normalsDownsampledUCHAR4, frame.d_normalsDownsampled, m_width, m_height);
#elif defined(CUDACACHE_UCHAR_NORMALS)
	CUDAImageUtil::computeNormals(d_helperNormals, d_helperCamPos, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(d_helperCamPos, m_width, m_height, d_helperNormals, inputDepthWidth, inputDepthHeight); //just use the memory from helpercampos
	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(frame.d_normalsDownsampledUCHAR4, d_helperCamPos, m_width, m_height);
#elif defined(CUDACACHE_FLOAT_NORMALS)
	CUDAImageUtil::computeNormals(d_helperNormals, d_helperCamPos, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_helperNormals, inputDepthWidth, inputDepthHeight);
	//CUDAImageUtil::convertNormalsFloat4ToUCHAR4(frame.d_normalsDownsampledUCHAR4, frame.d_normalsDownsampled, m_width, m_height);
#endif

	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_inputDepth, inputDepthWidth, inputDepthHeight);

	//CUDAImageUtil::resampleUCHAR4(frame.d_colorDownsampled, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	//CUDAImageUtil::jointBilateralFilterFloatMap(frame.d_colorDownsampled)

	//color
	CUDAImageUtil::resampleToIntensity(d_intensityHelper, m_width, m_height, d_color, inputColorWidth, inputColorHeight);
	if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::gaussFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, m_filterIntensitySigma, m_width, m_height);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::jointBilateralFilterFloat(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_intensityFilterSigma, 0.01f, m_width, m_height);
	//if (m_filterIntensitySigma > 0.0f) CUDAImageUtil::adaptiveBilateralFilterIntensity(frame.d_intensityDownsampled, d_intensityHelper, frame.d_depthDownsampled, m_filterIntensitySigma, 0.01f, 1.0f, m_width, m_height);
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

	float4 intrinsics = make_float4(m_intrinsics(0, 0), m_intrinsics(1, 1), m_intrinsics(0, 2), m_intrinsics(1, 2));
#ifdef CUDACACHE_UCHAR_NORMALS
	fuseCacheFramesCU(d_cache, d_validImages, intrinsics, d_transforms, numFrames, m_width, m_height,
		globalFrame.d_depthDownsampled, tmpFrame.d_intensityDerivsDownsampled, globalFrame.d_normalsDownsampledUCHAR4);
#else
	fuseCacheFramesCU(d_cache, d_validImages, intrinsics, d_transforms, numFrames, m_width, m_height,
		globalFrame.d_depthDownsampled, tmpFrame.d_intensityDerivsDownsampled, globalFrame.d_normalsDownsampled);
#endif
	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(globalFrame.d_cameraposDownsampled, globalFrame.d_depthDownsampled, MatrixConversion::toCUDA(m_intrinsicsInv), m_width, m_height);
#ifdef CUDACACHE_UCHAR_NORMALS
	CUDAImageUtil::computeNormals(d_helperNormals, globalFrame.d_cameraposDownsampled, m_width, m_height);
	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(globalFrame.d_normalsDownsampledUCHAR4, d_helperNormals, m_width, m_height);
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
	CUDAImageUtil::computeNormals(globalFrame.d_normalsDownsampled, globalFrame.d_cameraposDownsampled, m_width, m_height);
	//CUDAImageUtil::convertNormalsFloat4ToUCHAR4(globalFrame.d_normalsDownsampledUCHAR4, globalFrame.d_normalsDownsampled, m_width, m_height);
#endif
}
