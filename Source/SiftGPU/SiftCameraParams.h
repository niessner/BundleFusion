
#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

__align__(16)	//has to be aligned to 16 bytes
struct SiftCameraParams {

	unsigned int m_depthWidth;
	unsigned int m_depthHeight;
	unsigned int m_intensityWidth;
	unsigned int m_intensityHeight;

	float4x4 m_siftIntrinsics;
	float4x4 m_siftIntrinsicsInv;

	float4x4 m_downSampIntrinsics;
	float4x4 m_downSampIntrinsicsInv;
};
