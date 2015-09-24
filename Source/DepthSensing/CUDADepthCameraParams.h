#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

__align__(16)	//has to be aligned to 16 bytes
struct DepthCameraParams {
	float fx;
	float fy;
	float mx;
	float my;

	unsigned int m_imageWidth;
	unsigned int m_imageHeight;

	float m_sensorDepthWorldMin;	//render depth min
	float m_sensorDepthWorldMax;	//render depth max
};
