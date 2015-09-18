
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

};
