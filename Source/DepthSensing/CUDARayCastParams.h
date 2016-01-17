#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

__align__(16)	//has to be aligned to 16 bytes
struct RayCastParams {
	float4x4 m_viewMatrix;
	float4x4 m_viewMatrixInverse;
	float mx, my, fx, fy; //raycast intrinsics

	unsigned int m_width;
	unsigned int m_height;

	unsigned int m_numOccupiedSDFBlocks;
	unsigned int m_maxNumVertices;
	int m_splatMinimum;

	float m_minDepth;
	float m_maxDepth;
	float m_rayIncrement;
	float m_thresSampleDist;
	float m_thresDist;
	bool  m_useGradients;

	uint dummy0;
};