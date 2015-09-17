
#include "mLibCuda.h"

extern "C" void convertMatricesToPoses(float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans)
{

}


extern "C" void convertPosesToMatrices(float3* d_rot, float3* d_trans, unsigned int numImages, float4x4* d_transforms)
{

}