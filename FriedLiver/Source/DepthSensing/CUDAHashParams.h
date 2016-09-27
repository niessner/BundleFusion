

#include <cutil_inline.h>
#include <cutil_math.h>
#include <device_functions.h>

#include "cuda_SimpleMatrixUtil.h"

//TODO might have to be split into static and dynamics
__align__(16)	//has to be aligned to 16 bytes
struct HashParams {
	HashParams() {
	}

	float4x4		m_rigidTransform;
	float4x4		m_rigidTransformInverse;

	unsigned int	m_hashNumBuckets;
	unsigned int	m_hashBucketSize;
	unsigned int	m_hashMaxCollisionLinkedListSize;
	unsigned int	m_numSDFBlocks;

	int				m_SDFBlockSize;
	float			m_virtualVoxelSize;
	unsigned int	m_numOccupiedBlocks;	//occupied blocks in the viewing frustum
	
	float			m_maxIntegrationDistance;
	float			m_truncScale;
	float			m_truncation;
	unsigned int	m_integrationWeightSample;
	unsigned int	m_integrationWeightMax;

	float3			m_streamingVoxelExtents;
	int3			m_streamingGridDimensions;
	int3			m_streamingMinGridPos;
	unsigned int	m_streamingInitialChunkListSize;
	uint2			m_dummy;

};