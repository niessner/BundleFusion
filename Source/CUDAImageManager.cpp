
#include "stdafx.h"

#include "CUDAImageManager.h"

bool		CUDAImageManager::ManagedRGBDInputFrame::m_bIsOnGPU = false;
float*		CUDAImageManager::ManagedRGBDInputFrame::d_depthIntegrationGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::d_colorIntegrationGlobal = NULL;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::m_width = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::m_height = 0;