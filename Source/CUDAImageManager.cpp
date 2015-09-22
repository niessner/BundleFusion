
#include "stdafx.h"

#include "CUDAImageManager.h"

bool		CUDAImageManager::ManagedRGBDInputFrame::s_bIsOnGPU = false;
float*		CUDAImageManager::ManagedRGBDInputFrame::s_depthIntegrationGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::s_colorIntegrationGlobal = NULL;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_width = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_height = 0;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorGPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthGPU = NULL;

Timer CUDAImageManager::s_timer;