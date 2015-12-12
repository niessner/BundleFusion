
#include "stdafx.h"

#include "CUDAImageManager.h"
#include "SiftVisualization.h"

bool		CUDAImageManager::ManagedRGBDInputFrame::s_bIsOnGPU = false;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_width = 0;
unsigned int CUDAImageManager::ManagedRGBDInputFrame::s_height = 0;

float*		CUDAImageManager::ManagedRGBDInputFrame::s_depthIntegrationGlobal = NULL;
uchar4*		CUDAImageManager::ManagedRGBDInputFrame::s_colorIntegrationGlobal = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorGPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthGPU = NULL;

CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeColorCPU = NULL;
CUDAImageManager::ManagedRGBDInputFrame* CUDAImageManager::ManagedRGBDInputFrame::s_activeDepthCPU = NULL;

Timer CUDAImageManager::s_timer;

bool CUDAImageManager::process()
{
	if (!m_RGBDSensor->processDepth()) return false;	// Order is important!
	if (!m_RGBDSensor->processColor()) return false;

	if (GlobalBundlingState::get().s_enableGlobalTimings) { TimingLog::addLocalFrameTiming(); cudaDeviceSynchronize(); s_timer.start(); }

	m_data.push_back(ManagedRGBDInputFrame());
	ManagedRGBDInputFrame& frame = m_data.back();
	frame.alloc();

	////////////////////////////////////////////////////////////////////////////////////
	// Process Color
	////////////////////////////////////////////////////////////////////////////////////

	const unsigned int bufferDimColorInput = m_RGBDSensor->getColorWidth()*m_RGBDSensor->getColorHeight();
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_colorInput, m_RGBDSensor->getColorRGBX(), sizeof(uchar4)*bufferDimColorInput, cudaMemcpyHostToDevice));

	if ((m_RGBDSensor->getColorWidth() == m_widthIntegration) && (m_RGBDSensor->getColorHeight() == m_heightIntegration)) {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::copy<uchar4>(frame.m_colorIntegration, d_colorInput, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.m_colorIntegration, d_colorInput);
		}
		else {
			memcpy(frame.m_colorIntegration, m_RGBDSensor->getColorRGBX(), sizeof(uchar4)*bufferDimColorInput);
		}
	}
	else {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::resampleUCHAR4(frame.m_colorIntegration, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
		}
		else {
			CUDAImageUtil::resampleUCHAR4(frame.s_colorIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.m_colorIntegration, frame.s_colorIntegrationGlobal, sizeof(uchar4)*m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost));
			frame.s_activeColorGPU = &frame;
		}
	}

	////////////////////////////////////////////////////////////////////////////////////
	// Process Depth
	////////////////////////////////////////////////////////////////////////////////////

	const unsigned int bufferDimDepthInput = m_RGBDSensor->getDepthWidth()*m_RGBDSensor->getDepthHeight();
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthInput, m_RGBDSensor->getDepthFloat(), sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice));
	if (GlobalBundlingState::get().s_erodeSIFTdepth) {
		unsigned int numIter = 2;
		numIter = 2 * ((numIter + 1) / 2);
		for (unsigned int i = 0; i < numIter; i++) {
			if (i % 2 == 0) {
				CUDAImageUtil::erodeDepthMap(d_depthErodeHelper, d_depthInput, 3,
					m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f);
			}
			else {
				CUDAImageUtil::erodeDepthMap(d_depthInput, d_depthErodeHelper, 3,
					m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f);
			}
		}
	}
	if (GlobalBundlingState::get().s_depthFilter) { //smooth
		CUDAImageUtil::gaussFilterFloatMap(d_depthErodeHelper, d_depthInput, GlobalBundlingState::get().s_depthSigmaD, GlobalBundlingState::get().s_depthSigmaR,
			m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
		std::swap(d_depthInput, d_depthErodeHelper);
	}

	if ((m_RGBDSensor->getDepthWidth() == m_widthIntegration) && (m_RGBDSensor->getDepthHeight() == m_heightIntegration)) {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::copy<float>(frame.m_depthIntegration, d_depthInput, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.m_depthIntegration, d_depthInput);
		}
		else {
			if (GlobalBundlingState::get().s_erodeSIFTdepth) {
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.m_depthIntegration, d_depthInput, sizeof(float)*bufferDimDepthInput, cudaMemcpyDeviceToHost));
			}
			else {
				memcpy(frame.m_depthIntegration, m_RGBDSensor->getDepthFloat(), sizeof(float)*bufferDimDepthInput);
			}
		}
	}
	else {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::resampleFloat(frame.m_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInput, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
		}
		else {
			CUDAImageUtil::resampleFloat(frame.s_depthIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_depthInput, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.m_depthIntegration, frame.s_depthIntegrationGlobal, sizeof(float)*m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost));
			frame.s_activeDepthGPU = &frame;
		}
	}

	//!!!DEBUGGING
	//{
	//	DepthImage32 dImage(m_widthIntegration, m_heightIntegration);
	//	ColorImageR8G8B8A8 cImage(m_widthIntegration, m_heightIntegration);
	//	if (ManagedRGBDInputFrame::s_bIsOnGPU) {
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(dImage.getPointer(), frame.m_depthIntegration, sizeof(float)*m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost));
	//		MLIB_CUDA_SAFE_CALL(cudaMemcpy(cImage.getPointer(), frame.m_colorIntegration, sizeof(vec4uc)*m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost));
	//	}
	//	else {
	//		memcpy(dImage.getPointer(), frame.m_depthIntegration, sizeof(float)*m_widthIntegration*m_heightIntegration);
	//		memcpy(cImage.getPointer(), frame.m_colorIntegration, sizeof(vec4uc)*m_widthIntegration*m_heightIntegration);
	//	}
	//	FreeImageWrapper::saveImage("debug/test.png", ColorImageR32G32B32(dImage));
	//	PointCloudf pc;
	//	SiftVisualization::computePointCloud(pc, dImage.getPointer(), m_widthIntegration, m_heightIntegration,
	//		cImage.getPointer(), m_widthIntegration, m_heightIntegration, m_intrinsicsInv, mat4f::identity());
	//	PointCloudIOf::saveToFile("debug/test.ply", pc);
	//	std::cout << "waiting..." << std::endl;
	//	getchar();
	//}
	//!!!DEBUGGING

	//////////////////////////////////////////////////////////////////////////////////////
	//// SIFT Intensity Image
	//////////////////////////////////////////////////////////////////////////////////////
	//CUDAImageUtil::resampleToIntensity(d_intensitySIFT, m_widthSIFT, m_heightSIFT, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());

	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(true).timeSensorProcess = s_timer.getElapsedTimeMS(); }

	m_currFrame++;
	return true;
}
