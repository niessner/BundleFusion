
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
	if (m_currFrame + 1 > GlobalBundlingState::get().s_maxNumImages * GlobalBundlingState::get().s_submapSize) {
		std::cout << "WARNING: reached max #images, truncating sequence" << std::endl;
		return false;
	}

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
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depthInputRaw, m_RGBDSensor->getDepthFloat(), sizeof(float)*m_RGBDSensor->getDepthWidth()* m_RGBDSensor->getDepthHeight(), cudaMemcpyHostToDevice));

	////////////////////////////////////////////////////////////////////////////////////
	// Render to Color Space
	////////////////////////////////////////////////////////////////////////////////////
	if (GlobalAppState::get().s_bUseCameraCalibration)
	{
		//DepthImage32 depthImage(m_widthSIFTdepth, m_heightSIFTdepth);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthImage.getData(), d_depthInputRaw, sizeof(float)*depthImage.getNumPixels(), cudaMemcpyDeviceToHost));
		//FreeImageWrapper::saveImage("debug/_depth-orig.png", ColorImageR32G32B32(depthImage));
		
		m_imageCalibrator.process(DXUTGetD3D11DeviceContext(), d_depthInputRaw, m_SIFTdepthIntrinsics, m_RGBDSensor->getDepthIntrinsicsInv(), m_RGBDSensor->getDepthExtrinsics());
		
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(depthImage.getData(), d_depthInputRaw, sizeof(float)*depthImage.getNumPixels(), cudaMemcpyDeviceToHost));
		//FreeImageWrapper::saveImage("debug/_depth-new.png", ColorImageR32G32B32(depthImage));
		//ColorImageR8G8B8A8 colorImage(m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());
		//memcpy(colorImage.getData(), m_RGBDSensor->getColorRGBX(), sizeof(vec4uc)*colorImage.getNumPixels());
		//for (unsigned int i = 0; i < colorImage.getNumPixels(); i++) colorImage.getData()[i].w = 255;
		//colorImage.resize(m_widthSIFTdepth, m_heightSIFTdepth);
		//FreeImageWrapper::saveImage("debug/_color.png", colorImage);
		//int a = 5;
	}
	////////////////////////////////////////////////////////////////////////////////////

	if (GlobalBundlingState::get().s_erodeSIFTdepth) {
		unsigned int numIter = 2;
		numIter = 2 * ((numIter + 1) / 2);
		for (unsigned int i = 0; i < numIter; i++) {
			if (i % 2 == 0) {
				CUDAImageUtil::erodeDepthMap(d_depthInputFiltered, d_depthInputRaw, 3,
					m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f);
			}
			else {
				CUDAImageUtil::erodeDepthMap(d_depthInputRaw, d_depthInputFiltered, 3,
					m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight(), 0.05f, 0.3f);
			}
		}
	}
	if (GlobalBundlingState::get().s_depthFilter) { //smooth
		CUDAImageUtil::gaussFilterDepthMap(d_depthInputFiltered, d_depthInputRaw, GlobalBundlingState::get().s_depthSigmaD, GlobalBundlingState::get().s_depthSigmaR,
			m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
	}
	else {
		CUDAImageUtil::copy<float>(d_depthInputFiltered, d_depthInputRaw, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
	}

	//////////////////////////////////////////////////////////////////////////////////////
	//// Render to Color Space
	//////////////////////////////////////////////////////////////////////////////////////
	//if (GlobalAppState::get().s_bUseCameraCalibration)
	//{
	//	m_imageCalibrator.process(DXUTGetD3D11DeviceContext(), d_depthInputFiltered, m_SIFTdepthIntrinsics, m_RGBDSensor->getDepthIntrinsicsInv(), m_RGBDSensor->getDepthExtrinsicsInv());
	//}
	//////////////////////////////////////////////////////////////////////////////////////

	if ((m_RGBDSensor->getDepthWidth() == m_widthIntegration) && (m_RGBDSensor->getDepthHeight() == m_heightIntegration)) {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::copy<float>(frame.m_depthIntegration, d_depthInputFiltered, m_widthIntegration, m_heightIntegration);
			//std::swap(frame.m_depthIntegration, d_depthInput);
		}
		else {
			if (GlobalBundlingState::get().s_erodeSIFTdepth) {
				MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.m_depthIntegration, d_depthInputFiltered, sizeof(float)*bufferDimDepthInput, cudaMemcpyDeviceToHost));
			}
			else {
				memcpy(frame.m_depthIntegration, m_RGBDSensor->getDepthFloat(), sizeof(float)*bufferDimDepthInput);
			}
		}
	}
	else {
		if (ManagedRGBDInputFrame::s_bIsOnGPU) {
			CUDAImageUtil::resampleFloat(frame.m_depthIntegration, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
		}
		else {
			CUDAImageUtil::resampleFloat(frame.s_depthIntegrationGlobal, m_widthIntegration, m_heightIntegration, d_depthInputFiltered, m_RGBDSensor->getDepthWidth(), m_RGBDSensor->getDepthHeight());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(frame.m_depthIntegration, frame.s_depthIntegrationGlobal, sizeof(float)*m_widthIntegration*m_heightIntegration, cudaMemcpyDeviceToHost));
			frame.s_activeDepthGPU = &frame;
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////
	//// SIFT Intensity Image
	//////////////////////////////////////////////////////////////////////////////////////
	//CUDAImageUtil::resampleToIntensity(d_intensitySIFT, m_widthSIFT, m_heightSIFT, d_colorInput, m_RGBDSensor->getColorWidth(), m_RGBDSensor->getColorHeight());

	if (GlobalBundlingState::get().s_enableGlobalTimings) { cudaDeviceSynchronize(); s_timer.stop(); TimingLog::getFrameTiming(true).timeSensorProcess = s_timer.getElapsedTimeMS(); }

	m_currFrame++;
	return true;
}
