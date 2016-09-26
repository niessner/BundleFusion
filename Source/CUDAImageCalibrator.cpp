#include "stdafx.h"

#include "CUDAImageCalibrator.h"
#include "TimingLog.h"
#include <algorithm>


void CUDAImageCalibrator::OnD3D11DestroyDevice()
{
	MLIB_CUDA_SAFE_FREE(d_dummyColor);

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();
}

HRESULT CUDAImageCalibrator::OnD3D11CreateDevice(ID3D11Device* device, unsigned int width, unsigned int height)
{
	HRESULT hr = S_OK;

	m_width = width;
	m_height = height;
	const unsigned int bufferDimDepth = width * height;
	const unsigned int bufferDimColor = width * height;

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_dummyColor, sizeof(float4)*bufferDimColor));

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(device, width, height));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(device, width, height, formats));

	return hr;
}

HRESULT CUDAImageCalibrator::process(ID3D11DeviceContext* context, float* d_depth, const mat4f& colorIntrinsics, const mat4f& depthIntrinsicsInv, const mat4f& depthExtrinsics)
{
	HRESULT hr = S_OK;

	////////////////////////////////////////////////////////////////////////////////////
	// Render to Color Space
	////////////////////////////////////////////////////////////////////////////////////

	//Start Timing
	//if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.start(); }

	g_CustomRenderTarget.Clear(context);
	g_CustomRenderTarget.Bind(context);
	g_RGBDRenderer.RenderDepthMap(context,
		d_depth, d_dummyColor, m_width, m_height,
		depthIntrinsicsInv, depthExtrinsics, colorIntrinsics,
		g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(),
		GlobalAppState::get().s_remappingDepthDiscontinuityThresOffset, GlobalAppState::get().s_remappingDepthDiscontinuityThresLin);
	g_CustomRenderTarget.Unbind(context);
	g_CustomRenderTarget.copyToCuda(d_depth, 0);

	// Stop Timing
	//if (GlobalAppState::get().s_timingsDetailledEnabled) { cutilSafeCall(cudaDeviceSynchronize()); m_timer.stop(); TimingLog::totalTimeRemapDepth += m_timer.getElapsedTimeMS(); TimingLog::countTimeRemapDepth++; }

	return hr;
}

