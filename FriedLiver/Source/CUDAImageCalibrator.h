#pragma once

#include "RGBDSensor.h"
#include "GlobalAppState.h"

#include <cuda_runtime.h> 
#include <cuda_d3d11_interop.h> 

#include "DepthSensing/DX11RGBDRenderer.h"
#include "DepthSensing/DX11CustomRenderTarget.h"

class CUDAImageCalibrator
{
	public:

		CUDAImageCalibrator() { d_dummyColor = NULL; }
		~CUDAImageCalibrator() {}

		void OnD3D11DestroyDevice();
			
		HRESULT OnD3D11CreateDevice(ID3D11Device* device, unsigned int width, unsigned int height);

		HRESULT process(ID3D11DeviceContext* context, float* d_depth, const mat4f& colorIntrinsics, const mat4f& depthIntrinsicsInv, const mat4f& depthExtrinsics);

	private:

		unsigned int m_width, m_height;
		DX11RGBDRenderer			g_RGBDRenderer;
		DX11CustomRenderTarget		g_CustomRenderTarget;

		float4*						d_dummyColor; // don't need to render color but the rgbdrenderer expects a float4 color array...

		Timer m_timer;
};
