#pragma once

#include <D3D11.h>
#include "DXUT.h"
#include "DX11Utils.h"
#include "DX11CustomRenderTarget.h"

#include "cudaUtil.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h> 

#include "GlobalAppState.h"
#include "TimingLogDepthSensing.h"

class DX11RayIntervalSplatting
{
public:

	DX11RayIntervalSplatting();
	~DX11RayIntervalSplatting();

	HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height);

	void OnD3D11DestroyDevice();


	HRESULT rayIntervalSplatting(ID3D11DeviceContext* context, const HashDataStruct& hashData, RayCastData& rayCastData, RayCastParams& rayCastParams, unsigned int numVertices);

	cudaArray* mapMinToCuda() {
		return m_customRenderTargetMin.mapToCuda();
	}

	cudaArray* mapMaxToCuda() {
		return m_customRenderTargetMax.mapToCuda();
	}

	//! unmaps both min and max
	void unmapCuda() {
		m_customRenderTargetMin.unmapCuda();
		m_customRenderTargetMax.unmapCuda();
	}

private:

	HRESULT initialize(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height);

	void destroy();

	ID3D11InputLayout* m_VertexLayout;

	ID3D11VertexShader* m_pVertexShaderSplatting;
	//ID3D11GeometryShader* m_pGeometryShaderSplatting;
	ID3D11PixelShader* m_pPixelShaderSplatting;

	ID3D11DepthStencilState* m_pDepthStencilStateSplattingMin;
	ID3D11DepthStencilState* m_pDepthStencilStateSplattingMax;

	ID3D11RasterizerState* m_pRastState;

	ID3D11Buffer*				m_pVertexBufferFloat4;
	//ID3D11ShaderResourceView*	m_pVertexBufferFloat4SRV;
	cudaGraphicsResource*		m_dCudaVertexBufferFloat4;

	DX11CustomRenderTarget		m_customRenderTargetMin;
	DX11CustomRenderTarget		m_customRenderTargetMax;

	Timer m_timer;
	//Timer m_timerCUDA;
	//Timer m_timerDX11;
};

