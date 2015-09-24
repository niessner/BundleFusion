#pragma once

#include "DXUT.h"
#include "DX11Utils.h"

#include "cudaUtil.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h> 

class DX11CustomRenderTarget
{
public:
	DX11CustomRenderTarget();
	~DX11CustomRenderTarget();

	HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats);

	HRESULT OnResize(ID3D11Device* pd3dDevice, UINT width, UINT heigth);

	void OnD3D11DestroyDevice();

	void Clear(ID3D11DeviceContext* pd3dDeviceContext, float clearDepth = 1.f);
	void Bind(ID3D11DeviceContext* pd3dDeviceContext);
	void Unbind(ID3D11DeviceContext* pd3dDeviceContext);

	ID3D11ShaderResourceView*	GetSRV(UINT which = 0);
	ID3D11ShaderResourceView**	GetSRVs();

	void copyToCuda(float* d_target, UINT which = 0)
	{
		unsigned int numChannels = 0;
		if(m_TextureFormats[which] == DXGI_FORMAT_R32_FLOAT)			   numChannels = 1;
		else if(m_TextureFormats[which] == DXGI_FORMAT_R32G32B32A32_FLOAT) numChannels = 4;
		else															   MLIB_EXCEPTION("unknown texture format");

		cudaArray* in_array;
		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat[which], 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat[which], 0, 0));
		cudaMemcpyFromArray(d_target, in_array, 0, 0, numChannels*sizeof(float)*m_Width*m_Height, cudaMemcpyDeviceToDevice);
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat[which], 0));	// Unmap DX texture
	}

	cudaArray* mapToCuda(UINT which = 0)
	{
		cudaArray* in_array;
		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat[which], 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat[which], 0, 0));
		return in_array;
	}

	void unmapCuda(UINT which = 0) {
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat[which], 0));	// Unmap DX texture
	}

	void copyToHost(BYTE* &res, unsigned int& elementByteWidth, UINT which = 0) {
		ID3D11Device* pDevice = DXUTGetD3D11Device();
		ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
		ID3D11Texture2D* debugtex = NULL;


		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		m_Targets[which]->GetDesc( &desc );
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.MiscFlags = 0;
		pDevice->CreateTexture2D(&desc, NULL, &debugtex);
		pd3dImmediateContext->CopyResource( debugtex, m_Targets[which] );

		if (desc.Format == DXGI_FORMAT_R32_FLOAT) elementByteWidth = 1 * 4;
		else if (desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT) elementByteWidth = 4 * 4;
		else if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) elementByteWidth = 4 * 1;//else if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) elementByteWidth = 4 * 1;
		else MLIB_EXCEPTION("unknown texture format");


		BYTE *cpuMemory = new BYTE[desc.Height * desc.Width * elementByteWidth];
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		pd3dImmediateContext->Map(debugtex, D3D11CalcSubresource(0,0,0), D3D11_MAP_READ, 0, &mappedResource);	
		memcpy((void*)cpuMemory, (void*)mappedResource.pData, desc.Height * desc.Width * elementByteWidth);
		pd3dImmediateContext->Unmap( debugtex, 0 );
		SAFE_RELEASE(debugtex);

		res = cpuMemory;
	}

	UINT GetNumTargets() const;

	UINT getWidth() const
	{
		return m_Width;
	}
	UINT getHeight() const
	{
		return m_Height;
	}

private:

	UINT						m_uNumTargets;
	std::vector<DXGI_FORMAT>	m_TextureFormats;

	ID3D11Texture2D**			m_Targets;
	ID3D11ShaderResourceView**	m_TargetsSRV;
	ID3D11RenderTargetView**	m_TargetsRTV;

	ID3D11Texture2D*			m_DepthStencil;
	ID3D11DepthStencilView*		m_DepthStencilDSV;
	ID3D11ShaderResourceView*	m_DepthStencilSRV;

	UINT m_Width;
	UINT m_Height;

	cudaGraphicsResource**		m_dCudaFloat;
};
