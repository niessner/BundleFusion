
#include "stdafx.h"

#include "DXUT.h"
#include "DX11CustomRenderTarget.h"


DX11CustomRenderTarget::DX11CustomRenderTarget(void)
{	
	m_Targets = NULL;
	m_TargetsSRV = NULL;
	m_TargetsRTV = NULL;

	m_DepthStencil = NULL;
	m_DepthStencilDSV = NULL;
	m_DepthStencilSRV = NULL;

	m_dCudaFloat = NULL;
	
	m_Width = m_Height = 0;
	m_uNumTargets = 0;
}

DX11CustomRenderTarget::~DX11CustomRenderTarget(void)
{
}

HRESULT DX11CustomRenderTarget::OnResize( ID3D11Device* pd3dDevice, UINT width, UINT height)
{
	HRESULT hr = S_OK;

	if (width == m_Width && height == m_Height)	return hr;
	OnD3D11DestroyDevice();

	m_Width = width;
	m_Height = height;

	m_Targets = new ID3D11Texture2D*[m_uNumTargets];
	m_TargetsRTV = new ID3D11RenderTargetView*[m_uNumTargets];
	m_TargetsSRV = new ID3D11ShaderResourceView*[m_uNumTargets];
	
	//creating targets
	{
		for (UINT i = 0; i < m_uNumTargets; i++)
		{
			D3D11_TEXTURE2D_DESC textureDesc;
			ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
			textureDesc.Width = m_Width;
			textureDesc.Height = height;
			textureDesc.MipLevels = 1;
			textureDesc.ArraySize = 1;
			textureDesc.Format = m_TextureFormats[i];
			//TODO fix this for MSAA/CSAA
			textureDesc.SampleDesc.Count = 1;
			textureDesc.SampleDesc.Quality = 0;
			textureDesc.Usage = D3D11_USAGE_DEFAULT;
			textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
			textureDesc.CPUAccessFlags = 0;
			textureDesc.MiscFlags = 0;
	
			D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
			ZeroMemory(&rtvDesc, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
			rtvDesc.Format = textureDesc.Format;
			rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			rtvDesc.Texture2D.MipSlice = 0;

			D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
			ZeroMemory(&srvDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
			srvDesc.Format = textureDesc.Format;
			srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			srvDesc.Texture2D.MipLevels = 1;
			srvDesc.Texture2D.MostDetailedMip = 0;

			V_RETURN(pd3dDevice->CreateTexture2D(&textureDesc, NULL, &m_Targets[i]));
			V_RETURN(pd3dDevice->CreateRenderTargetView(m_Targets[i], &rtvDesc, &m_TargetsRTV[i]));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_Targets[i], &srvDesc, &m_TargetsSRV[i]));
		}
	}

	//creating depth stencil
	{
		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.Format = DXGI_FORMAT_R32_TYPELESS;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL;
		desc.MipLevels = 1;
		desc.ArraySize = 1;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.CPUAccessFlags = 0;
		desc.Height = height;
		desc.Width = width;
		V_RETURN(pd3dDevice->CreateTexture2D(&desc, NULL, &m_DepthStencil));


		D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
		ZeroMemory(&descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
		descDSV.Format = DXGI_FORMAT_D32_FLOAT;
		if (desc.SampleDesc.Count > 1)	{
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;
		} else {
			descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
			descDSV.Texture2D.MipSlice = 0;
		}

		V_RETURN(pd3dDevice->CreateDepthStencilView(m_DepthStencil, &descDSV, &m_DepthStencilDSV));

		D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
		ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
		descSRV.Format = DXGI_FORMAT_R32_FLOAT;
		descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		descSRV.Texture2D.MipLevels = 1;
		descSRV.Texture2D.MostDetailedMip = 0;

		V_RETURN(pd3dDevice->CreateShaderResourceView(m_DepthStencil, &descSRV, &m_DepthStencilSRV));
	}

	if (m_dCudaFloat) {
		for (UINT i = 0; i < m_uNumTargets; i++) {
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat[i]));
		}
		SAFE_DELETE_ARRAY(m_dCudaFloat)
	}

	m_dCudaFloat = new cudaGraphicsResource*[m_uNumTargets];
	for (UINT i = 0; i < m_uNumTargets; i++) {
		cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaFloat[i], m_Targets[i], cudaGraphicsRegisterFlagsNone));
		cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaFloat[i], cudaGraphicsMapFlagsReadOnly));
	}

	return hr;
}

void DX11CustomRenderTarget::OnD3D11DestroyDevice()
{
	for (UINT i = 0; i < m_uNumTargets; i++) {
		if (m_Targets)		SAFE_RELEASE(m_Targets[i]);
		if (m_TargetsRTV)	SAFE_RELEASE(m_TargetsRTV[i]);
		if (m_TargetsSRV)	SAFE_RELEASE(m_TargetsSRV[i]);
	}

	SAFE_DELETE_ARRAY(m_Targets);
	SAFE_DELETE_ARRAY(m_TargetsRTV);
	SAFE_DELETE_ARRAY(m_TargetsSRV);

	SAFE_RELEASE(m_DepthStencil);
	SAFE_RELEASE(m_DepthStencilDSV);
	SAFE_RELEASE(m_DepthStencilSRV);

	if (m_dCudaFloat) {
		for (UINT i = 0; i < m_uNumTargets; i++) {
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat[i]));
		}
		SAFE_DELETE_ARRAY(m_dCudaFloat)
	}

	m_Width = 0;
	m_Height = 0;
}

HRESULT DX11CustomRenderTarget::OnD3D11CreateDevice( ID3D11Device* pd3dDevice, unsigned int width, unsigned int height, const std::vector<DXGI_FORMAT>& formats)
{
	HRESULT hr = S_OK;
	m_TextureFormats = formats;
	m_uNumTargets = (unsigned int)formats.size();

	V_RETURN(OnResize(pd3dDevice, width, height));

	return hr;
}
 
void DX11CustomRenderTarget::Bind( ID3D11DeviceContext* pd3dDeviceContext )
{
	D3D11_VIEWPORT vp;
	vp.Width = (FLOAT)m_Width;
	vp.Height = (FLOAT)m_Height;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;

	pd3dDeviceContext->RSSetViewports( 1, &vp );
	pd3dDeviceContext->OMSetRenderTargets(m_uNumTargets, m_TargetsRTV, m_DepthStencilDSV);
}

void DX11CustomRenderTarget::Unbind( ID3D11DeviceContext* pd3dDeviceContext )
{
	D3D11_VIEWPORT vp;
	vp.Width = (FLOAT)DXUTGetDXGIBackBufferSurfaceDesc()->Width;
	vp.Height = (FLOAT)DXUTGetDXGIBackBufferSurfaceDesc()->Height;
	vp.MinDepth = 0;
	vp.MaxDepth = 1;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	pd3dDeviceContext->RSSetViewports( 1, &vp );

	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();

	pd3dDeviceContext->OMSetRenderTargets(1, &pRTV, pDSV);
}

void DX11CustomRenderTarget::Clear( ID3D11DeviceContext* pd3dDeviceContext, float clearDepth/* = 1.f*/ )
{
	for (UINT i = 0; i < m_uNumTargets; i++)
	{
		if (m_TextureFormats[i] == DXGI_FORMAT_R32G32B32A32_FLOAT)
		{
			float clearColor[] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 1.0f};
			pd3dDeviceContext->ClearRenderTargetView(m_TargetsRTV[i], clearColor);
		}
		else if (m_TextureFormats[i] == DXGI_FORMAT_R32_FLOAT)
		{
			float clearColor[] = {-std::numeric_limits<float>::infinity()};
			pd3dDeviceContext->ClearRenderTargetView(m_TargetsRTV[i], clearColor);
		}
		else if (m_TextureFormats[i] == DXGI_FORMAT_R8G8B8A8_UNORM) {//else if (m_TextureFormats[i] == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) {
			float clearColor[] = {0};
			pd3dDeviceContext->ClearRenderTargetView(m_TargetsRTV[i], clearColor);
		}
		else
		{
			throw MLIB_EXCEPTION("unknown texture format");
		}
	}
		
	pd3dDeviceContext->ClearDepthStencilView( m_DepthStencilDSV, D3D11_CLEAR_DEPTH, clearDepth, 0);
}

ID3D11ShaderResourceView* DX11CustomRenderTarget::GetSRV( UINT which /*= 0*/ )
{
	return m_TargetsSRV[which];
}

UINT DX11CustomRenderTarget::GetNumTargets() const
{
	return m_uNumTargets;
}

ID3D11ShaderResourceView** DX11CustomRenderTarget::GetSRVs()
{
	return m_TargetsSRV;
}
