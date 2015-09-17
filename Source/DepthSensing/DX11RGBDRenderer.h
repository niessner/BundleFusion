#pragma once

#include "DXUT.h"
#include "DX11Utils.h"

#include "cudaUtil.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h> 
#include "mLib.h"

struct CB_RGBDRenderer
{
	D3DXMATRIX		m_mIntrinsicInverse;
	D3DXMATRIX		m_mExtrinsic;			//model-view
	D3DXMATRIX		m_mIntrinsicNew;		//for 'real-world' depth range
	D3DXMATRIX		m_mProjection;			//for graphics rendering

	unsigned int	m_uScreenWidth;
	unsigned int	m_uScreenHeight;
	unsigned int	m_uDepthImageWidth;
	unsigned int	m_uDepthImageHeight;

	float			m_fDepthThreshOffset;
	float			m_fDepthThreshLin;
	float2			m_vDummy;
};

class DX11RGBDRenderer
{
public:

	DX11RGBDRenderer()
	{
		m_EmptyVS = NULL;
		m_RGBDRendererGS = NULL;
		m_RGBDRendererRawDepthPS = NULL;
		m_cbRGBDRenderer = NULL;
		m_PointSampler = NULL;
		m_LinearSampler = NULL;

		m_pTextureFloat = NULL;
		m_pTextureFloatSRV = NULL;
		m_dCudaFloat = NULL;

		m_pTextureFloat4 = NULL;
		m_pTextureFloat4SRV = NULL;
		m_dCudaFloat4 = NULL;

		m_width = 0;
		m_height = 0;
	}

	~DX11RGBDRenderer()
	{
		OnD3D11DestroyDevice();
	}

	HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height)
	{
		HRESULT hr = S_OK;

		D3D11_BUFFER_DESC desc;
		ZeroMemory(&desc, sizeof(D3D11_BUFFER_DESC));
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.MiscFlags = 0;
		desc.ByteWidth = sizeof( CB_RGBDRenderer );
		V_RETURN( pd3dDevice->CreateBuffer( &desc, NULL, &m_cbRGBDRenderer ) );

		ID3DBlob* pBlob = NULL;
		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "EmptyVS", "vs_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_EmptyVS));

		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "RGBDRendererGS", "gs_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_RGBDRendererGS));

		V_RETURN(CompileShaderFromFile(L"Shaders/RGBDRenderer.hlsl", "RGBDRendererRawDepthPS", "ps_5_0", &pBlob));
		V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_RGBDRendererRawDepthPS));

		SAFE_RELEASE(pBlob);
		
		D3D11_SAMPLER_DESC sdesc;
		ZeroMemory(&sdesc, sizeof(sdesc));
		sdesc.AddressU =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.AddressV =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.AddressW =  D3D11_TEXTURE_ADDRESS_CLAMP;
		sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
		V_RETURN(pd3dDevice->CreateSamplerState(&sdesc, &m_PointSampler));
		sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		V_RETURN(pd3dDevice->CreateSamplerState(&sdesc, &m_LinearSampler));

		V_RETURN(OnResize(pd3dDevice, width, height));

		return hr;
	}

	HRESULT OnResize(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height)
	{
		HRESULT hr = S_OK;
		if (width == 0 || height == 0)	return S_FALSE;
		if (m_width == width && m_height == height)	return hr;

		m_width = width;
		m_height = height;

		{
			SAFE_RELEASE(m_pTextureFloat);
			SAFE_RELEASE(m_pTextureFloatSRV);
			if (m_dCudaFloat) {
				cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat));
				m_dCudaFloat = NULL;
			}

			D3D11_TEXTURE2D_DESC descTex;
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32_FLOAT;
			descTex.Width = m_width;
			descTex.Height = m_height;

			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &m_pTextureFloat));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pTextureFloat, NULL, &m_pTextureFloatSRV));

			cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaFloat, m_pTextureFloat, cudaGraphicsRegisterFlagsNone));
			cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaFloat, cudaGraphicsMapFlagsWriteDiscard));
		}

		{
			SAFE_RELEASE(m_pTextureFloat4);
			SAFE_RELEASE(m_pTextureFloat4SRV);
			if (m_dCudaFloat4) {
				cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat4));
				m_dCudaFloat4 = NULL;
			}

			D3D11_TEXTURE2D_DESC descTex;
			ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
			descTex.Usage = D3D11_USAGE_DEFAULT;
			descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			descTex.CPUAccessFlags = 0;
			descTex.MiscFlags = 0;
			descTex.SampleDesc.Count = 1;
			descTex.SampleDesc.Quality = 0;
			descTex.ArraySize = 1;
			descTex.MipLevels = 1;
			descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
			descTex.Width = m_width;
			descTex.Height = m_height;

			V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &m_pTextureFloat4));
			V_RETURN(pd3dDevice->CreateShaderResourceView(m_pTextureFloat4, NULL, &m_pTextureFloat4SRV));

			cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaFloat4, m_pTextureFloat4, cudaGraphicsRegisterFlagsNone));
			cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaFloat4, cudaGraphicsMapFlagsWriteDiscard));
		}
		
		return hr;
	}

	void OnD3D11DestroyDevice()
	{
		SAFE_RELEASE(m_EmptyVS);
		SAFE_RELEASE(m_RGBDRendererGS);
		SAFE_RELEASE(m_RGBDRendererRawDepthPS);
		SAFE_RELEASE(m_cbRGBDRenderer);
		SAFE_RELEASE(m_PointSampler);
		SAFE_RELEASE(m_LinearSampler);

		SAFE_RELEASE(m_pTextureFloat);
		SAFE_RELEASE(m_pTextureFloatSRV);
		if (m_dCudaFloat)
		{
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat));
			m_dCudaFloat = NULL;
		}

		SAFE_RELEASE(m_pTextureFloat4);
		SAFE_RELEASE(m_pTextureFloat4SRV);
		if (m_dCudaFloat4)
		{
			cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaFloat4));
			m_dCudaFloat4 = NULL;
		}

		m_width = 0;
		m_height = 0;
	}

	HRESULT RenderDepthMap(
		ID3D11DeviceContext* pd3dDeviceContext, 
		float* d_depthMap,
		float4* d_colorMap,
		unsigned int width, 
		unsigned int height, 
		const mat4f& intrinsicDepthToWorld, 
		const mat4f& modelview, 
		const mat4f& intrinsicWorldToDepth, 
		unsigned int screenWidth,
		unsigned int screenheight,
		float depthThreshOffset,
		float depthThreshLin)
	{
		HRESULT hr = S_OK;
		V_RETURN(OnResize(DXUTGetD3D11Device(), width, height));	//returns if width/height did not change

		cudaArray* in_array;
		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat, 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat, 0, 0));
		cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_depthMap, sizeof(float)*width*height, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat, 0));	// Unmap DX texture

		cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaFloat4, 0));	// Map DX texture to Cuda
		cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, m_dCudaFloat4, 0, 0));
		cutilSafeCall(cudaMemcpyToArray(in_array, 0, 0, d_colorMap, 4*sizeof(float)*width*height, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaFloat4, 0));	// Unmap DX texture

		pd3dDeviceContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
		pd3dDeviceContext->IASetInputLayout( NULL );
		unsigned int stride = 0;
		unsigned int offset = 0;
		pd3dDeviceContext->IASetVertexBuffers(0, 0, NULL, &stride, &offset);

		ID3D11SamplerState* ss[] = {m_PointSampler, m_LinearSampler};
		pd3dDeviceContext->GSSetSamplers(0, 2, ss);
		pd3dDeviceContext->PSSetSamplers(0, 2, ss);
		pd3dDeviceContext->VSSetShader(m_EmptyVS, NULL, 0);
		pd3dDeviceContext->GSSetShader(m_RGBDRendererGS, NULL, 0);
		pd3dDeviceContext->PSSetShader(m_RGBDRendererRawDepthPS, NULL, 0);

		//mapping the constant buffer
		{
			D3D11_MAPPED_SUBRESOURCE MappedResource;
			V(pd3dDeviceContext->Map( m_cbRGBDRenderer, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ));
			CB_RGBDRenderer* pCB = ( CB_RGBDRenderer* )MappedResource.pData;
			memcpy(&pCB->m_mIntrinsicInverse, &intrinsicDepthToWorld, sizeof(float)*16);
			memcpy(&pCB->m_mIntrinsicNew, &intrinsicWorldToDepth, sizeof(float)*16);
			memcpy(&pCB->m_mExtrinsic, &modelview, sizeof(float)*16);
			pCB->m_uScreenHeight = screenheight;
			pCB->m_uScreenWidth = screenWidth;
			pCB->m_uDepthImageWidth = m_width;
			pCB->m_uDepthImageHeight = m_height;
			pCB->m_fDepthThreshOffset = depthThreshOffset;
			pCB->m_fDepthThreshLin = depthThreshLin;
			pd3dDeviceContext->Unmap( m_cbRGBDRenderer, 0 );

			pd3dDeviceContext->GSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
			pd3dDeviceContext->PSSetConstantBuffers(0, 1, &m_cbRGBDRenderer);
		}
		pd3dDeviceContext->GSSetShaderResources(0, 1, &m_pTextureFloatSRV);
		pd3dDeviceContext->PSSetShaderResources(0, 1, &m_pTextureFloatSRV);

		pd3dDeviceContext->GSSetShaderResources(1, 1, &m_pTextureFloat4SRV);
		pd3dDeviceContext->PSSetShaderResources(1, 1, &m_pTextureFloat4SRV);


		unsigned int numQuads = width*height;
		pd3dDeviceContext->Draw(numQuads, 0);

		//! reset the state
		pd3dDeviceContext->VSSetShader(NULL, NULL, 0);
		pd3dDeviceContext->GSSetShader(NULL, NULL, 0);
		pd3dDeviceContext->PSSetShader(NULL, NULL, 0);

		ID3D11ShaderResourceView* srvNULL[] = {NULL, NULL};
		pd3dDeviceContext->GSSetShaderResources(0, 2, srvNULL);
		pd3dDeviceContext->PSSetShaderResources(0, 2, srvNULL);	
		ID3D11Buffer* buffNULL[] = {NULL};
		pd3dDeviceContext->GSSetConstantBuffers(0, 1, buffNULL);
		pd3dDeviceContext->PSSetConstantBuffers(0, 1, buffNULL);

		return hr;
	}

private:

	unsigned int m_width;
	unsigned int m_height;

	ID3D11VertexShader*		m_EmptyVS;
	ID3D11GeometryShader*	m_RGBDRendererGS;
	ID3D11PixelShader*		m_RGBDRendererRawDepthPS;

	ID3D11Buffer*			m_cbRGBDRenderer;
	ID3D11SamplerState*		m_PointSampler;
	ID3D11SamplerState*		m_LinearSampler;

	ID3D11Texture2D*			m_pTextureFloat;
	ID3D11ShaderResourceView*	m_pTextureFloatSRV;
	cudaGraphicsResource*		m_dCudaFloat;

	ID3D11Texture2D*			m_pTextureFloat4;
	ID3D11ShaderResourceView*	m_pTextureFloat4SRV;
	cudaGraphicsResource*		m_dCudaFloat4;
};
