
#include "stdafx.h"

#include "DX11QuadDrawer.h"

extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height);
extern "C" void depthToHSV(float4* d_output, const float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth);

//--------------------------------------------------------------------------------------
// Constant buffers
//--------------------------------------------------------------------------------------
#pragma pack(push,1)
struct CB_POW_TWO_RATIOS
{
	float	fWidthoverNextPowOfTwo;
	float	fHeightoverNextPowOfTwo;
	float	fScale;
	float	dummy1;
};
#pragma pack(pop)

ID3D11InputLayout*	DX11QuadDrawer::s_VertexLayout = NULL;
ID3D11Buffer*		DX11QuadDrawer::s_VertexBuffer = NULL;
ID3D11Buffer*		DX11QuadDrawer::s_IndexBuffer = NULL;

ID3D11VertexShader*	DX11QuadDrawer::s_VertexShader = NULL;
ID3D11PixelShader*	DX11QuadDrawer::s_PixelShaderFloat = NULL;
ID3D11PixelShader*	DX11QuadDrawer::s_PixelShaderRGBA = NULL;
ID3D11PixelShader*	DX11QuadDrawer::s_PixelShader3 = NULL;

ID3D11Buffer*		DX11QuadDrawer::s_CBquad = NULL;

ID3D11SamplerState*	DX11QuadDrawer::s_PointSampler = NULL;

ID3D11Buffer*       DX11QuadDrawer::s_pcbVSPowTwoRatios = NULL;

ID3D11Texture2D*			DX11QuadDrawer::s_pTmpTexture = NULL;
ID3D11ShaderResourceView*	DX11QuadDrawer::s_pTmpTextureSRV = NULL;
cudaGraphicsResource*		DX11QuadDrawer::s_dCuda = NULL;

ID3D11Texture2D*			DX11QuadDrawer::s_pTmpTexture2 = NULL;
ID3D11ShaderResourceView*	DX11QuadDrawer::s_pTmpTexture2SRV = NULL;
cudaGraphicsResource*		DX11QuadDrawer::s_dCuda2 = NULL;

ID3D11Texture2D*			DX11QuadDrawer::s_pTmpTextureFloat = NULL;
ID3D11ShaderResourceView*	DX11QuadDrawer::s_pTmpTextureFloatSRV = NULL;
cudaGraphicsResource*		DX11QuadDrawer::s_dCudaFloat = NULL;

ID3D11Texture2D*			DX11QuadDrawer::s_pTmpTexture2Float = NULL;
ID3D11ShaderResourceView*	DX11QuadDrawer::s_pTmpTexture2FloatSRV = NULL;
cudaGraphicsResource*		DX11QuadDrawer::s_dCuda2Float = NULL;

unsigned int DX11QuadDrawer::s_tmpTextureWidth = 640;
unsigned int DX11QuadDrawer::s_tmpTextureHeight = 480;

unsigned int DX11QuadDrawer::s_tmpTextureWidth2 = 1280;
unsigned int DX11QuadDrawer::s_tmpTextureHeight2 = 960;

// static variables

DX11QuadDrawer::DX11QuadDrawer(void)
{
}


DX11QuadDrawer::~DX11QuadDrawer(void)
{
}

HRESULT DX11QuadDrawer::OnD3D11CreateDevice( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;

	V_RETURN(CompileShaderFromFile(L"Shaders/QuadDrawer.hlsl", "QuadVS", "vs_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_VertexShader));

	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	
	UINT numElements = sizeof( layout ) / sizeof( layout[0] );
	V_RETURN(pd3dDevice->CreateInputLayout(layout, numElements, pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &s_VertexLayout));
	SAFE_RELEASE(pBlob);


	V_RETURN(CompileShaderFromFile(L"Shaders/QuadDrawer.hlsl", "QuadFloatPS", "ps_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_PixelShaderFloat));
	SAFE_RELEASE(pBlob);

	V_RETURN(CompileShaderFromFile(L"Shaders/QuadDrawer.hlsl", "QuadRGBAPS", "ps_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_PixelShaderRGBA));
	SAFE_RELEASE(pBlob);

	//V_RETURN(CompileShaderFromFile(L"QuadDrawer.hlsl", "QuadPS2", "ps_5_0", &pBlob));
	//V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_PixelShader2));
	//SAFE_RELEASE(pBlob);

	//V_RETURN(CompileShaderFromFile(L"Float4Pyramid.hlsl", "QuadPS3", "ps_5_0", &pBlob));
	//V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_PixelShader3));
	//SAFE_RELEASE(pBlob);

	SimpleVertex vertices[] =
	{
		{ D3DXVECTOR3( -1.0f, -1.0f, 0.0f ), D3DXVECTOR2( 0.0f, 1.0f ) },
		{ D3DXVECTOR3( 1.0f, -1.0f, 0.0f ), D3DXVECTOR2( 1.0f, 1.0f ) },
		{ D3DXVECTOR3( 1.0f, 1.0f, 0.0f ), D3DXVECTOR2( 1.0f, 0.0f ) },
		{ D3DXVECTOR3( -1.0f, 1.0f, 0.0f ), D3DXVECTOR2( 0.0f, 0.0f ) }
	};

	D3D11_BUFFER_DESC desc;
	ZeroMemory(&desc, sizeof(D3D11_BUFFER_DESC));
	desc.Usage = D3D11_USAGE_IMMUTABLE;
	desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	desc.ByteWidth = sizeof(SimpleVertex) * 4;

	D3D11_SUBRESOURCE_DATA InitData;
	ZeroMemory( &InitData, sizeof(InitData) );
	InitData.pSysMem = vertices;
	V_RETURN( pd3dDevice->CreateBuffer( &desc, &InitData, &s_VertexBuffer ) );

	ZeroMemory(&desc, sizeof(D3D11_BUFFER_DESC));
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	desc.MiscFlags = 0;
	desc.ByteWidth = sizeof( CB_QUAD );
	V_RETURN( pd3dDevice->CreateBuffer( &desc, NULL, &s_CBquad ) );

	// Create and set index buffer
	DWORD indices[] =
	{
		2,1,0,//0,1,2,
		0,3,2//2,3,0
	};

	desc.Usage = D3D11_USAGE_DEFAULT;
	desc.ByteWidth = sizeof( DWORD ) * 6;
	desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	desc.CPUAccessFlags = 0;
	desc.MiscFlags = 0;
	InitData.pSysMem = indices;
	V_RETURN(pd3dDevice->CreateBuffer( &desc, &InitData, &s_IndexBuffer ));

	D3D11_SAMPLER_DESC sdesc;
	ZeroMemory(&sdesc, sizeof(sdesc));
	sdesc.AddressU =  D3D11_TEXTURE_ADDRESS_CLAMP;
	sdesc.AddressV =  D3D11_TEXTURE_ADDRESS_CLAMP;
	sdesc.AddressW =  D3D11_TEXTURE_ADDRESS_CLAMP;
	sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	V_RETURN(pd3dDevice->CreateSamplerState(&sdesc, &s_PointSampler));

	// constant buffers---------------------------------------------------------------------------
	D3D11_BUFFER_DESC Desc;
	ZeroMemory(&Desc, sizeof(D3D11_BUFFER_DESC));
	Desc.Usage = D3D11_USAGE_DYNAMIC;
	Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	Desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	Desc.MiscFlags = 0;

	Desc.ByteWidth = sizeof( CB_POW_TWO_RATIOS );
	V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, &s_pcbVSPowTwoRatios ) );
	//--------------------------------------------------------------------------------------------

	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tmp Buffer for CUDA Interop
	///////////////////////////////////////////////////////////////////////////////////////////////

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
	descTex.Width = s_tmpTextureWidth;
	descTex.Height = s_tmpTextureHeight;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTexture));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTexture, NULL, &s_pTmpTextureSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCuda, s_pTmpTexture, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCuda, cudaGraphicsMapFlagsWriteDiscard));

	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tmp Buffer2 for CUDA Interop
	///////////////////////////////////////////////////////////////////////////////////////////////

	descTex.Width = s_tmpTextureWidth2;
	descTex.Height = s_tmpTextureHeight2;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTexture2));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTexture2, NULL, &s_pTmpTexture2SRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCuda2, s_pTmpTexture2, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCuda2, cudaGraphicsMapFlagsWriteDiscard));

	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tmp Buffer for CUDA Interop Float
	///////////////////////////////////////////////////////////////////////////////////////////////

	descTex.Format = DXGI_FORMAT_R32_FLOAT;
	descTex.Width = s_tmpTextureWidth;
	descTex.Height = s_tmpTextureHeight;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTextureFloat));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTextureFloat, NULL, &s_pTmpTextureFloatSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCudaFloat, s_pTmpTextureFloat, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCudaFloat, cudaGraphicsMapFlagsWriteDiscard));

	///////////////////////////////////////////////////////////////////////////////////////////////
	// Tmp Buffer2 for CUDA Interop Float
	///////////////////////////////////////////////////////////////////////////////////////////////

	descTex.Width = s_tmpTextureWidth2;
	descTex.Height = s_tmpTextureHeight2;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTexture2Float));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTexture2Float, NULL, &s_pTmpTexture2FloatSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCuda2Float, s_pTmpTexture2Float, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCuda2Float, cudaGraphicsMapFlagsWriteDiscard));

	return hr;
}

void DX11QuadDrawer::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(s_VertexShader);
	SAFE_RELEASE(s_PixelShaderFloat);
	SAFE_RELEASE(s_PixelShaderRGBA);
	SAFE_RELEASE(s_PixelShader3);
	SAFE_RELEASE(s_VertexLayout);
	SAFE_RELEASE(s_VertexBuffer);
	SAFE_RELEASE(s_IndexBuffer);
	SAFE_RELEASE(s_CBquad);
	SAFE_RELEASE(s_PointSampler);
	SAFE_RELEASE(s_pcbVSPowTwoRatios);

	SAFE_RELEASE(s_pTmpTextureFloat);
	SAFE_RELEASE(s_pTmpTextureFloatSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCudaFloat));

	SAFE_RELEASE(s_pTmpTexture2Float);
	SAFE_RELEASE(s_pTmpTexture2FloatSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCuda2Float));

	SAFE_RELEASE(s_pTmpTexture);
	SAFE_RELEASE(s_pTmpTextureSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCuda));

	SAFE_RELEASE(s_pTmpTexture2);
	SAFE_RELEASE(s_pTmpTexture2SRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCuda2));
}

HRESULT DX11QuadDrawer::RenderQuadDynamicDEPTHasHSV(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const float* d_data, float minDepth, float maxDepth, unsigned int width, unsigned int height, float scale /*= 1.0f*/, D3DXVECTOR2 Pow2Ratios /*= D3DXVECTOR2(1.0f, 1.0f)*/, ID3D11PixelShader* pixelShader /*= NULL*/)
{
	//TODO this function is not very efficient...
	float4* d_dataFloat4;
	cutilSafeCall(cudaMalloc(&d_dataFloat4, sizeof(float4)*width*height));
	depthToHSV(d_dataFloat4, d_data, width, height, minDepth, maxDepth);
	HRESULT hr = RenderQuadDynamic(pd3dDevice, pd3dDeviceContext, (float*)d_dataFloat4, 4, width, height, scale, Pow2Ratios);
	cutilSafeCall(cudaFree(d_dataFloat4));
	return hr;
}


HRESULT DX11QuadDrawer::RenderQuadDynamicUCHAR4(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const uchar4* d_data, unsigned int width, unsigned int height, float scale /*= 1.0f*/, D3DXVECTOR2 Pow2Ratios /*= D3DXVECTOR2(1.0f, 1.0f)*/, ID3D11PixelShader* pixelShader /*= NULL*/)
{
	//TODO this function is not very efficient...
	float4* d_dataFloat4;
	cutilSafeCall(cudaMalloc(&d_dataFloat4, sizeof(float4)*width*height));
	convertColorRawToFloat4(d_dataFloat4, (BYTE*)d_data, width, height);
	HRESULT hr = RenderQuadDynamic(pd3dDevice, pd3dDeviceContext, (float*)d_dataFloat4, 4, width, height, scale, Pow2Ratios);
	cutilSafeCall(cudaFree(d_dataFloat4));
	return hr;
}


HRESULT DX11QuadDrawer::RenderQuadDynamic(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, float* d_data, unsigned int nChannels, unsigned int width, unsigned int height, float scale /*= 1.0f */, D3DXVECTOR2 Pow2Ratios /*= D3DXVECTOR2(1.0f, 1.0f)*/, ID3D11PixelShader* pixelShader /*= NULL*/)
{
	HRESULT hr = S_OK;

	// Alloc Buffers
	ID3D11Texture2D*			s_pLocalTmpTexture;
	ID3D11ShaderResourceView*	s_pLocalTmpTextureSRV;
	cudaGraphicsResource*		s_dLocalCuda;

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
	descTex.Width = width;
	descTex.Height = height;

	if(nChannels == 1)		descTex.Format = DXGI_FORMAT_R32_FLOAT;
	else if(nChannels == 4) descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	else					assert(false);

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pLocalTmpTexture));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pLocalTmpTexture, NULL, &s_pLocalTmpTextureSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dLocalCuda, s_pLocalTmpTexture, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dLocalCuda, cudaGraphicsMapFlagsWriteDiscard));

	// Render
	RenderQuadHelper(pd3dDeviceContext, d_data, s_dLocalCuda, s_pLocalTmpTextureSRV, nChannels*sizeof(float)*width*height, scale, Pow2Ratios, pixelShader);

	// Dealloc
	SAFE_RELEASE(s_pLocalTmpTexture);
	SAFE_RELEASE(s_pLocalTmpTextureSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dLocalCuda));

	return hr;
}

void DX11QuadDrawer::RenderQuadHelper(ID3D11DeviceContext* pd3dDeviceContext, float* d_data, cudaGraphicsResource* pCuda, ID3D11ShaderResourceView* pTmpTextureSRV, unsigned int size, float scale, D3DXVECTOR2 Pow2Ratios , ID3D11PixelShader* pixelShader)
{
	cudaArray* in_array;
	cutilSafeCall(cudaGraphicsMapResources(1, &pCuda, 0)); // Map DX texture to Cuda
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, pCuda, 0, 0));
	cudaMemcpyToArray(in_array, 0, 0, d_data, size, cudaMemcpyDeviceToDevice);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &pCuda, 0)); // Unmap DX texture
	RenderQuad(pd3dDeviceContext, pTmpTextureSRV, scale, Pow2Ratios, pixelShader);
}

void DX11QuadDrawer::RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, float* d_data, unsigned int nChannels, unsigned int width, unsigned int height, float scale /*= 1.0f */, D3DXVECTOR2 Pow2Ratios /*= float2(1.0f, 1.0f)*/, ID3D11PixelShader* pixelShader /*= NULL*/)
{
	if(nChannels == 4)
	{
		if	   (width == s_tmpTextureWidth && height == s_tmpTextureHeight)		RenderQuadHelper(pd3dDeviceContext, d_data, s_dCuda, s_pTmpTextureSRV, 4*sizeof(float)*width*height, scale, Pow2Ratios, pixelShader);
		else if(width == s_tmpTextureWidth2 && height == s_tmpTextureHeight2)	RenderQuadHelper(pd3dDeviceContext, d_data, s_dCuda2, s_pTmpTexture2SRV, 4*sizeof(float)*width*height, scale, Pow2Ratios, pixelShader);
		else																	assert(false);
	}
	else if(nChannels == 1)
	{
		if	   (width == s_tmpTextureWidth && height == s_tmpTextureHeight)		RenderQuadHelper(pd3dDeviceContext, d_data, s_dCudaFloat, s_pTmpTextureFloatSRV, sizeof(float)*width*height, scale, Pow2Ratios, pixelShader);
		else if(width == s_tmpTextureWidth2 && height == s_tmpTextureHeight2)	RenderQuadHelper(pd3dDeviceContext, d_data, s_dCuda2Float, s_pTmpTexture2FloatSRV, sizeof(float)*width*height, scale, Pow2Ratios, pixelShader);
		else																	assert(false);
	}
	else
	{
		assert(false);
	}
}

void DX11QuadDrawer::RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* srv, float scale /*= 1.0f */, D3DXVECTOR2 Pow2Ratios /*= float2(1.0f, 1.0f)*/, ID3D11PixelShader* pixelShader /*= NULL*/)
{
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	pd3dDeviceContext->Map( s_pcbVSPowTwoRatios, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
	CB_POW_TWO_RATIOS* pCSPowTwoRatios = ( CB_POW_TWO_RATIOS* )MappedResource.pData;
	pCSPowTwoRatios->fWidthoverNextPowOfTwo = Pow2Ratios.x;
	pCSPowTwoRatios->fHeightoverNextPowOfTwo = Pow2Ratios.y;
	pCSPowTwoRatios->fScale = scale;
	pd3dDeviceContext->Unmap( s_pcbVSPowTwoRatios, 0 );

	pd3dDeviceContext->VSSetConstantBuffers( 10, 1, &s_pcbVSPowTwoRatios );
	pd3dDeviceContext->PSSetConstantBuffers( 10, 1, &s_pcbVSPowTwoRatios );
	pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pd3dDeviceContext->IASetInputLayout(s_VertexLayout);
	pd3dDeviceContext->VSSetShader(s_VertexShader, NULL, 0);

	if (!pixelShader) {
		D3D11_SHADER_RESOURCE_VIEW_DESC desSRV;
		srv->GetDesc( &desSRV );
	
		//RGBA
		if (desSRV.Format == DXGI_FORMAT_R8G8B8A8_UNORM || 
			desSRV.Format == DXGI_FORMAT_R32G32B32A32_FLOAT ||
			desSRV.Format == DXGI_FORMAT_R16G16B16A16_FLOAT || 
			desSRV.Format == DXGI_FORMAT_B8G8R8A8_UNORM
			){
			pd3dDeviceContext->PSSetShader(s_PixelShaderRGBA, NULL, 0);
		}
		//float
		else if (desSRV.Format == DXGI_FORMAT_R32_FLOAT)
		{
			pd3dDeviceContext->PSSetShader(s_PixelShaderFloat, NULL, 0);
		} else {
			assert(false);
		}
	} else {
		pd3dDeviceContext->PSSetShader(pixelShader, NULL, 0);
	}

	
	UINT stride = sizeof( SimpleVertex );
	UINT offset = 0;
	pd3dDeviceContext->IASetVertexBuffers(0, 1, &s_VertexBuffer, &stride, &offset);
	pd3dDeviceContext->IASetIndexBuffer(s_IndexBuffer, DXGI_FORMAT_R32_UINT, 0);

	pd3dDeviceContext->PSSetShaderResources(10, 1, &srv);
	pd3dDeviceContext->PSSetSamplers(10, 1, &s_PointSampler);
	pd3dDeviceContext->DrawIndexed(6, 0, 0);

	ID3D11ShaderResourceView* srvNULL[] = {NULL};
	pd3dDeviceContext->PSSetShaderResources(10, 1, srvNULL);
}

void DX11QuadDrawer::RenderQuad( ID3D11DeviceContext* pd3dDeviceContext, ID3D11PixelShader* pixelShader, ID3D11ShaderResourceView** srvs, UINT numShaderResourceViews, D3DXVECTOR2 Pow2Ratios /*= float2(1.0f, 1.0f)*/ )
{
	D3D11_MAPPED_SUBRESOURCE MappedResource;
	pd3dDeviceContext->Map( s_pcbVSPowTwoRatios, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource );
	CB_POW_TWO_RATIOS* pCSPowTwoRatios = ( CB_POW_TWO_RATIOS* )MappedResource.pData;
	pCSPowTwoRatios->fWidthoverNextPowOfTwo = Pow2Ratios.x;
	pCSPowTwoRatios->fHeightoverNextPowOfTwo = Pow2Ratios.y;
	pd3dDeviceContext->Unmap( s_pcbVSPowTwoRatios, 0 );

	pd3dDeviceContext->VSSetConstantBuffers( 10, 1, &s_pcbVSPowTwoRatios );
	pd3dDeviceContext->PSSetConstantBuffers( 10, 1, &s_pcbVSPowTwoRatios );

	pd3dDeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	pd3dDeviceContext->IASetInputLayout(s_VertexLayout);

	pd3dDeviceContext->VSSetShader(s_VertexShader, NULL, 0);
	pd3dDeviceContext->PSSetShader(pixelShader, NULL, 0);


	UINT stride = sizeof( SimpleVertex );
	UINT offset = 0;
	pd3dDeviceContext->IASetVertexBuffers(0, 1, &s_VertexBuffer, &stride, &offset);
	pd3dDeviceContext->IASetIndexBuffer(s_IndexBuffer, DXGI_FORMAT_R32_UINT, 0);

	pd3dDeviceContext->PSSetShaderResources(10, numShaderResourceViews, srvs);
	pd3dDeviceContext->PSSetSamplers(10, 1, &s_PointSampler);

	pd3dDeviceContext->DrawIndexed(6, 0, 0);

	pd3dDeviceContext->VSSetShader(NULL, NULL, 0);
	pd3dDeviceContext->PSSetShader(NULL, NULL, 0);
	ID3D11SamplerState* samplerNULL[] = {NULL};
	pd3dDeviceContext->PSSetSamplers(10, 1, samplerNULL);

	ID3D11ShaderResourceView* srvsNULL[] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
	pd3dDeviceContext->PSSetShaderResources(10, numShaderResourceViews, srvsNULL);
}
