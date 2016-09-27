
#include "stdafx.h"

#include "DX11PhongLighting.h"

ID3D11PixelShader* DX11PhongLighting::s_PixelShaderPhong = 0;
ID3D11Buffer* DX11PhongLighting::s_ConstantBuffer = 0;

DX11PhongLighting::ConstantBufferLight DX11PhongLighting::s_ConstantBufferLightCPU;
ID3D11Buffer* DX11PhongLighting::s_ConstantBufferLight = NULL;

ID3D11Texture2D*			DX11PhongLighting::s_pDepthStencil = NULL;
ID3D11DepthStencilView*		DX11PhongLighting::s_pDepthStencilDSV = NULL;
ID3D11ShaderResourceView*	DX11PhongLighting::s_pDepthStencilSRV = NULL;

ID3D11Texture2D*			DX11PhongLighting::s_pColors = NULL;
ID3D11RenderTargetView*		DX11PhongLighting::s_pColorsRTV = NULL;
ID3D11ShaderResourceView*	DX11PhongLighting::s_pColorsSRV = NULL;

ID3D11RasterizerState*		DX11PhongLighting::s_pRastStateDefault = NULL;
ID3D11DepthStencilState*	DX11PhongLighting::s_pDepthStencilStateDefault = NULL;

ID3D11Texture2D*			DX11PhongLighting::s_pTmpTexturePositions = NULL;
ID3D11ShaderResourceView*	DX11PhongLighting::s_pTmpTexturePositionsSRV = NULL;
cudaGraphicsResource*		DX11PhongLighting::s_dCudaPositions = NULL;

ID3D11Texture2D*			DX11PhongLighting::s_pTmpTextureNormals = NULL;
ID3D11ShaderResourceView*	DX11PhongLighting::s_pTmpTextureNormalsSRV = NULL;
cudaGraphicsResource*		DX11PhongLighting::s_dCudaNormals = NULL;

ID3D11Texture2D*			DX11PhongLighting::s_pTmpTextureColors = NULL;
ID3D11ShaderResourceView*	DX11PhongLighting::s_pTmpTextureColorsSRV = NULL;
cudaGraphicsResource*		DX11PhongLighting::s_dCudaColors = NULL;

unsigned int				DX11PhongLighting::s_width = 0;
unsigned int				DX11PhongLighting::s_height = 0;


HRESULT DX11PhongLighting::OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width, unsigned int height)
{
	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;

	if (width == 0) width = DXUTGetWindowWidth();
	if (height == 0) height = DXUTGetWindowHeight();
	s_width = width;
	s_height = height;

	V_RETURN(CompileShaderFromFile(L"Shaders/PhongLighting.hlsl", "PhongPS", "ps_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &s_PixelShaderPhong));
		
	SAFE_RELEASE(pBlob);

	// Constant Buffer
	D3D11_BUFFER_DESC cbDesc;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	cbDesc.MiscFlags = 0;
	cbDesc.ByteWidth = sizeof(cbConstant);

	V_RETURN(pd3dDevice->CreateBuffer(&cbDesc, NULL, &s_ConstantBuffer))

	// default light settings
	s_ConstantBufferLightCPU.SetDefault();

	V_RETURN(s_ConstantBufferLightCPU.CreateDX11BufferAndCopy(pd3dDevice,&s_ConstantBufferLight));

	// Depth Stencil
	D3D11_TEXTURE2D_DESC descTex;
	ZeroMemory(&descTex, sizeof(D3D11_TEXTURE2D_DESC));
	descTex.Usage = D3D11_USAGE_DEFAULT;
	descTex.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	descTex.CPUAccessFlags = 0;
	descTex.MiscFlags = 0;
	descTex.SampleDesc.Count = 1;
	descTex.SampleDesc.Quality = 0;
	descTex.ArraySize = 1;
	descTex.MipLevels = 1;
	descTex.Format = DXGI_FORMAT_R32_TYPELESS;
	descTex.Width = s_width;
	descTex.Height = s_height;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pDepthStencil));

	D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
	ZeroMemory(&descDSV, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	descDSV.Format = DXGI_FORMAT_D32_FLOAT;
	descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	descDSV.Texture2D.MipSlice = 0;
	V_RETURN(pd3dDevice->CreateDepthStencilView(s_pDepthStencil, &descDSV, &s_pDepthStencilDSV));

	D3D11_SHADER_RESOURCE_VIEW_DESC descSRV;
	ZeroMemory(&descSRV, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
	descSRV.Format = DXGI_FORMAT_R32_FLOAT;
	descSRV.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	descSRV.Texture2D.MipLevels = 1;
	descSRV.Texture2D.MostDetailedMip = 0;
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pDepthStencil, &descSRV, &s_pDepthStencilSRV));

	// Render Targets
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
	descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;

	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pColors));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pColors, NULL, &s_pColorsSRV));
	V_RETURN(pd3dDevice->CreateRenderTargetView(s_pColors, NULL, &s_pColorsRTV));

	// Rasterizer Stage
	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;
		
	V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &s_pRastStateDefault))

	// Depth Stencil
	D3D11_DEPTH_STENCIL_DESC stenDesc;
	ZeroMemory(&stenDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	stenDesc.DepthEnable = true;
	stenDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	stenDesc.DepthFunc =  D3D11_COMPARISON_LESS;
	stenDesc.StencilEnable = false;

	V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &s_pDepthStencilStateDefault))

	///////////////////////////////////////////////////////////////////////////////////////////////
	// CUDA interop
	///////////////////////////////////////////////////////////////////////////////////////////////
	descTex.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	descTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	descTex.Width  = GlobalAppState::get().s_rayCastWidth;
	descTex.Height = GlobalAppState::get().s_rayCastHeight;

	// Positions
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTexturePositions));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTexturePositions, NULL, &s_pTmpTexturePositionsSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCudaPositions, s_pTmpTexturePositions, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCudaPositions, cudaGraphicsMapFlagsWriteDiscard));

	// Normals
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTextureNormals));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTextureNormals, NULL, &s_pTmpTextureNormalsSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCudaNormals, s_pTmpTextureNormals, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCudaNormals, cudaGraphicsMapFlagsWriteDiscard));

	descTex.Width  = GlobalAppState::get().s_rayCastWidth;
	descTex.Height = GlobalAppState::get().s_rayCastHeight;

	// Colors
	V_RETURN(pd3dDevice->CreateTexture2D(&descTex, NULL, &s_pTmpTextureColors));
	V_RETURN(pd3dDevice->CreateShaderResourceView(s_pTmpTextureColors, NULL, &s_pTmpTextureColorsSRV));

	cutilSafeCall(cudaGraphicsD3D11RegisterResource(&s_dCudaColors, s_pTmpTextureColors, cudaGraphicsRegisterFlagsNone));
	cutilSafeCall(cudaGraphicsResourceSetMapFlags(s_dCudaColors, cudaGraphicsMapFlagsWriteDiscard));

	return hr;
}

void DX11PhongLighting::render(ID3D11DeviceContext* pd3dDeviceContext, float4* d_positions, float4* d_normals, float4* d_colors, bool useMaterial, unsigned int width, unsigned int height, const vec3f& overlayColor)
{
	cudaArray* in_array;
			
	// Positions
	cutilSafeCall(cudaGraphicsMapResources(1, &s_dCudaPositions, 0)); // Map DX texture to Cuda
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, s_dCudaPositions, 0, 0));
	cudaMemcpyToArray(in_array, 0, 0, d_positions, 4*sizeof(float)*width*height, cudaMemcpyDeviceToDevice);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &s_dCudaPositions, 0)); // Unmap DX texture

	// Normals
	cutilSafeCall(cudaGraphicsMapResources(1, &s_dCudaNormals, 0)); // Map DX texture to Cuda
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, s_dCudaNormals, 0, 0));
	cudaMemcpyToArray(in_array, 0, 0, d_normals, 4*sizeof(float)*width*height, cudaMemcpyDeviceToDevice);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &s_dCudaNormals, 0)); // Unmap DX texture

	// Colors
	cutilSafeCall(cudaGraphicsMapResources(1, &s_dCudaColors, 0)); // Map DX texture to Cuda
	cutilSafeCall(cudaGraphicsSubResourceGetMappedArray(&in_array, s_dCudaColors, 0, 0));
	cudaMemcpyToArray(in_array, 0, 0, d_colors, 4*sizeof(float)*width*height, cudaMemcpyDeviceToDevice);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &s_dCudaColors, 0)); // Unmap DX texture

	render(pd3dDeviceContext, s_pTmpTexturePositionsSRV, s_pTmpTextureNormalsSRV, s_pTmpTextureColorsSRV, useMaterial, width, height, overlayColor);
}

void DX11PhongLighting::render(ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* positions, ID3D11ShaderResourceView* normals, ID3D11ShaderResourceView* colors, bool useMaterial, unsigned int width, unsigned int height, const vec3f& overlayColor)
{
	// save render targets
	ID3D11RenderTargetView* rtv = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* dsv = DXUTGetD3D11DepthStencilView();

	// setup new render targets
	static float ClearColor[4] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
	pd3dDeviceContext->ClearRenderTargetView(s_pColorsRTV,	ClearColor);
	pd3dDeviceContext->ClearDepthStencilView(s_pDepthStencilDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	pd3dDeviceContext->RSSetState(s_pRastStateDefault);
	pd3dDeviceContext->OMSetRenderTargets(1, &s_pColorsRTV, s_pDepthStencilDSV);
	pd3dDeviceContext->OMSetDepthStencilState(s_pDepthStencilStateDefault, 0);

	// Initialize Constant Buffers
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	HRESULT hr = pd3dDeviceContext->Map(s_ConstantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if(FAILED(hr)) return;
	cbConstant *cbufferConstant = (cbConstant*)mappedResource.pData;
	cbufferConstant->useMaterial = useMaterial ? 1 : 0;
	cbufferConstant->overlayColor = D3DXVECTOR3(overlayColor.x, overlayColor.y, overlayColor.z);
	pd3dDeviceContext->Unmap(s_ConstantBuffer, 0);

	// copy lightbuffer to gpu
	s_ConstantBufferLightCPU.CopyToDX11Buffer(pd3dDeviceContext,s_ConstantBufferLight);

	pd3dDeviceContext->PSSetConstantBuffers(0, 1, &s_ConstantBuffer);
	pd3dDeviceContext->PSSetConstantBuffers(1, 1, &s_ConstantBufferLight);
						
	ID3D11ShaderResourceView* srvs[] = {positions, normals, colors};
	DX11QuadDrawer::RenderQuad(pd3dDeviceContext, s_PixelShaderPhong, srvs, 3);

	ID3D11Buffer* nullCB[] = { NULL };
	pd3dDeviceContext->PSSetConstantBuffers(0, 1, nullCB);
	pd3dDeviceContext->PSSetConstantBuffers(1, 1, nullCB);

	// reset render targets
	pd3dDeviceContext->OMSetRenderTargets(1, &rtv, dsv);
	pd3dDeviceContext->OMSetDepthStencilState(0, 0);
	pd3dDeviceContext->RSSetState(0);
}

void DX11PhongLighting::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(s_PixelShaderPhong);
	SAFE_RELEASE(s_ConstantBuffer);
	SAFE_RELEASE(s_ConstantBufferLight);

	SAFE_RELEASE(s_pDepthStencil);
	SAFE_RELEASE(s_pDepthStencilDSV);
	SAFE_RELEASE(s_pDepthStencilSRV);

	SAFE_RELEASE(s_pColors);
	SAFE_RELEASE(s_pColorsRTV);
	SAFE_RELEASE(s_pColorsSRV);

	SAFE_RELEASE(s_pRastStateDefault);
	SAFE_RELEASE(s_pDepthStencilStateDefault);

	///////////////////////////////////////////////////////////////////////////////////////////////
	// CUDA interop
	///////////////////////////////////////////////////////////////////////////////////////////////
			
	// Positions
	SAFE_RELEASE(s_pTmpTexturePositions);
	SAFE_RELEASE(s_pTmpTexturePositionsSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCudaPositions));

	// Normals
	SAFE_RELEASE(s_pTmpTextureNormals);
	SAFE_RELEASE(s_pTmpTextureNormalsSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCudaNormals));

	// Colors
	SAFE_RELEASE(s_pTmpTextureColors);
	SAFE_RELEASE(s_pTmpTextureColorsSRV);
	cutilSafeCall(cudaGraphicsUnregisterResource(s_dCudaColors));

	s_width = 0;
	s_height = 0;
}

ID3D11Buffer* DX11PhongLighting::GetLightBuffer()
{
	return s_ConstantBufferLight;
}

DX11PhongLighting::ConstantBufferLight& DX11PhongLighting::GetLightBufferCPU()
{
	return s_ConstantBufferLightCPU;
}

ID3D11ShaderResourceView* DX11PhongLighting::GetDepthStencilSRV()
{
	return s_pDepthStencilSRV;
}

ID3D11ShaderResourceView* DX11PhongLighting::GetColorsSRV()
{
	return s_pColorsSRV;
}
