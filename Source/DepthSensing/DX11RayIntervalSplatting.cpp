#include "stdafx.h"

#include "DX11Utils.h"

#include "DepthCameraUtil.h"
#include "VoxelUtilHashSDF.h"
#include "RayCastSDFUtil.h"
#include "TimingLogDepthSensing.h"

#include "DX11RayIntervalSplatting.h"

extern "C" void resetRayIntervalSplatCUDA(RayCastData& data, const RayCastParams& params);
extern "C" void rayIntervalSplatCUDA(const HashDataStruct& hashData, const RayCastData &rayCastData, const RayCastParams &rayCastParams);



HRESULT DX11RayIntervalSplatting::OnD3D11CreateDevice( ID3D11Device* pd3dDevice, unsigned int width, unsigned int height )
{
	HRESULT hr = S_OK;

	V_RETURN(initialize(pd3dDevice, width, height));

	return  hr;
}

void DX11RayIntervalSplatting::OnD3D11DestroyDevice()
{
	destroy();
}

HRESULT DX11RayIntervalSplatting::initialize( ID3D11Device* pd3dDevice, unsigned int width, unsigned int height )
{
	HRESULT hr = S_OK;

	ID3DBlob* pBlob = NULL;

	// Ray Interval
	V_RETURN(CompileShaderFromFile(L"Shaders/RayIntervalSplatting.hlsl", "VS", "vs_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pVertexShaderSplatting));
	//SAFE_RELEASE(pBlob);

	D3D11_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};

	UINT numElements = sizeof( layout ) / sizeof( layout[0] );
	V_RETURN(pd3dDevice->CreateInputLayout(layout, numElements, pBlob->GetBufferPointer(), pBlob->GetBufferSize(), &m_VertexLayout));
	SAFE_RELEASE(pBlob);

	//V_RETURN(CompileShaderFromFile(L"Shaders/RayIntervalSplatting.hlsl", "GS", "gs_5_0", &pBlob));
	//V_RETURN(pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pGeometryShaderSplatting));
	//SAFE_RELEASE(pBlob);

	V_RETURN(CompileShaderFromFile(L"Shaders/RayIntervalSplatting.hlsl", "PS", "ps_5_0", &pBlob));
	V_RETURN(pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &m_pPixelShaderSplatting));
	SAFE_RELEASE(pBlob);

	{
		// create vertex buffer, register with cuda
		unsigned int maxVertices = GlobalAppState::get().s_hashNumSDFBlocks * 6;
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory(&bufferDesc, sizeof(D3D11_BUFFER_DESC));
		bufferDesc.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc.ByteWidth = sizeof(float4) * maxVertices;
		bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER; //D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.CPUAccessFlags = 0;
		bufferDesc.MiscFlags = 0;

		//D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		//ZeroMemory( &srvDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC) );
		//srvDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
		//srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		//srvDesc.Buffer.FirstElement = 0;
		//srvDesc.Buffer.NumElements = maxVertices;

		V_RETURN(pd3dDevice->CreateBuffer(&bufferDesc, NULL, &m_pVertexBufferFloat4));
		//V_RETURN(pd3dDevice->CreateShaderResourceView(m_pVertexBufferFloat4, &srvDesc, &m_pVertexBufferFloat4SRV));

		cutilSafeCall(cudaGraphicsD3D11RegisterResource(&m_dCudaVertexBufferFloat4, m_pVertexBufferFloat4, cudaGraphicsRegisterFlagsNone));
		cutilSafeCall(cudaGraphicsResourceSetMapFlags(m_dCudaVertexBufferFloat4, cudaGraphicsMapFlagsWriteDiscard));
	}

	// Depth Stencil State
	D3D11_DEPTH_STENCIL_DESC stenDesc;
	ZeroMemory(&stenDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
	stenDesc.DepthEnable = true;
	stenDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	stenDesc.DepthFunc =  D3D11_COMPARISON_LESS;
	stenDesc.StencilEnable = false;
	V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &m_pDepthStencilStateSplattingMin));

	stenDesc.DepthFunc =  D3D11_COMPARISON_GREATER;
	V_RETURN(pd3dDevice->CreateDepthStencilState(&stenDesc, &m_pDepthStencilStateSplattingMax));

	// Rasterizer Stage
	D3D11_RASTERIZER_DESC rastDesc;
	ZeroMemory(&rastDesc, sizeof(D3D11_RASTERIZER_DESC));
	rastDesc.FillMode = D3D11_FILL_SOLID;
	rastDesc.CullMode = D3D11_CULL_NONE;
	rastDesc.FrontCounterClockwise = false;

	V_RETURN(pd3dDevice->CreateRasterizerState(&rastDesc, &m_pRastState));

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);

	V_RETURN(m_customRenderTargetMin.OnD3D11CreateDevice(pd3dDevice, width, height, formats));
	V_RETURN(m_customRenderTargetMax.OnD3D11CreateDevice(pd3dDevice, width, height, formats));

	return  hr;
}

void DX11RayIntervalSplatting::destroy()
{
	SAFE_RELEASE(m_pVertexShaderSplatting);
	//SAFE_RELEASE(m_pGeometryShaderSplatting);
	SAFE_RELEASE(m_pPixelShaderSplatting);

	SAFE_RELEASE(m_pDepthStencilStateSplattingMin);
	SAFE_RELEASE(m_pDepthStencilStateSplattingMax);

	SAFE_RELEASE(m_pRastState);

	SAFE_RELEASE(m_VertexLayout);

	SAFE_RELEASE(m_pVertexBufferFloat4);
	//SAFE_RELEASE(m_pVertexBufferFloat4SRV);
	if (m_dCudaVertexBufferFloat4)
	{
		cutilSafeCall(cudaGraphicsUnregisterResource(m_dCudaVertexBufferFloat4));
		m_dCudaVertexBufferFloat4 = NULL;
	}

	m_customRenderTargetMin.OnD3D11DestroyDevice();
	m_customRenderTargetMax.OnD3D11DestroyDevice();
}

HRESULT DX11RayIntervalSplatting::rayIntervalSplatting(ID3D11DeviceContext* context, const HashDataStruct& hashData,
													   RayCastData& rayCastData, RayCastParams& rayCastParams, unsigned int numVertices)
{
	HRESULT hr = S_OK;

	// Start query for timing
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		cutilSafeCall(cudaDeviceSynchronize());
		m_timer.start();
	}
	cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaVertexBufferFloat4, 0));	// Map DX texture to Cuda
	size_t num_bytes;
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&rayCastData.d_vertexBuffer, &num_bytes, m_dCudaVertexBufferFloat4));
	// minimum
	//resetRayIntervalSplatCUDA(rayCastData, rayCastParams);
	rayCastParams.m_splatMinimum = 1;
	rayCastData.updateParams(rayCastParams);
	rayIntervalSplatCUDA(hashData, rayCastData, rayCastParams);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaVertexBufferFloat4, 0));	// Unmap DX texture

	// Setup Pipeline
	context->OMSetDepthStencilState(m_pDepthStencilStateSplattingMin, 0);
	context->RSSetState(m_pRastState);

	unsigned int stride = sizeof(float4);
	unsigned int offset = 0;
	context->IASetVertexBuffers(0, 1, &m_pVertexBufferFloat4, &stride, &offset);		
	context->IASetInputLayout(m_VertexLayout);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	context->VSSetShader(m_pVertexShaderSplatting, 0, 0);

	//context->GSSetShaderResources(0, 1, &m_pVertexBufferFloat4SRV);

	//context->GSSetShader(m_pGeometryShaderSplatting, 0, 0);
	context->PSSetShader(m_pPixelShaderSplatting, 0, 0);

	/////////////////////////////////////////////////////////////////////////////
	// Splat minimum
	/////////////////////////////////////////////////////////////////////////////

	m_customRenderTargetMin.Clear(context);
	m_customRenderTargetMin.Bind(context);
	context->Draw(numVertices, 0);	
	m_customRenderTargetMin.Unbind(context);

	/////////////////////////////////////////////////////////////////////////////
	// Splat maximum
	/////////////////////////////////////////////////////////////////////////////

	cutilSafeCall(cudaGraphicsMapResources(1, &m_dCudaVertexBufferFloat4, 0));	// Map DX texture to Cuda
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&rayCastData.d_vertexBuffer, &num_bytes, m_dCudaVertexBufferFloat4));
	// maximum
	//resetRayIntervalSplatCUDA(rayCastData, rayCastParams);
	rayCastParams.m_splatMinimum = 0;
	rayCastData.updateParams(rayCastParams);
	rayIntervalSplatCUDA(hashData, rayCastData, rayCastParams);
	cutilSafeCall(cudaGraphicsUnmapResources(1, &m_dCudaVertexBufferFloat4, 0));	// Unmap DX texture

	context->OMSetDepthStencilState(m_pDepthStencilStateSplattingMax, 0);
	
	//context->GSSetShaderResources(0, 1, &m_pVertexBufferFloat4SRV);

	m_customRenderTargetMax.Clear(context, 0.f);
	m_customRenderTargetMax.Bind(context);
	context->Draw(numVertices, 0);
	m_customRenderTargetMax.Unbind(context);
	// Wait for query
	if(GlobalAppState::getInstance().s_timingsDetailledEnabled)
	{
		GlobalAppState::get().WaitForGPU(); 
		m_timer.stop();
		TimingLogDepthSensing::totalTimeRayIntervalSplatting+=m_timer.getElapsedTimeMS();
		TimingLogDepthSensing::countTimeRayIntervalSplatting++;
	}
	
	// Reset Pipeline

	context->RSSetState(0);
	context->OMSetDepthStencilState(m_pDepthStencilStateSplattingMin, 0); // Min is also default state

	//ID3D11ShaderResourceView* nullSRV[] = { NULL };

	//context->GSSetShaderResources(0, 1, nullSRV);

	context->VSSetShader(0, 0, 0);
	//context->GSSetShader(0, 0, 0);
	context->PSSetShader(0, 0, 0);

	return hr;
}

DX11RayIntervalSplatting::DX11RayIntervalSplatting()
{
	// Ray Interval
	m_pVertexShaderSplatting = NULL;
	//m_pGeometryShaderSplatting = NULL;
	m_pPixelShaderSplatting = NULL;

	m_pDepthStencilStateSplattingMin = NULL;
	m_pDepthStencilStateSplattingMax = NULL;

	m_pRastState = NULL;

	m_pVertexBufferFloat4 = NULL;
	//m_pVertexBufferFloat4SRV = NULL;
	m_dCudaVertexBufferFloat4 = NULL;

	m_VertexLayout = NULL;
}

DX11RayIntervalSplatting::~DX11RayIntervalSplatting()
{
}
