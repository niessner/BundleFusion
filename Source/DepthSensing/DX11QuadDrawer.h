#pragma once

#include "DX11Utils.h"
#include <cuda_runtime.h> 
#include <cstdlib>
#include "cudaUtil.h"
#include <cuda_d3d11_interop.h> 

struct SimpleVertex {
	D3DXVECTOR3 pos;
	D3DXVECTOR2 pex;
};

struct CB_QUAD {
	D3DXMATRIX mWorldViewProjection;
	UINT width;
	UINT height;
	D3DXVECTOR2 dummy;
};

class DX11QuadDrawer
{
public:
	DX11QuadDrawer(void);
	~DX11QuadDrawer(void);

	static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice);
	static void OnD3D11DestroyDevice();

	static HRESULT RenderQuadDynamicDEPTHasHSV(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const float* d_data, float minDepth, float maxDepth, unsigned int width, unsigned int height, float scale = 1.0f, D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL);
	static HRESULT RenderQuadDynamicUCHAR4(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, const uchar4* d_data, unsigned int width, unsigned int height, float scale = 1.0f, D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL);
	static HRESULT RenderQuadDynamic(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dDeviceContext, float* d_data, unsigned int nChannels, unsigned int width, unsigned int height, float scale = 1.0f, D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL);
	
	static void RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, float* d_data, unsigned int nChannels, unsigned int width, unsigned int height, float scale = 1.0f , D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL );
	static void RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, ID3D11PixelShader* pixelShader, ID3D11ShaderResourceView** srvs, UINT numShaderResourceViews, D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f));
	static void RenderQuad(ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* srv, float scale = 1.0f, D3DXVECTOR2 Pow2Ratios = D3DXVECTOR2(1.0f, 1.0f), ID3D11PixelShader* pixelShader = NULL);

private:

	static void RenderQuadHelper(ID3D11DeviceContext* pd3dDeviceContext, float* d_data, cudaGraphicsResource* pCuda, ID3D11ShaderResourceView* pTmpTextureSRV, unsigned int size, float scale, D3DXVECTOR2 Pow2Ratios , ID3D11PixelShader* pixelShader);
	
	static ID3D11InputLayout*	s_VertexLayout;
	static ID3D11Buffer*		s_VertexBuffer;
	static ID3D11Buffer*		s_IndexBuffer;

	static ID3D11VertexShader*	s_VertexShader;
	static ID3D11PixelShader*	s_PixelShaderFloat;
	static ID3D11PixelShader*	s_PixelShaderRGBA;
	static ID3D11PixelShader*	s_PixelShader3;

	static ID3D11Buffer*		s_CBquad;

	static ID3D11SamplerState*	s_PointSampler;

	static ID3D11Buffer*		s_pcbVSPowTwoRatios;

	// For CUDA<->DX interop
	static ID3D11Texture2D*				s_pTmpTextureFloat;
	static ID3D11ShaderResourceView*	s_pTmpTextureFloatSRV;
	static cudaGraphicsResource*		s_dCudaFloat;
	
	static ID3D11Texture2D*				s_pTmpTexture2Float;
	static ID3D11ShaderResourceView*	s_pTmpTexture2FloatSRV;
	static cudaGraphicsResource*		s_dCuda2Float;

	static ID3D11Texture2D*				s_pTmpTexture;
	static ID3D11ShaderResourceView*	s_pTmpTextureSRV;
	static cudaGraphicsResource*		s_dCuda;
	
	static ID3D11Texture2D*				s_pTmpTexture2;
	static ID3D11ShaderResourceView*	s_pTmpTexture2SRV;
	static cudaGraphicsResource*		s_dCuda2;

	static unsigned int					s_tmpTextureWidth;
	static unsigned int					s_tmpTextureHeight;

	static unsigned int					s_tmpTextureWidth2;
	static unsigned int					s_tmpTextureHeight2;
};
