#pragma once

#include "DX11QuadDrawer.h"
#include "DX11Utils.h"
#include "GlobalAppState.h"

class DX11PhongLighting
{
	public:
		
		struct ConstantBufferLight : ConstantBufferBase<ConstantBufferLight>
		{
			D3DXVECTOR4 lightAmbient;
			D3DXVECTOR4 lightDiffuse;
			D3DXVECTOR4 lightSpecular;

			D3DXVECTOR3 lightDirection;
			float materialShininess;

			D3DXVECTOR4 materialAmbient;
			D3DXVECTOR4 materialSpecular;
			D3DXVECTOR4 materialDiffuse;

			void SetDefault() {
				// default light settings
				lightAmbient		= D3DXVECTOR4(GlobalAppState::get().s_lightAmbient.getData());
				lightDiffuse		= D3DXVECTOR4(GlobalAppState::get().s_lightDiffuse.getData());
				lightSpecular		= D3DXVECTOR4(GlobalAppState::get().s_lightSpecular.getData());
				lightDirection		= D3DXVECTOR3(GlobalAppState::get().s_lightDirection.getData());


				materialAmbient   = D3DXVECTOR4(GlobalAppState::get().s_materialAmbient.getData());
				materialSpecular  = D3DXVECTOR4(GlobalAppState::get().s_materialSpecular.getData());
				materialDiffuse	  = D3DXVECTOR4(GlobalAppState::get().s_materialDiffuse.getData());
				materialShininess = GlobalAppState::get().s_materialShininess;
			}
		};

		static HRESULT OnD3D11CreateDevice(ID3D11Device* pd3dDevice, unsigned int width = 0, unsigned int height = 0);

		static void render(ID3D11DeviceContext* pd3dDeviceContext, float4* d_positions, float4* d_normals, float4* d_colors, bool useMaterial, unsigned int width, unsigned int height, const vec3f& overlayColor = vec3f(0.0f, 0.0f, 0.0f));
		static void render(ID3D11DeviceContext* pd3dDeviceContext, ID3D11ShaderResourceView* positions, ID3D11ShaderResourceView* normals, ID3D11ShaderResourceView* colors, bool useMaterial, unsigned int width, unsigned int height, const vec3f& overlayColor = vec3f(0.0f, 0.0f, 0.0f));

		static void OnD3D11DestroyDevice();

		static HRESULT OnResize(ID3D11Device* pd3dDevice, UINT width, UINT height) {
			HRESULT hr = S_OK;
			if (s_width != width || s_height != height) {
				OnD3D11DestroyDevice();
				V_RETURN(OnD3D11CreateDevice(pd3dDevice, width, height));
			}
			return hr;
		}

		static ID3D11Buffer* GetLightBuffer();
		static ConstantBufferLight& GetLightBufferCPU();
		static ID3D11ShaderResourceView* GetDepthStencilSRV();
		static ID3D11ShaderResourceView* GetColorsSRV();

		static unsigned int getWidth() {
			return s_width;
		}
		static unsigned int getHeight() {
			return s_height;
		}
private:
	static unsigned int s_width;
	static unsigned int s_height;


	struct cbConstant
	{
		unsigned int useMaterial;
		D3DXVECTOR3 overlayColor;
	};

	static ID3D11Buffer* s_ConstantBuffer;

	static ConstantBufferLight s_ConstantBufferLightCPU;
	static ID3D11Buffer* s_ConstantBufferLight;

	static ID3D11PixelShader* s_PixelShaderPhong;

	static ID3D11Texture2D*				s_pDepthStencil;
	static ID3D11DepthStencilView*		s_pDepthStencilDSV;
	static ID3D11ShaderResourceView*	s_pDepthStencilSRV;

	static ID3D11Texture2D*				s_pColors;
	static ID3D11RenderTargetView*		s_pColorsRTV;
	static ID3D11ShaderResourceView*	s_pColorsSRV;

	static ID3D11RasterizerState*		s_pRastStateDefault;
	static ID3D11DepthStencilState*		s_pDepthStencilStateDefault;

	// Cuda interop
	static ID3D11Texture2D*				s_pTmpTexturePositions;
	static ID3D11ShaderResourceView*	s_pTmpTexturePositionsSRV;
	static cudaGraphicsResource*		s_dCudaPositions;

	static ID3D11Texture2D*				s_pTmpTextureNormals;
	static ID3D11ShaderResourceView*	s_pTmpTextureNormalsSRV;
	static cudaGraphicsResource*		s_dCudaNormals;

	static ID3D11Texture2D*				s_pTmpTextureColors;
	static ID3D11ShaderResourceView*	s_pTmpTextureColorsSRV;
	static cudaGraphicsResource*		s_dCudaColors;
};
