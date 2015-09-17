#pragma once

#ifndef _DX11_UTILS_H
#define _DX11_UTILS_H

#include "DXUT.h"
#include <iostream>

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=NULL; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=NULL; } }
#endif

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif

namespace DX
{
	typedef struct uint4 {
		uint4() {x = 0; y = 0; z = 0; w = 0;}
		uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) {this->x = x; this->y = y; this->z = z; this->w = w;}
		unsigned int x, y, z, w;
	} uint4;

	typedef struct uint3 {
		uint3() {x = 0; y = 0; z = 0;}
		uint3(unsigned int x, unsigned int y, unsigned int z) {this->x = x; this->y = y; this->z = z;}
		UINT x,y,z;
	} uint3;

	typedef struct uint2 {
		uint2() {x = 0; y = 0;}
		uint2(unsigned int x, unsigned int y) {this->x = x; this->y = y;}
		UINT x,y;
	} uint2;

	typedef struct int4 {
		int4() {x = 0; y = 0; z = 0; w = 0;}
		int4(int x, int y, int z,int w) {this->x = x; this->y = y; this->z = z; this->w = w;}
		int x, y, z, w;
	} int4;

	typedef struct int3 {
		int3() {x = 0; y = 0; z = 0;}
		int3(int x, int y, int z) {this->x = x; this->y = y; this->z = z;}
		int x,y,z;
	} int3;

	typedef struct int2 {
		int2() {x = 0; y = 0;}
		int2(int x, int y) {this->x = x; this->y = y;}
		int x,y;
	} int2;
};

struct ResourceDesc
{
	UINT m_Dimension;
	DX::uint3 m_Size;

	UINT m_ElementsPerEntry;
	UINT m_ElementSizeInBytes;
	DXGI_FORMAT m_OriginalFormat;
};

void* GetResourceData(ID3D11Resource* pResource, ResourceDesc& rdesc );

template<typename CB> //to get the real size of the derived constant buffer (sizeof(CB))
struct ConstantBufferBase {
	// create a new constant buffer
	static HRESULT CreateDX11Buffer(ID3D11Device* pd3dDevice, ID3D11Buffer** constantBuffer)
	{
		HRESULT hr;
		D3D11_BUFFER_DESC constantBufferDesc;
		constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		constantBufferDesc.ByteWidth = sizeof(CB);
		constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		constantBufferDesc.MiscFlags = 0;
		constantBufferDesc.StructureByteStride = 0;

		V_RETURN(pd3dDevice->CreateBuffer( &constantBufferDesc, NULL, constantBuffer ));
		return hr;
	}

	// create a new constant buffer and fill it with the data
	HRESULT CreateDX11BufferAndCopy(ID3D11Device* pd3dDevice, ID3D11Buffer** constantBuffer) const
	{
		HRESULT hr;
		D3D11_BUFFER_DESC constantBufferDesc;
		constantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		constantBufferDesc.ByteWidth = sizeof(CB);
		constantBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
		constantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		constantBufferDesc.MiscFlags = 0;
		constantBufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA constantBufferInitData;
		constantBufferInitData.pSysMem = this;
		V_RETURN(pd3dDevice->CreateBuffer( &constantBufferDesc, &constantBufferInitData, constantBuffer ));
		return hr;
	}

	// copy data to constant buffer
	HRESULT CopyToDX11Buffer(ID3D11DeviceContext* context, ID3D11Buffer* constantBuffer) const
	{
		HRESULT hr;
		D3D11_MAPPED_SUBRESOURCE mappedResource;
		V_RETURN(context->Map( constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource ));
		memcpy( mappedResource.pData, this, sizeof(CB) );
		context->Unmap(constantBuffer, 0 );
		return hr;
	}
};

HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut, const D3D_SHADER_MACRO* pDefines = NULL, DWORD pCompilerFlags = 0);

double GetTime();
bool OpenFileDialog(OPENFILENAME& ofn);
void* CreateAndCopyToDebugBuf( ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer, bool returnCPUMemory = false );
void* CreateAndCopyToDebugTexture2D( ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Texture2D* pBufferTex, bool returnCPUMemory = false );
void AddDefinitionToMacro( D3D_SHADER_MACRO source[9], D3D_SHADER_MACRO dest[10], D3D_SHADER_MACRO &addMacro );
void SetRSCulling (ID3D11RasterizerState** rs, D3D11_CULL_MODE cm);
void SetRSDrawing(ID3D11RasterizerState** rs, D3D11_FILL_MODE fm);

#endif
