
#include "stdafx.h"

#include "DX11Utils.h"


#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <shlwapi.h>
#include "SDKmisc.h"

void AddDefinitionToMacro( D3D_SHADER_MACRO source[9], D3D_SHADER_MACRO dest[10], D3D10_SHADER_MACRO &addMacro )
{
	for (UINT j = 0;; j++) {
		assert(j < 9);
		if (source[j].Name == 0)	{
			dest[j] = addMacro;
			dest[j+1] = source[j];
			break;
		}
		dest[j] = source[j];
	}

}

double GetTime()
{
	unsigned __int64 pf;
	QueryPerformanceFrequency( (LARGE_INTEGER *)&pf );
	double freq_ = 1.0 / (double)pf;

	unsigned __int64 val;
	QueryPerformanceCounter( (LARGE_INTEGER *)&val );
	return (val) * freq_;

}


bool OpenFileDialog(OPENFILENAME& ofn)
{
	ofn.lStructSize = sizeof(ofn);	
	ofn.hwndOwner = DXUTGetHWND();
	ofn.hInstance = GetModuleHandle(NULL);	
	ofn.Flags |= OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR; 	
	static WCHAR lastDir[MAX_PATH] = L"";
	ofn.lpstrInitialDir = lastDir;
	if(lastDir[0] == L'\0') {
		WCHAR baseDir[MAX_PATH] = L"";
		GetCurrentDirectory(MAX_PATH, baseDir); 
		PathCombine(lastDir, baseDir, L"Media");
	}

	BOOL isWindowed = DXUTIsWindowed();
	WINDOWPLACEMENT oldPlacement;
	if(isWindowed==FALSE)
	{
		DXUTToggleFullScreen();

		ZeroMemory(&oldPlacement, sizeof(oldPlacement));
		oldPlacement.length = sizeof(oldPlacement);
		GetWindowPlacement(DXUTGetHWND(), &oldPlacement);

		WINDOWPLACEMENT tmpPlacement = oldPlacement;
		tmpPlacement.showCmd = SW_SHOWMAXIMIZED;

		SetWindowPlacement(DXUTGetHWND(), &tmpPlacement);
	}	
	BOOL result = GetOpenFileName(&ofn);	
	if(isWindowed==FALSE)
	{
		SetWindowPlacement(DXUTGetHWND(), &oldPlacement);
		DXUTToggleFullScreen();
	}	
	if(result != FALSE)
	{
		for (UINT i = 0; i < MAX_PATH; i++) {
			lastDir[i] = ofn.lpstrFile[i];
			if (ofn.lpstrFile[i] == '\0')	break;
		}
		PathRemoveFileSpec(lastDir);
	}


	return result != FALSE;
}

static const bool s_bUsePreCompiledShaders = true;
static bool b_CompiledShaderDirectoryCreated = false;

HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut, const D3D_SHADER_MACRO* pDefines, DWORD pCompilerFlags)
{
	HRESULT hr = S_OK;

	std::wstring compiledShaderDirectory(L"CompiledShaders/");
	if (!b_CompiledShaderDirectoryCreated) {
		CreateDirectory(compiledShaderDirectory.c_str(), NULL);
		b_CompiledShaderDirectoryCreated = true;
	}
	std::wstring compiledFilename(compiledShaderDirectory);

	std::wstring fileName(szFileName);
	unsigned int found = (unsigned int)fileName.find_last_of(L"/\\");
	fileName = fileName.substr(found+1);
	compiledFilename.append(fileName);


	compiledFilename.push_back('.');
	std::string entryPoint(szEntryPoint);
	unsigned int oldLen = (unsigned int)compiledFilename.length();
	compiledFilename.resize(entryPoint.length() + oldLen);
	std::copy(entryPoint.begin(), entryPoint.end(), compiledFilename.begin()+oldLen);

	if (pDefines) {
		compiledFilename.push_back('.');
		for (unsigned int i = 0; pDefines[i].Name != NULL; i++) {
			std::string name(pDefines[i].Name);
			if (name[0] == '\"')				name[0] = 'x';
			if (name[name.length()-1] == '\"')	name[name.length()-1] = 'x';
			std::string def(pDefines[i].Definition);
			if (def[0] == '\"')				def[0] = 'x';
			if (def[def.length()-1] == '\"')	def[def.length()-1] = 'x';
			oldLen = (unsigned int)compiledFilename.length();
			compiledFilename.resize(oldLen + name.length() + def.length());
			std::copy(name.begin(), name.end(), compiledFilename.begin()+oldLen);
			std::copy(def.begin(), def.end(), compiledFilename.begin()+name.length()+oldLen);
		}
	}
	if (pCompilerFlags) {
		compiledFilename.push_back('.');
		std::wstringstream ss;
		ss << pCompilerFlags;
		std::wstring cf;
		ss >> cf;
		compiledFilename.append(cf);
	}
	compiledFilename.push_back('.');	
	compiledFilename.push_back('p');



	HANDLE hFindShader;
	HANDLE hFindCompiled;
	WIN32_FIND_DATA findData_shader;
	WIN32_FIND_DATA findData_compiled;
	hFindShader = FindFirstFile(szFileName, &findData_shader);
	hFindCompiled = FindFirstFile(compiledFilename.c_str(), &findData_compiled);
	if (!s_bUsePreCompiledShaders || hFindCompiled == INVALID_HANDLE_VALUE || CompareFileTime(&findData_shader.ftLastWriteTime, &findData_compiled.ftLastWriteTime) > 0) {

		// find the file
		WCHAR str[MAX_PATH];
		V_RETURN( DXUTFindDXSDKMediaFileCch( str, MAX_PATH, szFileName ) );

		DWORD dwShaderFlags = 0;
		dwShaderFlags |= D3DCOMPILE_ENABLE_STRICTNESS;
		//dwShaderFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
		//dwShaderFlags |= D3DCOMPILE_PARTIAL_PRECISION;
#if defined( DEBUG ) || defined( _DEBUG )
		// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
		// Setting this flag improves the shader debugging experience, but still allows 
		// the shaders to be optimized and to run exactly the way they will run in 
		// the release configuration of this program.
		dwShaderFlags |= D3DCOMPILE_DEBUG;
		//dwShaderFlags |= D3DXSHADER_SKIPOPTIMIZATION;
		//dwShaderFlags |= D3DCOMPILE_WARNINGS_ARE_ERRORS;
		//dwShaderFlags |= D3DCOMPILE_SKIP_OPTIMIZATION;
		//dwShaderFlags |= D3DCOMPILE_IEEE_STRICTNESS;
#endif
		dwShaderFlags |= pCompilerFlags;

		ID3DBlob* pErrorBlob;
		hr = D3DX11CompileFromFile( str, pDefines, NULL, szEntryPoint, szShaderModel, 
			dwShaderFlags, 0, NULL, ppBlobOut, &pErrorBlob, NULL );
		if( FAILED(hr) )
		{
			if( pErrorBlob != NULL )
				OutputDebugStringA( (char*)pErrorBlob->GetBufferPointer() );
			SAFE_RELEASE( pErrorBlob );
			return hr;
		}
		SAFE_RELEASE( pErrorBlob );

		std::ofstream compiledFile(compiledFilename.c_str(), std::ios::out | std::ios::binary);
		compiledFile.write((char*)(*ppBlobOut)->GetBufferPointer(), (*ppBlobOut)->GetBufferSize());
		compiledFile.close();
	} else {
		std::ifstream compiledFile(compiledFilename.c_str(), std::ios::in | std::ios::binary);
		assert(compiledFile.is_open());
		unsigned int size_data = findData_compiled.nFileSizeLow;

		V_RETURN(D3DCreateBlob(size_data,ppBlobOut));
		compiledFile.read((char*)(*ppBlobOut)->GetBufferPointer(), size_data);
		compiledFile.close();

	}

	return hr;
}



void* CreateAndCopyToDebugBuf( ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer, bool returnCPUMemory )
{
	ID3D11Buffer* debugbuf = NULL;

	D3D11_BUFFER_DESC desc;
	ZeroMemory( &desc, sizeof(desc) );
	pBuffer->GetDesc( &desc );
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.MiscFlags = 0;
	pDevice->CreateBuffer(&desc, NULL, &debugbuf);

	pd3dImmediateContext->CopyResource( debugbuf, pBuffer );


	INT *cpuMemory = new INT[desc.ByteWidth/sizeof(UINT)];
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	pd3dImmediateContext->Map(debugbuf, D3D11CalcSubresource(0,0,0), D3D11_MAP_READ, 0, &mappedResource);	
	memcpy((void*)cpuMemory, (void*)mappedResource.pData, desc.ByteWidth);
	pd3dImmediateContext->Unmap( debugbuf, 0 );

	for(unsigned int i = 0; i<desc.ByteWidth/sizeof(UINT); i++)
	{
		if(cpuMemory[i] != 0)
		{
			//std::cout << "test" << std::endl;
		}
	}

	SAFE_RELEASE(debugbuf);
	if (!returnCPUMemory) {
		SAFE_DELETE_ARRAY(cpuMemory);
		return NULL;
	} else {
		return (void*)cpuMemory;
	}
}


void* CreateAndCopyToDebugTexture2D( ID3D11Device* pDevice, ID3D11DeviceContext* pd3dImmediateContext, ID3D11Texture2D* pBufferTex, bool returnCPUMemory )
{
	ID3D11Texture2D* debugtex = NULL;

	D3D11_TEXTURE2D_DESC desc;
	ZeroMemory( &desc, sizeof(desc) );
	pBufferTex->GetDesc( &desc );
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.MiscFlags = 0;
	pDevice->CreateTexture2D(&desc, NULL, &debugtex);

	pd3dImmediateContext->CopyResource( debugtex, pBufferTex );


	FLOAT *cpuMemory = new FLOAT[desc.Height * desc.Width];
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	pd3dImmediateContext->Map(debugtex, D3D11CalcSubresource(0,0,0), D3D11_MAP_READ, 0, &mappedResource);	
	memcpy((void*)cpuMemory, (void*)mappedResource.pData, desc.Height * desc.Width * sizeof(FLOAT));
	pd3dImmediateContext->Unmap( debugtex, 0 );


	SAFE_RELEASE(debugtex);
	if (!returnCPUMemory) {
		SAFE_DELETE_ARRAY(cpuMemory);
		return NULL;
	} else {
		return (void*)cpuMemory;
	}

}




void* GetResourceData(ID3D11Resource* pResource, ResourceDesc& rdesc )
{
	ID3D11Device *pDevice = DXUTGetD3D11Device();
	ID3D11DeviceContext *pd3dImmediateContext = DXUTGetD3D11DeviceContext();

	D3D11_RESOURCE_DIMENSION rType;
	pResource->GetType(&rType);
	
	assert(rType == D3D11_RESOURCE_DIMENSION_TEXTURE2D);	//don't support any other types yet - tbd


	ID3D11Texture2D* debugtex = NULL;

	if (rType ==  D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
		
		ID3D11Texture2D* texture = (ID3D11Texture2D*)pResource;

		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory( &desc, sizeof(desc) );
		texture->GetDesc( &desc );
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;	
		desc.Usage = D3D11_USAGE_STAGING;
		desc.BindFlags = 0;
		desc.MiscFlags = 0;
		pDevice->CreateTexture2D(&desc, NULL, &debugtex);

		rdesc.m_Dimension = 2;
		rdesc.m_Size.x = desc.Width;
		rdesc.m_Size.y = desc.Height;
		rdesc.m_Size.z = 1;
		rdesc.m_OriginalFormat = desc.Format;

		if (desc.Format == DXGI_FORMAT_R32_FLOAT ||
			desc.Format == DXGI_FORMAT_R32_SINT ||
			desc.Format == DXGI_FORMAT_R32_UINT) {
			rdesc.m_ElementsPerEntry = 1;
			rdesc.m_ElementSizeInBytes = 4;
		} else if (
			desc.Format == DXGI_FORMAT_R8G8B8A8_UINT ||
			desc.Format == DXGI_FORMAT_R8G8B8A8_SINT ||
			desc.Format == DXGI_FORMAT_R8G8B8A8_SNORM ||
			desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
			rdesc.m_ElementsPerEntry = 4;
			rdesc.m_ElementSizeInBytes = 1;
		}	else {
			assert(false);
		}
	}
	
	pd3dImmediateContext->CopyResource( debugtex, pResource );

	void *cpuMemory = new CHAR[rdesc.m_Size.x * rdesc.m_Size.y * rdesc.m_Size.z * rdesc.m_ElementsPerEntry * rdesc.m_ElementSizeInBytes];
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	pd3dImmediateContext->Map(debugtex, D3D11CalcSubresource(0,0,0), D3D11_MAP_READ, 0, &mappedResource);
	memcpy((void*)cpuMemory, (void*)mappedResource.pData, rdesc.m_Size.x * rdesc.m_Size.y * rdesc.m_Size.z * rdesc.m_ElementsPerEntry * rdesc.m_ElementSizeInBytes);
	pd3dImmediateContext->Unmap( debugtex, 0 );


	SAFE_RELEASE(debugtex);
	//SAFE_DELETE_ARRAY(cpuMemory);

	return cpuMemory;
}

void SetRSCulling (ID3D11RasterizerState** rs, D3D11_CULL_MODE cm) {
	ID3D11Device* pd3dDevice = DXUTGetD3D11Device();
	D3D11_RASTERIZER_DESC RasterDesc;
	ZeroMemory( &RasterDesc, sizeof(D3D11_RASTERIZER_DESC) );
	(*rs)->GetDesc(&RasterDesc);
	RasterDesc.CullMode = cm;
	SAFE_RELEASE(*rs);
	pd3dDevice->CreateRasterizerState(&RasterDesc, rs);
}

void SetRSDrawing(ID3D11RasterizerState** rs, D3D11_FILL_MODE fm) {
	ID3D11Device* pd3dDevice = DXUTGetD3D11Device();
	D3D11_RASTERIZER_DESC RasterDesc;
	ZeroMemory( &RasterDesc, sizeof(D3D11_RASTERIZER_DESC) );
	(*rs)->GetDesc(&RasterDesc);
	RasterDesc.FillMode = fm;
	SAFE_RELEASE(*rs);
	pd3dDevice->CreateRasterizerState(&RasterDesc, rs);
} 

/*

void SaveCamera( CModelViewerCamera &camera, const char* filename )
{
	std::ofstream file;
	file.open(filename);
	if (file.is_open()) {
		file.precision(15);
		file << camera.GetEyePt()->x << " " << camera.GetEyePt()->y << " " << camera.GetEyePt()->z << std::endl;
		file << camera.GetLookAtPt()->x << " " << camera.GetLookAtPt()->y << " " << camera.GetLookAtPt()->z << std::endl;

		file << camera.GetViewQuat().x << " " << camera.GetViewQuat().y << " " << camera.GetViewQuat().z << " " << camera.GetViewQuat().w << std::endl;
		file << camera.GetWorldQuat().x << " " << camera.GetWorldQuat().y << " " << camera.GetWorldQuat().z << " " << camera.GetWorldQuat().w << std::endl;
	}

	file.close();
	return;


}

void LoadCamera( CModelViewerCamera &camera, const char* filename )
{
	std::ifstream file;
	file.open(filename);
	if (file.is_open() && file.good()) {
		D3DXVECTOR3 eye, lookAt;
		D3DXQUATERNION world, view;

		file >> eye.x >> eye.y >> eye.z;
		file >> lookAt.x >> lookAt.y >> lookAt.z;

		file >> view.x >> view.y >> view.z >> view.w;
		file >> world.x >> world.y >> world.z >> world.w;

		camera.SetViewParams(&eye, &lookAt);
		camera.SetViewQuat(view);
		camera.SetWorldQuat(world);
	}

	file.close();
	return;

}

*/