#include "stdafx.h"

#include "GlobalAppState.h"


HRESULT GlobalAppState::OnD3D11CreateDevice(ID3D11Device* pd3dDevice)
{
	HRESULT hr = S_OK;

	/////////////////////////////////////////////////////
	// Query
	/////////////////////////////////////////////////////

	D3D11_QUERY_DESC queryDesc;
	queryDesc.Query = D3D11_QUERY_EVENT;
	queryDesc.MiscFlags = 0;

	hr = pd3dDevice->CreateQuery(&queryDesc, &m_pQuery);
	if(FAILED(hr)) return hr;

	return  hr;
}

void GlobalAppState::OnD3D11DestroyDevice()
{
	SAFE_RELEASE(m_pQuery);
}

void GlobalAppState::WaitForGPU()
{
	DXUTGetD3D11DeviceContext()->Flush();
	DXUTGetD3D11DeviceContext()->End(m_pQuery);
	DXUTGetD3D11DeviceContext()->Flush();

	while (S_OK != DXUTGetD3D11DeviceContext()->GetData(m_pQuery, NULL, 0, 0));
}