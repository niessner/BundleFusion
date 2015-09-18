
#include "stdafx.h"

#ifdef USE_DEPTHSENSING

#include "DepthSensing.h"



#include <windows.h>
#include <d3d11.h>
#include <xnamath.h>
#include "DX11Utils.h"

#include "GlobalAppState.h"
#include "TimingLogDepthSensing.h"
#include "StdOutputLogger.h"
#include "Util.h"


#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"

#include "DX11RGBDRenderer.h"
#include "DX11QuadDrawer.h"
#include "DX11CustomRenderTarget.h"
#include "DX11PhongLighting.h"

#include "CUDASceneRepHashSDF.h"
#include "CUDARayCastSDF.h"
#include "CUDAMarchingCubesHashSDF.h"
#include "CUDAHistogramHashSDF.h"
#include "CUDASceneRepChunkGrid.h"
#include "CUDAImageManager.h"

#include <iomanip>


//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN      1
#define IDC_TOGGLEREF             3
#define IDC_CHANGEDEVICE          4
#define IDC_TEST                  5



//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------

bool CALLBACK		ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void* pUserContext);
void CALLBACK		OnFrameMove(double fTime, float fElapsedTime, void* pUserContext);
LRESULT CALLBACK	MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext);
void CALLBACK		OnGUIEvent(UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext);
void CALLBACK		OnKeyboard(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext);
bool CALLBACK		IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext);
HRESULT CALLBACK	OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
HRESULT CALLBACK	OnD3D11ResizedSwapChain(ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext);
void CALLBACK		OnD3D11ReleasingSwapChain(void* pUserContext);
void CALLBACK		OnD3D11DestroyDevice(void* pUserContext);
void CALLBACK		OnD3D11FrameRender(ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext);


void RenderText();
void RenderHelp();


//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

CDXUTDialogResourceManager	g_DialogResourceManager; // manager for shared resources of dialogs
CDXUTTextHelper*            g_pTxtHelper = NULL;
bool						g_renderText = true;
bool						g_bRenderHelp = true;

CModelViewerCamera          g_Camera;               // A model viewing camera
DX11RGBDRenderer			g_RGBDRenderer;
DX11CustomRenderTarget		g_CustomRenderTarget;

CUDASceneRepHashSDF*		g_sceneRep			= NULL;
CUDARayCastSDF*				g_rayCast			= NULL;
CUDAMarchingCubesHashSDF*	g_marchingCubesHashSDF = NULL;
CUDAHistrogramHashSDF*		g_historgram = NULL;
CUDASceneRepChunkGrid*		g_chunkGrid = NULL;


CUDAImageManager*			g_CudaImageManager = NULL;
RGBDSensor*					g_RGBDSensor = NULL;




RGBDSensor* getRGBDSensor() {
	return g_RGBDSensor;
}

void ResetDepthSensing();
void StopScanningAndExtractIsoSurfaceMC(const std::string& filename = "./scans/scan.ply");



int startDepthSensing(Bundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager)
{
	g_RGBDSensor = sensor;
	g_CudaImageManager = imageManager;

	// Set DXUT callbacks
	DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
	DXUTSetCallbackMsgProc(MsgProc);
	DXUTSetCallbackKeyboard(OnKeyboard);
	DXUTSetCallbackFrameMove(OnFrameMove);

	DXUTSetCallbackD3D11DeviceAcceptable(IsD3D11DeviceAcceptable);
	DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
	DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
	DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);
	DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
	DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);

	DXUTInit(true, true); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
	DXUTSetCursorSettings(true, true); // Show the cursor and clip it when in full screen
	DXUTCreateWindow(GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight, L"VoxelHashing", false);

	DXUTSetIsInGammaCorrectMode(false);	//gamma fix (for kinect color)

	DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight);
	DXUTMainLoop(); // Enter into the DXUT render loop


	return DXUTGetExitCode();
}

//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
	// For the first device created if its a REF device, optionally display a warning dialog box
	static bool s_bFirstTime = true;
	if( s_bFirstTime )
	{
		s_bFirstTime = false;
		if( ( DXUT_D3D9_DEVICE == pDeviceSettings->ver && pDeviceSettings->d3d9.DeviceType == D3DDEVTYPE_REF ) ||
			( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
			pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
		{
			DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
		}
	}

	return true;
}

//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
	g_Camera.FrameMove( fElapsedTime );
	// Update the camera's position based on user input 
}

//--------------------------------------------------------------------------------------
// Render the statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos( 2, 0 );
	g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 0.0f, 1.0f ) );
	g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
	g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
	if (!g_bRenderHelp) {
		g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 1.0f, 1.0f, 1.0f ) );
		g_pTxtHelper->DrawTextLine(L"\tPress F1 for help");
	}
	g_pTxtHelper->End();


	if (g_bRenderHelp) {
		RenderHelp();
	}
}

void RenderHelp() 
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos( 2, 40 );
	g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.0f, 0.0f, 1.0f ) );
	g_pTxtHelper->DrawTextLine( L"Controls " );
	g_pTxtHelper->DrawTextLine(L"  \tF1:\t Hide help");
	g_pTxtHelper->DrawTextLine(L"  \tF2:\t Screenshot");
	g_pTxtHelper->DrawTextLine(L"  \t'R':\t Reset scan");
	g_pTxtHelper->DrawTextLine(L"  \t'9':\t Extract geometry (Marching Cubes)");
	g_pTxtHelper->DrawTextLine(L"  \t'8':\t Save recorded input data to sensor file (if enabled)");
	g_pTxtHelper->DrawTextLine(L"  \t'<tab>':\t Switch to free-view mode");
	g_pTxtHelper->DrawTextLine(L"  \t");
	g_pTxtHelper->DrawTextLine(L"  \t'1':\t Visualize reconstruction (default)");
	g_pTxtHelper->DrawTextLine(L"  \t'2':\t Visualize input depth");
	g_pTxtHelper->DrawTextLine(L"  \t'3':\t Visualize input color");
	g_pTxtHelper->DrawTextLine(L"  \t'4':\t Visualize input normals");
	g_pTxtHelper->DrawTextLine(L"  \t'5':\t Visualize phong shaded");
	g_pTxtHelper->DrawTextLine(L"  \t'H':\t GPU hash statistics");
	g_pTxtHelper->DrawTextLine(L"  \t'T':\t Print detailed timings");
	g_pTxtHelper->DrawTextLine(L"  \t'M':\t Debug hash");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Save hash to file");
	g_pTxtHelper->DrawTextLine(L"  \t'N':\t Load hash from file");
	g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
						 void* pUserContext )
{
	// Pass messages to dialog resource manager calls so GUI state is updated correctly
	*pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
	if( *pbNoFurtherProcessing )
		return 0;

	g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

	return 0;
}


void StopScanningAndExtractIsoSurfaceMC(const std::string& filename)
{
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	if (GlobalAppState::get().s_sensorIdx == 7) { //! hack for structure sensor
		std::cout << "[marching cubes] stopped receiving frames from structure sensor" << std::endl;
		getRGBDSensor()->stopReceivingFrames();
	}

	Timer t;


	g_marchingCubesHashSDF->clearMeshBuffer();
	if (!GlobalAppState::get().s_streamingEnabled) {
		//g_chunkGrid->stopMultiThreading();
		//g_chunkGrid->streamInToGPUAll();
		g_marchingCubesHashSDF->extractIsoSurface(g_sceneRep->getHashData(), g_sceneRep->getHashParams(), g_rayCast->getRayCastData());
		//g_chunkGrid->startMultiThreading();
	} else {
		vec4f posWorld = vec4f(g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
		vec3f p(posWorld.x, posWorld.y, posWorld.z);
		g_marchingCubesHashSDF->extractIsoSurface(*g_chunkGrid, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);
	}

	const mat4f& rigidTransform = mat4f::identity();//g_sceneRep->getLastRigidTransform();
	g_marchingCubesHashSDF->saveMesh(filename, &rigidTransform);

	std::cout << "Mesh generation time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

void ResetDepthSensing()
{
	g_sceneRep->reset();
	g_chunkGrid->reset();
	g_Camera.Reset();
}


void StopScanningAndSaveSDFHash(const std::string& filename = "test.hash") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;

	vec4f posWorld = vec4f(g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	g_chunkGrid->saveToFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);

	std::cout << "Saving Time " << t.getElapsedTime() << " seconds" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}


void StopScanningAndLoadSDFHash(const std::string& filename = "test.hash") {
	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();

	Timer t;

	vec4f posWorld = vec4f(g_sceneRep->getLastRigidTransform()*GlobalAppState::get().s_streamingPos, 1.0f); // trans lags one frame
	vec3f p(posWorld.x, posWorld.y, posWorld.z);

	ResetDepthSensing();
	g_chunkGrid->loadFromFile(filename, g_rayCast->getRayCastData(), p, GlobalAppState::getInstance().s_streamingRadius);

	std::cout << "Loading Time " << t.getElapsedTime() << " seconds" << std::endl;

	GlobalAppState::get().s_integrationEnabled = false;
	std::cout << "Integration enabled == false" << std::endl; 
	GlobalAppState::get().s_trackingEnabled = false;
	std::cout << "Tracking enabled == false" << std::endl;

	//g_sceneRep->debugHash();
	//g_chunkGrid->debugCheckForDuplicates();
}

//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
static int whichScreenshot = 0;


void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{

	if( bKeyDown ) {
		wchar_t sz[200];

		switch( nChar )
		{
		case VK_F1:
			g_bRenderHelp = !g_bRenderHelp;
			break;
		case VK_F2:
			swprintf_s(sz, 200, L"screenshot%d.bmp", whichScreenshot++);
			DXUTSnapD3D11Screenshot(sz, D3DX11_IFF_BMP);
			std::wcout << std::wstring(sz) << std::endl;
			break;
		case '\t':
			g_renderText = !g_renderText;
			break;
		case '1':
			GlobalAppState::get().s_RenderMode = 1;
			break;
		case '2':
			GlobalAppState::get().s_RenderMode = 2;
			break;
		case '3':
			GlobalAppState::get().s_RenderMode = 3;
			break;
		case '4':
			GlobalAppState::get().s_RenderMode = 4;
			break;
		case '5':
			GlobalAppState::get().s_RenderMode = 5;
			break;
		case '6':
			GlobalAppState::get().s_RenderMode = 6;
			break;
		case '7':
			GlobalAppState::get().s_RenderMode = 7;
			break;
			//case '8':
			//GlobalAppState::get().s_RenderMode = 8;
		case '8':
			{
				if (GlobalAppState::getInstance().s_recordData) {
					if (GlobalAppState::get().s_sensorIdx == 7) { //! hack for structure sensor
						std::cout << "[dump frames] stopped receiving frames from structure sensor" << std::endl;
						getRGBDSensor()->stopReceivingFrames();
					}
					g_RGBDSensor->saveRecordedFramesToFile(GlobalAppState::getInstance().s_recordDataFile);
				} else {
					std::cout << "Cannot save recording: enable \"s_recordData\" in parameter file" << std::endl;
				}
				break;
			}
			break;
		case '9':
			StopScanningAndExtractIsoSurfaceMC();
			break;
		case '0':
			GlobalAppState::get().s_RenderMode = 0;
			break;
		case 'T':
			GlobalAppState::get().s_timingsDetailledEnabled = !GlobalAppState::get().s_timingsDetailledEnabled;
			break;
		case 'Z':
			GlobalAppState::get().s_timingsTotalEnabled = !GlobalAppState::get().s_timingsTotalEnabled;
			break;
			//case VK_F3:
			//	GlobalAppState::get().s_texture_threshold += 0.02;
			//	std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			//	if(GlobalAppState::get().s_texture_threshold>1.0f)
			//		GlobalAppState::get().s_texture_threshold = 1.0f;
			//	break;
			//case VK_F4:
			//	GlobalAppState::get().s_texture_threshold -= 0.02;
			//	std::cout<<GlobalAppState::get().s_texture_threshold<<std::endl;
			//	if(GlobalAppState::get().s_texture_threshold<0.0f)
			//		GlobalAppState::get().s_texture_threshold = 0.0f;
			//	break;
		case 'R':
			ResetDepthSensing();
			break;
		case 'H':
			g_historgram->computeHistrogram(g_sceneRep->getHashData(), g_sceneRep->getHashParams());
			break;
		case 'M':
			g_sceneRep->debugHash();
			if (g_chunkGrid)	g_chunkGrid->debugCheckForDuplicates();
			break;
		case 'D':
			getRGBDSensor()->savePointCloud("test.ply");
			break;
		case 'N':
			StopScanningAndSaveSDFHash("test.hash");
			break;
		case 'B':
			StopScanningAndLoadSDFHash("test.hash");
			break;
		case 'I':
			{
				GlobalAppState::get().s_integrationEnabled = !GlobalAppState::get().s_integrationEnabled;
				if (GlobalAppState::get().s_integrationEnabled)		std::cout << "integration enabled" << std::endl;
				else std::cout << "integration disabled" << std::endl;
			}

		default:
			break;
		}
	}
}

//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
	switch( nControlID )
	{
		// Standard DXUT controls
	case IDC_TOGGLEFULLSCREEN:
		DXUTToggleFullScreen(); 
		break;
	case IDC_TOGGLEREF:
		DXUTToggleREF(); 
		break;
	case IDC_TEST:
		break;
	}
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
	return true;
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
	HRESULT hr = S_OK;

	V_RETURN(GlobalAppState::get().OnD3D11CreateDevice(pd3dDevice));

	ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

	V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
	g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );


	V_RETURN(DX11QuadDrawer::OnD3D11CreateDevice(pd3dDevice));
	V_RETURN(DX11PhongLighting::OnD3D11CreateDevice(pd3dDevice));

	TimingLogDepthSensing::init();

	std::vector<DXGI_FORMAT> formats;
	formats.push_back(DXGI_FORMAT_R32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);
	formats.push_back(DXGI_FORMAT_R32G32B32A32_FLOAT);

	V_RETURN(g_RGBDRenderer.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight));
	V_RETURN(g_CustomRenderTarget.OnD3D11CreateDevice(pd3dDevice, GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight, formats));

	D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	g_Camera.SetViewParams( &vecEye, &vecAt );


	g_sceneRep = new CUDASceneRepHashSDF(CUDASceneRepHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	g_rayCast = new CUDARayCastSDF(CUDARayCastSDF::parametersFromGlobalAppState(GlobalAppState::get(), g_CudaImageManager->getIntrinsics(), g_CudaImageManager->getIntrinsicsInv()));

	g_marchingCubesHashSDF = new CUDAMarchingCubesHashSDF(CUDAMarchingCubesHashSDF::parametersFromGlobalAppState(GlobalAppState::get()));
	g_historgram = new CUDAHistrogramHashSDF(g_sceneRep->getHashParams());

	g_chunkGrid = new CUDASceneRepChunkGrid(g_sceneRep, 
		GlobalAppState::get().s_streamingVoxelExtents, 
		GlobalAppState::get().s_streamingGridDimensions,
		GlobalAppState::get().s_streamingMinGridPos,
		GlobalAppState::get().s_streamingInitialChunkListSize,
		GlobalAppState::get().s_streamingEnabled,
		GlobalAppState::get().s_streamingOutParts);


	if (!GlobalAppState::get().s_reconstructionEnabled) {
		GlobalAppState::get().s_RenderMode = 2;
	}

	if (GlobalAppState::get().s_sensorIdx == 7) { // structure sensor
		getRGBDSensor()->startReceivingFrames();
	}

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11DestroyDevice();
	DXUTGetGlobalResourceCache().OnDestroyDevice();
	SAFE_DELETE( g_pTxtHelper );

	DX11QuadDrawer::OnD3D11DestroyDevice();
	DX11PhongLighting::OnD3D11DestroyDevice();
	GlobalAppState::get().OnD3D11DestroyDevice();

	g_RGBDRenderer.OnD3D11DestroyDevice();
	g_CustomRenderTarget.OnD3D11DestroyDevice();

	SAFE_DELETE(g_sceneRep);
	SAFE_DELETE(g_rayCast);
	SAFE_DELETE(g_marchingCubesHashSDF);
	SAFE_DELETE(g_historgram);
	SAFE_DELETE(g_chunkGrid);

	TimingLogDepthSensing::destroy();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
										 const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
	HRESULT hr = S_OK;

	V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

	// Setup the camera's projection parameters
	g_Camera.SetWindow( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height );
	g_Camera.SetButtonMasks( MOUSE_MIDDLE_BUTTON, MOUSE_WHEEL, MOUSE_LEFT_BUTTON );

	//g_Camera.SetRotateButtons(true, false, false);

	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	//D3DXVECTOR3 vecEye ( 0.0f, 0.0f, 0.0f );
	//D3DXVECTOR3 vecAt ( 0.0f, 0.0f, 1.0f );
	//g_Camera.SetViewParams( &vecEye, &vecAt );
	g_Camera.SetProjParams( D3DX_PI / 4, fAspectRatio, 0.1f, 10.0f );


	V_RETURN(DX11PhongLighting::OnResize(pd3dDevice, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height));

	return hr;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}

void reconstruction()
{
	//only if binary dump
	//if (GlobalAppState::get().s_sensorIdx == 3) 
	//	std::cout << "[ frame " << g_RGBDAdapter.getFrameNumber() << " ] " << g_sceneRep->getHashParams().m_numOccupiedBlocks << std::endl;
	//
	mat4f transformation = mat4f::identity();
	if (GlobalAppState::get().s_sensorIdx == 3 && GlobalAppState::get().s_binaryDumpSensorUseTrajectory) {
		transformation = g_RGBDSensor->getRigidTransform();
	}
	else {
		//TODO get SIFT transfrom from bundler
	}

	//if (g_RGBDAdapter.getFrameNumber() == 1 && GlobalAppState::get().s_binaryDumpSensorUseTrajectory) {
	//	transformation = g_RGBDAdapter.getRigidTransform();
	//	std::cout << transformation << std::endl;
	//}



	//unsigned int frameIdx = g_RGBDAdapter.getFrameNumber() - 1;
	//if (g_RGBDAdapter.getFrameNumber() == 19) {
	//	StopScanningAndExtractIsoSurfaceMC("./Scans/scans_prev.ply");
	//}
	//if (g_RGBDAdapter.getFrameNumber() > 20 && g_RGBDAdapter.getFrameNumber() < 25) {
	//	transformation.setIdentity();
	//}

	////std::cout << "frame number " << g_RGBDAdapter.getFrameNumber() << std::endl;
	//g_CudaImageManager->storeRGBDImage(g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), transformation);
	//
	//if (g_RGBDAdapter.getFrameNumber() == 25) {
	//	StopScanningAndExtractIsoSurfaceMC("./Scans/scans.ply");

	//	for (unsigned int i = 20; i < 25; i++) {
	//		g_sceneRep->deIntegrate(g_CudaImageManager->getCameraToWorld(i), g_CudaImageManager->getDepthCameraData(i), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
	//	}

	//	StopScanningAndExtractIsoSurfaceMC("./Scans/scans_deint.ply");
	//}
	 

	//g_CudaImageManager->storeRGBDImage(g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), transformation);

 //	g_sceneRep->integrate(transformation, g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
 //	StopScanningAndExtractIsoSurfaceMC("./Scans/scans.ply");
 //
 //	for (unsigned int i = 0; i < 100; i++) {
 //		g_CudaDepthSensor.process(DXUTGetD3D11DeviceContext());
 //		//transformation = g_RGBDAdapter.getRigidTransform() * mat4f::rotationX(50.0f);
	//	transformation.setIdentity();
 //		g_CudaImageManager->storeRGBDImage(g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), transformation);
 //
 //		g_sceneRep->integrate(transformation, g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
 //		//Util::writeToImage(g_CudaDepthSensor.getDepthCameraData().d_depthData, g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), "Int_depth" + std::to_string(i + 1) + ".png");
	//	//Util::writeToImage(g_CudaDepthSensor.getDepthCameraData().d_colorData, g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), "Int_color" + std::to_string(i + 1) + ".png");
 //	}

	//TimingLogDepthSensing::printTimings();

	//for (unsigned int i = 0; i < 100; i++) {
	//	g_sceneRep->deIntegrate(g_CudaImageManager->getCameraToWorld(i+1), g_CudaImageManager->getDepthCameraData(i+1), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
	//	//Util::writeToImage(g_CudaImageManager->getDepthCameraData(i + 1).d_depthData, g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), "deInt_depth" + std::to_string(i + 1) + ".png");
	//	//Util::writeToImage(g_CudaImageManager->getDepthCameraData(i + 1).d_colorData, g_RGBDAdapter.getWidth(), g_RGBDAdapter.getHeight(), "deInt_color" + std::to_string(i + 1) + ".png");
	//}	

	//StopScanningAndExtractIsoSurfaceMC("./Scans/scans_deint.ply");
	////g_sceneRep->debugHash();
	//std::cout << "DONE" << std::endl;
	//while (1);

	if (GlobalAppState::getInstance().s_recordData) {
		g_RGBDSensor->recordTrajectory(transformation);
	}


	if (transformation(0, 0) == -std::numeric_limits<float>::infinity()) {
		std::cout << "!!! TRACKING LOST !!!" << std::endl;
		GlobalAppState::get().s_reconstructionEnabled = false;
		return;
	}

	if (GlobalAppState::get().s_streamingEnabled) {
		vec4f posWorld = transformation*vec4f(GlobalAppState::getInstance().s_streamingPos, 1.0f); // trans laggs one frame *trans
		vec3f p(posWorld.x, posWorld.y, posWorld.z);

		g_chunkGrid->streamOutToCPUPass0GPU(p, GlobalAppState::get().s_streamingRadius, true, true);
		g_chunkGrid->streamInToGPUPass1GPU(true);

		//g_chunkGrid->debugCheckForDuplicates();
	}

	if (GlobalAppState::get().s_integrationEnabled) {
		//////////// MATTHIAS TODO INTEGRATE
		g_sceneRep->integrate(transformation, g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());

		//g_sceneRep->deIntegrate(transformation, g_CudaDepthSensor.getDepthCameraData(), g_CudaDepthSensor.getDepthCameraParams(), g_chunkGrid->getBitMaskGPU());
		//g_sceneRep->debugHash();
		//std::cout << "DONE" << std::endl;
		//while (1);
	} else {
		//compactification is required for the raycast splatting
		g_sceneRep->setLastRigidTransformAndCompactify(transformation, g_CudaDepthSensor.getDepthCameraData());
	}

	//{
	//	Util::saveScreenshot(g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, "ray/raycast_" + std::to_string(g_RGBDAdapter.getFrameNumber()) + ".png");
	//	g_sceneRep->debug("ray/hash_", g_RGBDAdapter.getFrameNumber());
	//}

}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------

void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime, float fElapsedTime, void* pUserContext )
{
	// If the settings dialog is being shown, then render it instead of rendering the app's scene
	//if(g_D3DSettingsDlg.IsActive())
	//{
	//	g_D3DSettingsDlg.OnRender(fElapsedTime);
	//	return;
	//}

	// Clear the back buffer
	static float ClearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearRenderTargetView(pRTV, ClearColor);
	pd3dImmediateContext->ClearDepthStencilView(pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0);

	// if we have received any valid new depth data we may need to draw
	HRESULT bGotDepth = g_CudaDepthSensor.process(pd3dImmediateContext);	//TODO process bundling

	//// Filtering
	//g_CudaDepthSensor.setFiterDepthValues(GlobalAppState::get().s_depthFilter, GlobalAppState::get().s_depthSigmaD, GlobalAppState::get().s_depthSigmaR);
	//g_CudaDepthSensor.setFiterIntensityValues(GlobalAppState::get().s_colorFilter, GlobalAppState::get().s_colorSigmaD, GlobalAppState::get().s_colorSigmaR);

	HRESULT hr = S_OK;

	///////////////////////////////////////
	// Render
	///////////////////////////////////////

	//Start Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.start(); }


	mat4f view = MatrixConversion::toMlib(*g_Camera.GetViewMatrix());

	mat4f t = mat4f::identity();
	t(1,1) *= -1.0f;	view = t * view * t;	//t is self-inverse

	if (bGotDepth == S_OK) {
		if (GlobalAppState::getInstance().s_recordData) {
			g_RGBDSensor->recordFrame();
			if (!GlobalAppState::get().s_reconstructionEnabled) {
				g_RGBDSensor->recordTrajectory(mat4f::zero());
			}
		}

		if (GlobalAppState::get().s_reconstructionEnabled) {
			reconstruction();
		}
	}

	if(GlobalAppState::get().s_RenderMode == 1)	{
		//default render mode (render ray casted depth)
		const mat4f& renderIntrinsics = g_CudaImageManager->getIntrinsics();

		//always render, irrespective whether there is a new depth frame available
		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());		
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
#ifdef STRUCTURE_SENSOR
		if (GlobalAppState::get().s_sensorIdx == 7) {
			ID3D11Texture2D* pSurface;
			HRESULT hr = DXUTGetDXGISwapChain()->GetBuffer( 0, __uuidof( ID3D11Texture2D ), reinterpret_cast< void** >( &pSurface ) );
			if (pSurface) {
				float* tex = (float*)CreateAndCopyToDebugTexture2D(pd3dDevice, pd3dImmediateContext, pSurface, true); //!!! TODO just copy no create
				((StructureSensor*)getRGBDSensor())->updateFeedbackImage((BYTE*)tex);
				SAFE_DELETE_ARRAY(tex);
			}
		}
#endif
	}
	else if (GlobalAppState::get().s_RenderMode == 2) {
		//default render mode (render ray casted color)
		const mat4f& renderIntrinsics = g_CudaImageManager->getIntrinsics();

		g_CustomRenderTarget.Clear(pd3dImmediateContext);
		g_CustomRenderTarget.Bind(pd3dImmediateContext);
		g_RGBDRenderer.RenderDepthMap(pd3dImmediateContext, g_rayCast->getRayCastData().d_depth, g_rayCast->getRayCastData().d_colors, g_rayCast->getRayCastParams().m_width, g_rayCast->getRayCastParams().m_height, MatrixConversion::toMlib(g_rayCast->getRayCastParams().m_intrinsicsInverse), view, renderIntrinsics, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight(), GlobalAppState::get().s_renderingDepthDiscontinuityThresOffset, GlobalAppState::get().s_renderingDepthDiscontinuityThresLin);
		g_CustomRenderTarget.Unbind(pd3dImmediateContext);

		DX11PhongLighting::render(pd3dImmediateContext, g_CustomRenderTarget.GetSRV(1), g_CustomRenderTarget.GetSRV(2), g_CustomRenderTarget.GetSRV(3), !GlobalAppState::get().s_useColorForRendering, g_CustomRenderTarget.getWidth(), g_CustomRenderTarget.getHeight());
		DX11QuadDrawer::RenderQuad(pd3dImmediateContext, DX11PhongLighting::GetColorsSRV(), 1.0f);
	} else {
		std::cout << "Unknown render mode " << GlobalAppState::get().s_RenderMode << std::endl;
	}

	// Stop Timing
	if (GlobalAppState::get().s_timingsDetailledEnabled) { GlobalAppState::get().WaitForGPU(); GlobalAppState::get().s_Timer.stop(); TimingLogDepthSensing::totalTimeRender += GlobalAppState::get().s_Timer.getElapsedTimeMS(); TimingLogDepthSensing::countTimeRender++; }


	TimingLogDepthSensing::printTimings();
	if (g_renderText) RenderText();


	DXUT_EndPerfEvent();
}

#endif