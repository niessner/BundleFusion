
#include "stdafx.h"

#include "KinectSensor.h"

#ifdef KINECT

KinectSensor::KinectSensor()
{
	// get resolution as DWORDS, but store as LONGs to avoid casts later
	DWORD width = 0;
	DWORD height = 0;

	NuiImageResolutionToSize(cDepthResolution, width, height);
	unsigned int depthWidth = static_cast<unsigned int>(width);
	unsigned int depthHeight = static_cast<unsigned int>(height);

	NuiImageResolutionToSize(cColorResolution, width, height);
	unsigned int colorWidth  = static_cast<unsigned int>(width);
	unsigned int colorHeight = static_cast<unsigned int>(height);

	RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight);

	m_colorToDepthDivisor = colorWidth/depthWidth;

	m_hNextDepthFrameEvent = INVALID_HANDLE_VALUE;
	m_pDepthStreamHandle = INVALID_HANDLE_VALUE;
	m_hNextColorFrameEvent = INVALID_HANDLE_VALUE;
	m_pColorStreamHandle = INVALID_HANDLE_VALUE;

	m_colorCoordinates = new LONG[depthWidth*depthHeight*2];

	m_bDepthImageIsUpdated = false;
	m_bDepthImageCameraIsUpdated = false;
	m_bNormalImageCameraIsUpdated = false;

	initializeDepthIntrinsics(2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 320.0f, 240.0f);
	initializeDepthExtrinsics(mat4f::identity());

	//MLIB_WARNING("TODO initialize color intrs/extr");
	initializeColorIntrinsics(2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 2.0f*NUI_CAMERA_SKELETON_TO_DEPTH_IMAGE_MULTIPLIER_320x240, 320.0f, 240.0f);
	initializeColorExtrinsics(mat4f::identity());

}

KinectSensor::~KinectSensor()
{
	if (NULL != m_pNuiSensor)
	{
		m_pNuiSensor->NuiShutdown();
		m_pNuiSensor->Release();
	}

	CloseHandle(m_hNextDepthFrameEvent);
	CloseHandle(m_hNextColorFrameEvent);

	// done with pixel data
	SAFE_DELETE_ARRAY(m_colorCoordinates);
}

void KinectSensor::createFirstConnected()
{
	INuiSensor* pNuiSensor = NULL;
	HRESULT hr = S_OK;

	int iSensorCount = 0;
	hr = NuiGetSensorCount(&iSensorCount);
	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return; }

	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i) {
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))	{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)	{
			m_pNuiSensor = pNuiSensor;
			break;
		}

		// This sensor wasn't OK, so release it since we're not using it
		pNuiSensor->Release();
	}

	if (NULL == m_pNuiSensor) {
		return;
	}

	// Initialize the Kinect and specify that we'll be using depth
	//hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX); 
	hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH); 
	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return;  }

	// Create an event that will be signaled when depth data is available
	m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// Open a depth image stream to receive depth frames
	hr = m_pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_DEPTH,
		//NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
		cDepthResolution,
		(8000 << NUI_IMAGE_PLAYER_INDEX_SHIFT),
		2,
		m_hNextDepthFrameEvent,
		&m_pDepthStreamHandle);
	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return; }

	// Create an event that will be signaled when color data is available
	m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	// Open a color image stream to receive color frames
	hr = m_pNuiSensor->NuiImageStreamOpen(
		NUI_IMAGE_TYPE_COLOR,
		cColorResolution,
		0,
		2,
		m_hNextColorFrameEvent,
		&m_pColorStreamHandle );
	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return; }

	INuiColorCameraSettings* colorCameraSettings;
	HRESULT hrFlag = m_pNuiSensor->NuiGetColorCameraSettings(&colorCameraSettings);

	if (hr != E_NUI_HARDWARE_FEATURE_UNAVAILABLE)
	{
		m_kinect4Windows = true;
	}

	//TODO MATTHIAS: does this function have to be called every frame?

	USHORT* test = new USHORT[getDepthWidth()*getDepthHeight()];
	// Get offset x, y coordinates for color in depth space
	// This will allow us to later compensate for the differences in location, angle, etc between the depth and color cameras
	m_pNuiSensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
		cColorResolution,
		cDepthResolution,
		getDepthWidth()*getDepthHeight(),
		test,
		getDepthWidth()*getDepthHeight()*2,
		m_colorCoordinates
		);
	SAFE_DELETE_ARRAY(test);

	// Start with near mode on (if possible)
	m_bNearMode = false;
	if (m_kinect4Windows) {
		toggleNearMode();
	}

	//toggleAutoWhiteBalance();
}

bool KinectSensor::processDepth()
{
	HRESULT hr = S_OK;

	//wait until data is available
	if (!(WAIT_OBJECT_0 == WaitForSingleObject(m_hNextDepthFrameEvent, 0)))	return false;

	// This code allows to get depth up to 8m
	BOOL bNearMode = false;
	if(m_kinect4Windows)
	{
		bNearMode = true;
	}

	INuiFrameTexture * pTexture = NULL;
	NUI_IMAGE_FRAME imageFrame;

	hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
	hr = m_pNuiSensor->NuiImageFrameGetDepthImagePixelFrameTexture(m_pDepthStreamHandle, &imageFrame, &bNearMode, &pTexture);

	NUI_LOCKED_RECT LockedRect;
	hr = pTexture->LockRect(0, &LockedRect, NULL, 0);
	if ( FAILED(hr) ) { return false; }

	NUI_DEPTH_IMAGE_PIXEL * pBuffer =  (NUI_DEPTH_IMAGE_PIXEL *) LockedRect.pBits;

	////#pragma omp parallel for
	//	for (int j = 0; j < (int)getDepthWidth()*(int)getDepthHeight(); j++)	{
	//		m_depthD16[j] = pBuffer[j].depth;
	//	}

	USHORT* test = new USHORT[getDepthWidth()*getDepthHeight()];

	float* depth = getDepthFloat();
	for (unsigned int j = 0; j < getDepthHeight(); j++) {
		for (unsigned int i = 0; i < getDepthWidth(); i++) {

			unsigned int desIdx = j*getDepthWidth() + i;
			unsigned int srcIdx = j*getDepthWidth() + (getDepthWidth() - i - 1);	//x-flip of the kinect

			const USHORT& d = pBuffer[srcIdx].depth;
			if (d == 0)
				depth[desIdx] = -std::numeric_limits<float>::infinity();
			else
				depth[desIdx] = (float)d * 0.001f;

			test[srcIdx] = d *8;
		}
	}

	hr = pTexture->UnlockRect(0);
	if ( FAILED(hr) ) { return false; };

	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);

	// Get offset x, y coordinates for color in depth space
	// This will allow us to later compensate for the differences in location, angle, etc between the depth and color cameras
	m_pNuiSensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
		cColorResolution,
		cDepthResolution,
		getDepthWidth()*getDepthHeight(),
		test,
		getDepthWidth()*getDepthHeight()*2,
		m_colorCoordinates
		);


	delete [] test;

	if (FAILED(hr)) { return false; };
	return true;
}

bool KinectSensor::processColor()
{
	if (! (WAIT_OBJECT_0 == WaitForSingleObject(m_hNextColorFrameEvent, 0)) )	return false;

	NUI_IMAGE_FRAME imageFrame;

	HRESULT hr = S_OK;
	hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pColorStreamHandle, 0, &imageFrame);
	if ( FAILED(hr) ) { return false; }

	NUI_LOCKED_RECT LockedRect;
	hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	if (FAILED(hr)) { return false; }

	// loop over each row and column of the color
#pragma omp parallel for
	for (int yi = 0; yi < (int)getColorHeight(); ++yi) {
		LONG y = yi;

		LONG* pDest = ((LONG*)m_colorRGBX) + (int)getColorWidth() * y;
		for (LONG x = 0; x < (int)getColorWidth(); ++x) {
			// calculate index into depth array
			//int depthIndex = x/m_colorToDepthDivisor + y/m_colorToDepthDivisor * getDepthWidth();	//TODO x flip
			int depthIndex = (getDepthWidth() - 1 - x/m_colorToDepthDivisor) + y/m_colorToDepthDivisor * getDepthWidth();

			// retrieve the depth to color mapping for the current depth pixel
			LONG colorInDepthX = m_colorCoordinates[depthIndex * 2];
			LONG colorInDepthY = m_colorCoordinates[depthIndex * 2 + 1];

			// make sure the depth pixel maps to a valid point in color space
			if ( colorInDepthX >= 0 && colorInDepthX < (int)getColorWidth() && colorInDepthY >= 0 && colorInDepthY < (int)getColorHeight() ) {
				// calculate index into color array
				LONG colorIndex = colorInDepthY * (int)getColorWidth() + colorInDepthX;	//TODO x flip
				//LONG colorIndex = colorInDepthY * (int)getColorWidth() + (getColorWidth() - 1 - colorInDepthX);

				// set source for copy to the color pixel
				LONG* pSrc = ((LONG *)LockedRect.pBits) + colorIndex;					
				LONG tmp = *pSrc;
				vec4uc* bgr = (vec4uc*)&tmp;
				std::swap(bgr->x, bgr->z);

				tmp|=0xFF000000; // Flag for is valid

				*pDest = tmp;
			} else {
				*pDest = 0x00000000;
			}
			pDest++;
		}
	}


	hr = imageFrame.pFrameTexture->UnlockRect(0);
	if (FAILED(hr)) { return false; };

	hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pColorStreamHandle, &imageFrame);
	if (FAILED(hr)) { return false; };


	return true;
}

bool KinectSensor::toggleNearMode()
{
	HRESULT hr = E_FAIL;

	if ( m_pNuiSensor )
	{
		hr = m_pNuiSensor->NuiImageStreamSetImageFrameFlags(m_pDepthStreamHandle, m_bNearMode ? 0 : NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE);

		if ( SUCCEEDED(hr) )
		{
			m_bNearMode = !m_bNearMode;
		}
	}

	if (FAILED(hr)) { return false; };
	return true;
}

bool KinectSensor::toggleAutoWhiteBalance()
{
	INuiColorCameraSettings* colorCameraSettings;
	HRESULT hr = S_OK;
	hr = m_pNuiSensor->NuiGetColorCameraSettings(&colorCameraSettings);
	if (hr != E_NUI_HARDWARE_FEATURE_UNAVAILABLE) {	//feature only supported with windows Kinect

		BOOL ex;
		colorCameraSettings->GetAutoExposure(&ex);
		colorCameraSettings->SetAutoExposure(!ex);

		//double exposure;
		//colorCameraSettings->GetExposureTime(&exposure);

		//double minExp; colorCameraSettings->GetMinExposureTime(&minExp);
		//double maxExp; colorCameraSettings->GetMaxExposureTime(&maxExp);
		//std::cout << exposure << std::endl;
		//std::cout << minExp << std::endl;
		//std::cout << maxExp << std::endl;
		//colorCameraSettings->SetExposureTime(6000);

		//double fr;
		//colorCameraSettings->GetFrameInterval(&fr);
		//std::cout << fr << std::endl;

		//double gain;
		//hr = colorCameraSettings->GetGain(&gain);
		//std::cout << gain << std::endl;
		//double minG; colorCameraSettings->GetMinGain(&minG);
		//double maxG; colorCameraSettings->GetMaxGain(&maxG);
		//std::cout << minG << std::endl;
		//std::cout << maxG << std::endl;

		hr = colorCameraSettings->SetGain(4);

		BOOL ab;
		colorCameraSettings->GetAutoWhiteBalance(&ab);
		colorCameraSettings->SetAutoWhiteBalance(!ab);

		colorCameraSettings->SetWhiteBalance(4000);	//this is a wild guess; but it seems that the previously 'auto-set' value cannot be obtained
		//LONG min; colorCameraSettings->GetMinWhiteBalance(&min);
		//LONG max; colorCameraSettings->GetMaxWhiteBalance(&max);
		//std::cout << min << std::endl;
		//std::cout << max << std::endl;
	}

	if (FAILED(hr)) { return false; };
	return true;
}

#endif