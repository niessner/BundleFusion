
#include "stdafx.h"

#include "KinectOneSensor.h"

#ifdef KINECT_ONE

//#include "NuiKinectFusionApi.h"

KinectOneSensor::KinectOneSensor()
{
	const unsigned int colorWidth  = 1920;
	const unsigned int colorHeight = 1080;

	const unsigned int depthWidth  = 512;
	const unsigned int depthHeight = 424;

	RGBDSensor::init(depthWidth, depthHeight, depthWidth, depthHeight, 2); // color is mapped to depth space

	m_inputColorWidth = colorWidth;
	m_inputColorHeight = colorHeight;

	m_pKinectSensor = NULL;
	m_pDepthRawPixelBuffer = NULL;
	m_pColorCoordinates = NULL;
	m_pDepthVisibilityTestMap = NULL;
	m_pDepthUndistortedPixelBuffer = NULL;
	m_pDepthDistortionMap = NULL;
	m_pDepthDistortionLT = NULL;

	createFirstConnected();

	// create heap storage for color pixel data in RGBX format
	m_pColorRGBX = new RGBQUAD[colorWidth*colorHeight];

	IMultiSourceFrame* pMultiSourceFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IColorFrame* pColorFrame = NULL;

	HRESULT hr = S_FALSE;
	while(hr != S_OK) hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

	if(SUCCEEDED(hr))
	{
		IDepthFrameReference* pDepthFrameReference = NULL;

		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}

		SafeRelease(pDepthFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		IColorFrameReference* pColorFrameReference = NULL;

		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}

		SafeRelease(pColorFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		INT64 nDepthTime = 0;
		IFrameDescription* pDepthFrameDescription = NULL;
		int nDepthWidth = 0;
		int nDepthHeight = 0;

		IFrameDescription* pColorFrameDescription = NULL;
		int nColorWidth = 0;
		int nColorHeight = 0;
		RGBQUAD *pColorBuffer = NULL;

		// get depth frame data
		hr = pDepthFrame->get_RelativeTime(&nDepthTime);

		if (SUCCEEDED(hr))
		{
			// Intrinsics
			CameraIntrinsics intrinsics = {};
			m_pCoordinateMapper->GetDepthCameraIntrinsics(&intrinsics);
			initializeDepthIntrinsics(intrinsics.FocalLengthX, intrinsics.FocalLengthY, intrinsics.PrincipalPointX, intrinsics.PrincipalPointY);
			initializeColorIntrinsics(intrinsics.FocalLengthX, intrinsics.FocalLengthY, intrinsics.PrincipalPointX, intrinsics.PrincipalPointY);
			//initializeDepthIntrinsics( 3.6214298524455461e+002, 3.6220435291479595e+002, 2.5616216259758841e+002, 2.0078875999601487e+002);
			//initializeColorIntrinsics(1.0556311615223119e+003, 1.0557253330803749e+003, 9.4264212485622727e+002, 5.3125563902269801e+002);

			// Extrinsics
			//Matrix3f R; R.setIdentity(); Vector3f t; t.setZero();
			//initializeColorExtrinsics(R, t);

			//R(0, 0) = 9.9998621443730407e-001; R(0, 1) = -1.2168971895208723e-003; R(0, 2) = -5.1078465697614612e-003;
			//R(1, 0) = 1.2178529255848945e-003; R(1, 1) =  9.9999924148788211e-001; R(1, 2) =  1.8400519565076159e-004;
			//R(2, 0) = 5.1076187799924972e-003; R(2, 1) = -1.9022326492404629e-004; R(2, 2) =  9.9998693793744509e-001;

			//t[0] = -5.1589449841384898e+001;  t[1] = -1.1102720138477913e+000; t[2] = -1.2127048071059605e+001;
			//t[0] /= 1000.0f; t[1] /= 1000.0f; t[2] /= 1000.0f;

			//initializeDepthExtrinsics(R, t);

			initializeDepthExtrinsics(mat4f::identity());
			initializeColorExtrinsics(mat4f::identity());
		}
		SafeRelease(pDepthFrameDescription);
		SafeRelease(pColorFrameDescription);
	}
	{
		// distortion
		unsigned int depthBufferSize = depthWidth * depthHeight;
		m_pDepthRawPixelBuffer = new UINT16[depthBufferSize];
		m_pColorCoordinates = new ColorSpacePoint[depthBufferSize];
		m_pDepthVisibilityTestMap = new UINT16[(colorWidth >> cVisibilityTestQuantShift) * (colorHeight >> cVisibilityTestQuantShift)]; 
		m_pDepthUndistortedPixelBuffer = new UINT16[depthBufferSize];
		m_pDepthDistortionMap = new DepthSpacePoint[depthBufferSize];
		m_pDepthDistortionLT = new UINT[depthBufferSize];

		setupUndistortion();
	}
	m_bFirstFrame = true;

	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pMultiSourceFrame);
}

KinectOneSensor::~KinectOneSensor()
{
	if (m_pKinectSensor)			m_pKinectSensor->Release();
	if (m_pMultiSourceFrameReader)	m_pMultiSourceFrameReader->Release();
	if (m_pCoordinateMapper)		m_pCoordinateMapper->Release();
	if (m_pColorCoordinates)		delete [] m_pColorCoordinates;

	if (m_depthSpacePoints)			delete [] m_depthSpacePoints;
	if (m_cameraSpacePoints)		delete [] m_cameraSpacePoints;

	if (m_pColorRGBX)
	{
		delete [] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	SAFE_DELETE_ARRAY(m_pDepthRawPixelBuffer);
	SAFE_DELETE_ARRAY(m_pDepthVisibilityTestMap);
	SAFE_DELETE_ARRAY(m_pDepthUndistortedPixelBuffer);
	SAFE_DELETE_ARRAY(m_pDepthDistortionMap);
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
}

void KinectOneSensor::createFirstConnected()
{
	HRESULT hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return; };

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get coordinate mapper and the frame reader
		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->OpenMultiSourceFrameReader(
				FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
				&m_pMultiSourceFrameReader);
		}
	}

	if (FAILED(hr)) { std::cerr << "failed to initialize kinect sensor" << std::endl; return; };
}

bool KinectOneSensor::processDepth()
{
	IMultiSourceFrame* pMultiSourceFrame = NULL;
	IDepthFrame* pDepthFrame = NULL;
	IColorFrame* pColorFrame = NULL;

	HRESULT hr = m_pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

	if(SUCCEEDED(hr))
	{
		IDepthFrameReference* pDepthFrameReference = NULL;

		hr = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}

		SafeRelease(pDepthFrameReference);
	}
	if (SUCCEEDED(hr)) hr = copyDepth(pDepthFrame);

	if (SUCCEEDED(hr))
	{
		IColorFrameReference* pColorFrameReference = NULL;

		hr = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hr))
		{
			hr = pColorFrameReference->AcquireFrame(&pColorFrame);
		}

		SafeRelease(pColorFrameReference);
	}

	if (SUCCEEDED(hr))
	{
		INT64 nDepthTime = 0;
		IFrameDescription* pDepthFrameDescription = NULL;

		UINT nDepthBufferSize = 0;
		//UINT16 *pDepthBuffer = NULL;

		IFrameDescription* pColorFrameDescription = NULL;

		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nColorBufferSize = 0;
		RGBQUAD *pColorBuffer = NULL;

		// get depth frame data
		if (SUCCEEDED(hr)) hr = pDepthFrame->get_RelativeTime(&nDepthTime);
		if (SUCCEEDED(hr)) hr = pDepthFrame->get_FrameDescription(&pDepthFrameDescription);
		//if (SUCCEEDED(hr)) hr = pDepthFrame->AccessUnderlyingBuffer(&nDepthBufferSize, &pDepthBuffer);

		// get color frame data
		if (SUCCEEDED(hr)) hr = pColorFrame->get_FrameDescription(&pColorFrameDescription);
		if (SUCCEEDED(hr)) hr = pColorFrame->get_RawColorImageFormat(&imageFormat);

		if (SUCCEEDED(hr))
		{
			if (m_pColorRGBX)
			{
				pColorBuffer = m_pColorRGBX;
				nColorBufferSize = m_inputColorWidth * m_inputColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nColorBufferSize, reinterpret_cast<BYTE*>(pColorBuffer), ColorImageFormat_Rgba);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		if (SUCCEEDED(hr))
		{	
			// Make sure we've received valid data
			//if (pDepthBuffer && pColorBuffer)
			if (m_pDepthUndistortedPixelBuffer && pColorBuffer)
			{
#pragma omp parallel for
				for(int i = 0; i<(int)(m_depthWidth * m_depthHeight); i++)
				{
					float dF = (float)m_pDepthUndistortedPixelBuffer[i]*0.001f;//(float)pDepthBuffer[i]*0.001f;
					if (m_pDepthUndistortedPixelBuffer[i] != 0)
						int a = 5;
					if(dF >= GlobalAppState::get().s_sensorDepthMin && dF <= GlobalAppState::get().s_sensorDepthMax) 
						getDepthFloat()[i] = dF;
					else																				 
						getDepthFloat()[i] = -std::numeric_limits<float>::infinity();
				}
				incrementRingbufIdx();

				//{
				//	UINT depthSize = m_depthHeight*m_depthWidth;
				//	UINT colorSize = m_inputColorHeight*m_inputColorWidth;
				//	float* depth = new float[depthSize];
				//	ml::vec4uc* color = new ml::vec4uc[colorSize];

				//	for (UINT i = 0; i < depthSize; i++) {
				//		if (m_pDepthRawPixelBuffer[i] == 0) depth[i] = -std::numeric_limits<float>::infinity();
				//		else depth[i] = (float)m_pDepthRawPixelBuffer[i]*0.001f;
				//	}
				//	for (UINT i = 0; i < colorSize; i++) {
				//		color[i] = ml::vec4uc(m_pColorRGBX[i].rgbRed, m_pColorRGBX[i].rgbGreen, m_pColorRGBX[i].rgbBlue, 255);
				//	}

				//	ml::DepthImage dImage(m_depthHeight, m_depthWidth, depth);
				//	ml::ColorImageRGB cdImage(dImage);
				//	ml::ColorImageR8G8B8A8 cImage(m_inputColorHeight, m_inputColorWidth, color);
				//	ml::FreeImageWrapper::saveImage("depth.png", cdImage);
				//	ml::FreeImageWrapper::saveImage("color.png", cImage);
				//	int a = 5;
				//	SAFE_DELETE_ARRAY(depth);
				//	SAFE_DELETE_ARRAY(color);
				//}
				
				hr = mapColorToDepth();

				//if (SUCCEEDED(hr)) {
				//	DepthImage dImage(m_depthHeight, m_depthWidth, m_depthFloat[0]);
				//	ColorImageRGB cdImage(dImage);
				//	ColorImageR8G8B8A8 cImage(m_colorHeight, m_colorWidth, m_colorRGBX);
				//	FreeImageWrapper::saveImage("d-color.png", cImage);
				//	FreeImageWrapper::saveImage("d-depth.png", cdImage);
				//}
				//int a = 5;
			}
		}

		SafeRelease(pDepthFrameDescription);
		SafeRelease(pColorFrameDescription);
	}

	SafeRelease(pDepthFrame);
	SafeRelease(pColorFrame);
	SafeRelease(pMultiSourceFrame);

	if (SUCCEEDED(hr) && m_bFirstFrame) {
		m_bFirstFrame = false;
		return false;
	}

	if (FAILED(hr)) return false;
	return true;
}

HRESULT KinectOneSensor::setupUndistortion()
{
	HRESULT hr = E_UNEXPECTED;

	float focalLengthX = m_depthIntrinsics(0,0) / m_depthWidth;
	float focalLengthY = m_depthIntrinsics(1,1) / m_depthHeight;
	float principalPointX = m_depthIntrinsics(0,2) / m_depthWidth;
	float principalPointY = m_depthIntrinsics(1,2) / m_depthHeight;

	if (m_depthIntrinsics(0,2) != 0)
	{

		const UINT width = getDepthWidth();
		const UINT height = getDepthHeight();
		const UINT depthBufferSize = width * height;

		CameraSpacePoint cameraFrameCorners[4] = //at 1 meter distance. Take into account that depth frame is mirrored
		{
			{ -principalPointX / focalLengthX, principalPointY / focalLengthY, 1.f }, // LT
			{ (1.f - principalPointX) / focalLengthX, principalPointY / focalLengthY, 1.f }, // RT 
			{ -principalPointX / focalLengthX, (principalPointY - 1.f) / focalLengthY, 1.f }, // LB
			{ (1.f - principalPointX) / focalLengthX, (principalPointY - 1.f) / focalLengthY, 1.f } // RB
		};

		for(UINT rowID = 0; rowID < height; rowID++)
		{
			const float rowFactor = float(rowID) / float(height - 1);
			const CameraSpacePoint rowStart = 
			{
				cameraFrameCorners[0].X + (cameraFrameCorners[2].X - cameraFrameCorners[0].X) * rowFactor,
				cameraFrameCorners[0].Y + (cameraFrameCorners[2].Y - cameraFrameCorners[0].Y) * rowFactor,
				1.f
			};

			const CameraSpacePoint rowEnd = 
			{
				cameraFrameCorners[1].X + (cameraFrameCorners[3].X - cameraFrameCorners[1].X) * rowFactor,
				cameraFrameCorners[1].Y + (cameraFrameCorners[3].Y - cameraFrameCorners[1].Y) * rowFactor,
				1.f
			};

			const float stepFactor = 1.f / float(width - 1);
			const CameraSpacePoint rowDelta = 
			{
				(rowEnd.X - rowStart.X) * stepFactor,
				(rowEnd.Y - rowStart.Y) * stepFactor,
				0
			};

			//_ASSERT(width == NUI_DEPTH_RAW_WIDTH);
			_ASSERT(width == 512);
			CameraSpacePoint cameraCoordsRow[512]; //! todo

			CameraSpacePoint currentPoint = rowStart;
			for(UINT i = 0; i < width; i++)
			{
				cameraCoordsRow[i] = currentPoint;
				currentPoint.X += rowDelta.X;
				currentPoint.Y += rowDelta.Y;
			}

			hr = m_pCoordinateMapper->MapCameraPointsToDepthSpace(width, cameraCoordsRow, width, &m_pDepthDistortionMap[rowID * width]);
			if(FAILED(hr)) throw MLIB_EXCEPTION("Failed to initialize Kinect Coordinate Mapper.");
		}

		if (nullptr == m_pDepthDistortionLT) throw MLIB_EXCEPTION("Failed to initialize Kinect Fusion depth image distortion Lookup Table.");

		UINT* pLT = m_pDepthDistortionLT;
		for(UINT i = 0; i < depthBufferSize; i++, pLT++)
		{
			//nearest neighbor depth lookup table 
			UINT x = UINT(m_pDepthDistortionMap[i].X + 0.5f);
			UINT y = UINT(m_pDepthDistortionMap[i].Y + 0.5f);

			*pLT = (x < width && y < height)? x + y * width : UINT_MAX; 
		}
	}
	else
	{
		return S_FALSE; // invalid intrinsics
	}
	return S_OK;
}

HRESULT KinectOneSensor::copyDepth(IDepthFrame* pDepthFrame)
{
	// Check the frame pointer
	if (NULL == pDepthFrame)
	{
		return E_INVALIDARG;
	}

	UINT nBufferSize = 0;
	UINT16 *pBuffer = NULL;

	HRESULT hr = pDepthFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);
	if (FAILED(hr))
	{
		return hr;
	}

	//copy and remap depth
	const UINT bufferLength =  getDepthWidth() * getDepthHeight();
	UINT16 * pDepth = m_pDepthUndistortedPixelBuffer;
	UINT16 * pRawDepth = m_pDepthRawPixelBuffer;
	for(UINT i = 0; i < bufferLength; i++, pDepth++, pRawDepth++)
	{
		const UINT id = m_pDepthDistortionLT[i];
		*pDepth = id < bufferLength? pBuffer[id] : 0;
		*pRawDepth = pBuffer[i];
	}

	return S_OK;
}

HRESULT KinectOneSensor::mapColorToDepth()
{
	HRESULT hr = S_OK;

	// Get the coordinates to convert color to depth space
	hr = m_pCoordinateMapper->MapDepthFrameToColorSpace(m_depthWidth * m_depthHeight, m_pDepthRawPixelBuffer, 
		m_depthWidth * m_depthHeight, m_pColorCoordinates);
	if (FAILED(hr)) return hr;

	// construct dense depth points visibility test map so we can test for depth points that are invisible in color space
	const UINT16* const pDepthEnd = m_pDepthRawPixelBuffer + m_depthWidth * m_depthHeight;
	const ColorSpacePoint* pColorPoint = m_pColorCoordinates;
	const UINT testMapWidth = UINT(m_inputColorWidth >> cVisibilityTestQuantShift);
	const UINT testMapHeight = UINT(m_inputColorHeight >> cVisibilityTestQuantShift);
	ZeroMemory(m_pDepthVisibilityTestMap, testMapWidth * testMapHeight * sizeof(UINT16));
	for(const UINT16* pDepth = m_pDepthRawPixelBuffer; pDepth < pDepthEnd; pDepth++, pColorPoint++)
	{
		const UINT x = UINT(pColorPoint->X + 0.5f) >> cVisibilityTestQuantShift;
		const UINT y = UINT(pColorPoint->Y + 0.5f) >> cVisibilityTestQuantShift;
		if(x < testMapWidth && y < testMapHeight)
		{
			const UINT idx = y * testMapWidth + x;
			const UINT16 oldDepth = m_pDepthVisibilityTestMap[idx];
			const UINT16 newDepth = *pDepth;
			if(!oldDepth || oldDepth > newDepth)
			{
				m_pDepthVisibilityTestMap[idx] = newDepth;
			}
		}
	}


	// Loop over each row and column of the destination color image and copy from the source image
	// Note that we could also do this the other way, and convert the depth pixels into the color space, 
	// avoiding black areas in the converted color image and repeated color images in the background
	// However, then the depth would have radial and tangential distortion like the color camera image,
	// which is not ideal for Kinect Fusion reconstruction.

	float* depth = getDepthFloat(); // invalidate depth where there is no color information
#pragma omp parallel for
	for (int y = 0; y < (int)m_depthHeight; y++) {
		const UINT depthImagePixels = m_depthWidth * m_depthHeight;
		const UINT testMapWidth = UINT(m_inputColorWidth >> cVisibilityTestQuantShift);

		UINT destIndex = y * m_depthWidth;
		for (UINT x = 0; x < (UINT)m_depthWidth; ++x, ++destIndex)
		{
			unsigned int pixelColor = 0;
			const UINT mappedIndex = m_pDepthDistortionLT[destIndex];
			if(mappedIndex < depthImagePixels)
			{
				// retrieve the depth to color mapping for the current depth pixel
				const ColorSpacePoint colorPoint = m_pColorCoordinates[mappedIndex];

				// make sure the depth pixel maps to a valid point in color space
				const UINT colorX = (UINT)(colorPoint.X + 0.5f);
				const UINT colorY = (UINT)(colorPoint.Y + 0.5f);
				if (colorX < m_inputColorWidth && colorY < m_inputColorHeight)
				{
					const UINT16 depthValue = m_pDepthRawPixelBuffer[mappedIndex];
					const UINT testX = colorX >> cVisibilityTestQuantShift;
					const UINT testY = colorY >> cVisibilityTestQuantShift;
					const UINT testIdx = testY * testMapWidth + testX;
					const UINT16 depthTestValue = m_pDepthVisibilityTestMap[testIdx];
					_ASSERT(depthValue >= depthTestValue);
					if(depthValue - depthTestValue < cDepthVisibilityTestThreshold)
					{
						// calculate index into color array
						const UINT colorIndex = colorX + (colorY * m_inputColorWidth);
						const RGBQUAD& pixel = m_pColorRGBX[colorIndex];
						pixelColor |= pixel.rgbRed;
						pixelColor <<= 8;
						pixelColor |= pixel.rgbGreen;
						pixelColor <<= 8;
						pixelColor |= pixel.rgbBlue;
						pixelColor |= 0xFF000000;
					}
				}
			}
			if (pixelColor == 0) {
				m_colorRGBX[destIndex] = vec4uc(0,0,0,255);
				depth[destIndex] = -std::numeric_limits<float>::infinity();
			}
			else 
				((LONG*)m_colorRGBX)[destIndex] = pixelColor;
		}
	}

	return hr;
}

#endif