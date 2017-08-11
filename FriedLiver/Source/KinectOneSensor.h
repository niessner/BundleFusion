#pragma once

//Only working with Kinect 2.0 SDK (which wants to run on Win8)

#include "GlobalAppState.h"

#ifdef KINECT_ONE

#include <Kinect.h>
#include "RGBDSensor.h"

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL) {
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

class KinectOneSensor : public RGBDSensor
{
public:
	static const int            cVisibilityTestQuantShift = 2; // shift by 2 == divide by 4
	static const UINT16         cDepthVisibilityTestThreshold = 50; //50 mm

	KinectOneSensor();

	~KinectOneSensor();

	void createFirstConnected();

	bool processDepth();

	bool processColor() {
		return true;
	}

	std::string getSensorName() const {
		return "KinectOne";
	}

	bool saveDepth(float *p_depth){return S_OK;};

private:

	HRESULT copyDepth( IDepthFrame* pDepthFrame );
	HRESULT setupUndistortion();
	/// <summary>
	/// Adjust color to the same space as depth
	/// </summary>
	/// <returns>S_OK for success, or failure code</returns>
	HRESULT mapColorToDepth();

	// Current Kinect
	IKinectSensor*          m_pKinectSensor;

	RGBQUAD*                m_pColorRGBX;

	// Frame reader
	IMultiSourceFrameReader* m_pMultiSourceFrameReader;

	// Mapping color to depth
	unsigned int			 m_inputColorWidth;
	unsigned int			 m_inputColorHeight;
	ICoordinateMapper*		 m_pCoordinateMapper; 
	ColorSpacePoint*         m_pColorCoordinates; 
	UINT16*                     m_pDepthRawPixelBuffer;
	UINT16*                     m_pDepthVisibilityTestMap;
	// Distortion
	UINT16*                     m_pDepthUndistortedPixelBuffer;
	DepthSpacePoint*            m_pDepthDistortionMap;
	UINT*                       m_pDepthDistortionLT;

	unsigned int m_depthPointCount;
	DepthSpacePoint* m_depthSpacePoints;
	CameraSpacePoint* m_cameraSpacePoints;

	bool				m_bFirstFrame;
};

#endif
