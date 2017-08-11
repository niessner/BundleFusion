#pragma once

/************************************************************************/
/* Kinect Sensor (the old version of a Kinect)                          */
/************************************************************************/

#include "GlobalAppState.h"

#ifdef KINECT

#include "RGBDSensor.h"
#include <NuiApi.h>
#include <NuiSkeleton.h>

class KinectSensor : public RGBDSensor
{
public:
	KinectSensor();
	//! Constructor; allocates CPU memory and creates handles

	//! Destructor; releases allocated ressources
	~KinectSensor();

	//! Initializes the sensor
	void createFirstConnected();

	//! gets the next depth frame
	bool processDepth();

	//! maps the color to depth data and copies depth and color data to the GPU
	bool processColor();

	std::string getSensorName() const {
		return "Kinect";
	}

	//! toggles near mode if possible (only available on a windows Kinect)
	bool toggleNearMode();

	//! Toggle enable auto white balance
	bool toggleAutoWhiteBalance();
	

private:
	INuiSensor*		m_pNuiSensor;

	static const NUI_IMAGE_RESOLUTION   cDepthResolution = NUI_IMAGE_RESOLUTION_640x480;
	static const NUI_IMAGE_RESOLUTION   cColorResolution = NUI_IMAGE_RESOLUTION_640x480;

	HANDLE			m_hNextDepthFrameEvent;
	HANDLE			m_pDepthStreamHandle;
	HANDLE			m_hNextColorFrameEvent;
	HANDLE			m_pColorStreamHandle;

	LONG*			m_colorCoordinates;		// for mapping depth to color

	LONG			m_colorToDepthDivisor;

	bool			m_bDepthImageIsUpdated;
	bool			m_bDepthImageCameraIsUpdated;
	bool			m_bNormalImageCameraIsUpdated;

	bool			m_kinect4Windows;
	bool			m_bNearMode;

};
#endif