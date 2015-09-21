#pragma once

#include "GlobalAppState.h"

#ifdef STRUCTURE_SENSOR

#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "RGBDSensor.h"

#include <uplink.h>
#include "Uplink/desktop-server.h"

//------------------------------------------------------------------------------

struct ExampleServerSession : uplink::DesktopServerSession
{
	ExampleServerSession(int socketDescriptor, uplink::Server* server)
		: DesktopServerSession(socketDescriptor, server)
	{
	}

	virtual void onCustomCommand(const std::string& command)
	{
		// FIXME: Implement.
	}

	virtual bool onMessage(const uplink::Message& message)
	{
		switch (message.kind())
		{
		case uplink::MessageKind_DeviceMotionEvent:
			{
				std::cout << "IMU" << std::endl;
				break;
			}

		case uplink::MessageKind_CameraFrame:
			{
				//s_timer.frame();
				const uplink::CameraFrame& cameraFrame = message.as<uplink::CameraFrame>();
				static unsigned long long count = 0;
				//std::cout << "RGBD " << count << std::endl;

				UCHAR* colorBuffer = NULL;
				if (!cameraFrame.colorImage.isEmpty())
				{
					colorBuffer = (UCHAR*)cameraFrame.colorImage.planes[0].buffer;
				}

				USHORT* depthBuffer = (USHORT*)cameraFrame.depthImage.planes[0].buffer;
				int     depthWidth  = int(cameraFrame.depthImage.width);
				int     depthHeight = int(cameraFrame.depthImage.height);
				
				// Convert shifts to depth values.
				uplink::shift2depth(depthBuffer, depthWidth * depthHeight);

				static bool storeCalibration = true;

				if (!cameraFrame.colorImage.isEmpty() && !cameraFrame.depthImage.isEmpty()) { // valid
					if (storeCalibration) {
						if (colorBuffer != NULL) { // wait for first color frame

							server().updateCalibration(cameraFrame.depthImage.cameraInfo.calibration,
								cameraFrame.colorImage.cameraInfo.calibration,
								(unsigned int)cameraFrame.depthImage.width, (unsigned int)cameraFrame.depthImage.height,
								(unsigned int)cameraFrame.colorImage.width, (unsigned int)cameraFrame.colorImage.height);
							//server().ui().setPingPongColorCameraInfo(cameraFrame.colorImage.cameraInfo);
							storeCalibration = false;
						}
					}
					server().receive(depthBuffer, colorBuffer);
					count++;
				}

				// Send ping-pong feedback image.
				// FIXME: This const-cast sucks.
				//if (sendPingPongColorFeedback && !cameraFrame.colorImage.isEmpty())
				//	sendImage(const_cast<uplink::Image&>(cameraFrame.colorImage));
				const uplink::Image& feedback = server().getFeedbackImage();
				if (!feedback.isEmpty()) {
					//FreeImageWrapper::saveImage("test.png", ColorImageR8G8B8(feedback.height, feedback.width, (vec3uc*)feedback.planes[0].buffer));
					sendImage(const_cast<uplink::Image&>(feedback));
				}
				//static unsigned long long count = 0; // FIXME: Use a real steady-rate timer.
				//if (0 == count++ % 150)
				//{
				//	uplink_log_info("Camera frame input rate: %f Hz", server()._currentSession->channelStats[uplink::MessageKind_CameraFrame].receiving.adapteredRate());
				//	uplink_log_info("Device motion event input rate: %f Hz", server()._currentSession->channelStats[uplink::MessageKind_DeviceMotionEvent].receiving.adapteredRate());
				//	uplink_log_info("Feedback Image output rate: %f Hz", server()._currentSession->imageQueue.currentPoppingRate());
				//}
				//std::cout << "FPS: " << s_timer.framesPerSecond() << std::endl;

				break;
			}

		default:
			{
				std::cout << "Other" << std::endl;
				break;
			}
		}

		return true;
	}


	//static FrameTimer s_timer;
};


//------------------------------------------------------------------------------

struct ExampleSessionSetup : uplink::SessionSetup
{
	ExampleSessionSetup()
	{
		//addSetColorModeAction(uplink::ColorMode_VGA);
		addSetColorModeAction(uplink::ColorMode_1296x968);
		addSetDepthModeAction(uplink::DepthMode_VGA);
		addSetRegistrationModeAction(uplink::RegistrationMode_RegisteredDepth);
		addSetFrameSyncModeAction(uplink::FrameSyncMode_Depth);

		addSetSporadicFrameColorAction(false);
		addSetSporadicFrameColorDivisorAction(1);

		uplink::ChannelSettings channelSettings;
		channelSettings.droppingStrategy = uplink::DroppingStrategy_RandomOne;
		channelSettings.droppingThreshold = 90;
		channelSettings.bufferingStrategy = uplink::BufferingStrategy_Some;

		addSetRGBDFrameChannelAction(channelSettings);

		addSetSendMotionAction(false);
		addSetMotionRateAction(100);

		addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_Locked);
		addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_Locked);

		addSetDepthCameraCodecAction(uplink::ImageCodecId_CompressedShifts);
		addSetColorCameraCodecAction(uplink::ImageCodecId_JPEG);
		addSetFeedbackImageCodecAction(uplink::ImageCodecId_JPEG);
	}
};

//------------------------------------------------------------------------------

struct ExampleServerDelegate : uplink::ServerDelegate
{
	virtual uplink::ServerSession* newSession(int socketDescriptor, uplink::Server* server)
	{
		_server = server;

		return new ExampleServerSession(socketDescriptor, server);
	}

	virtual void onConnect(uintptr_t sessionId)
	{
		_server->_currentSession->sendSessionSetup(
			ExampleSessionSetup()
			//SporadicColorSessionSetup()
			//Depth60FPSSessionSetup()
			//WXGASessionSetup()
			);
	}

	uplink::Server* _server;
};

//------------------------------------------------------------------------------

class StructureSensor : public RGBDSensor
{
public:
	StructureSensor() :
		m_server("UplinkTool", 6666, &m_serverDelegate)
	{
		m_server.init(20, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight); // depth/color buffer size //!!! TODO
		m_oldDepth = NULL;
		m_oldColor = NULL;
	}
	~StructureSensor() { }

	virtual void reset()
	{
		RGBDSensor::reset();
	}


	void createFirstConnected() {
		waitForConnection();

		// get calibration
		uplink::CameraCalibration calibrationDepth, calibrationColor;
		unsigned int depthWidth, depthHeight, colorWidth, colorHeight;
		m_server.getCalibration(calibrationDepth, calibrationColor, depthWidth, depthHeight, colorWidth, colorHeight);
		RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight, 1);

		mat4f depthExtrinsics = quatf(calibrationDepth.qw, calibrationDepth.qx, calibrationDepth.qy, calibrationDepth.qz).matrix4x4(); // rotation
		depthExtrinsics.setTranslation(vec3f(calibrationDepth.tx, calibrationDepth.ty, calibrationDepth.tz));
		RGBDSensor::initializeDepthIntrinsics(calibrationDepth.fx, calibrationDepth.fy, calibrationDepth.cx, calibrationDepth.cy);
		RGBDSensor::initializeDepthExtrinsics(depthExtrinsics);

		mat4f colorExtrinsics = quatf(calibrationColor.qw, calibrationColor.qx, calibrationColor.qy, calibrationColor.qz).matrix4x4(); // rotation
		colorExtrinsics.setTranslation(vec3f(calibrationColor.tx, calibrationColor.ty, calibrationColor.tz));
		RGBDSensor::initializeColorIntrinsics(calibrationColor.fx, calibrationColor.fy, calibrationColor.cx, calibrationColor.cy);
		RGBDSensor::initializeColorExtrinsics(mat4f::identity());

		std::cout << "depth intrinsics: " << std::endl << m_depthIntrinsics << std::endl;
		std::cout << "color intrinsics: " << std::endl << m_colorIntrinsics << std::endl;
		std::cout << "depth extrinsics: " << std::endl << m_depthExtrinsics << std::endl;
		std::cout << "color extrinsics: " << std::endl << m_colorExtrinsics << std::endl;
	}

	bool processDepth();

	// already in processDepth
	bool processColor() { return true; }

	void startReceivingFrames() { m_bIsReceivingFrames = true; m_server.startReceiving(); }
	void stopReceivingFrames() { m_bIsReceivingFrames = false; m_server.stopReceiving(); }

	void updateFeedbackImage(BYTE* tex) {
		m_server.updateFeedbackImage(tex);
	}

private:

	void waitForConnection() {
		std::cout << "waiting for connection... ";
		m_server.startListening();
		while (!m_server.hasCalibration()) {
			// wait for calibration
		}
		std::cout << "ready!" << std::endl;
	}

	ExampleServerDelegate m_serverDelegate;
	uplink::DesktopServer m_server;

	float* m_oldDepth;
	UCHAR* m_oldColor;
};

#endif