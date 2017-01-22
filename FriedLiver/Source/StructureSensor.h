#pragma once

#include "GlobalAppState.h"
#include "GlobalBundlingState.h"

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
	ExampleServerSession(int socketDescriptor, uplink::Server* server, bool sendFeedbackImage)
		: DesktopServerSession(socketDescriptor, server)
	{
		m_bSendFeedbackImage = sendFeedbackImage;
	}

	void toggleExposureAndWhiteBalance(bool lock = false) //lock forces the lock
	{
		uplink::SessionSetup sessionSetup;

		static bool toggle = true;

		if (toggle || lock)
		{
			sessionSetup.addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_Locked);
			sessionSetup.addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_Locked);

			std::cout << "awb/exp LOCKED" << std::endl;
			//uplink_log_info("Locked exposure and white balance.");
		}
		else
		{
			sessionSetup.addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_ContinuousAuto);
			sessionSetup.addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_ContinuousAuto);

			std::cout << "awb/exp unlocked" << std::endl;
			//uplink_log_info("Automatic exposure and white balance.");
		}

		server()._currentSession->sendSessionSetup(sessionSetup);

		toggle = !toggle;
	}

	virtual void onCustomCommand(const std::string& command)
	{
		if (command == "RecordButtonPressed")
		{
			if (!server().isRecording()) {
				toggleExposureAndWhiteBalance(true); //lock the awb/autoexp
				std::cout << "record button pressed" << std::endl;
				server().setRecordPressed(true);
			}
		}
		else if (command == "AutoLevelButtonPressed")
		{
			if (!server().isRecording()) toggleExposureAndWhiteBalance();
			// cannot toggle during scanning!
		}
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
				//static unsigned long long count = 0;
				//std::cout << "RGBD " << count << std::endl;

				UCHAR* colorBuffer = NULL;
				if (!cameraFrame.colorImage.isEmpty())
				{
					colorBuffer = (UCHAR*)cameraFrame.colorImage.planes[0].buffer;
				}

				USHORT* depthBuffer = (USHORT*)cameraFrame.depthImage.planes[0].buffer;
				int     depthWidth  = int(cameraFrame.depthImage.width);
				int     depthHeight = int(cameraFrame.depthImage.height);

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
					if (server().isRecording()) { //receive
						// Convert shifts to depth values.
						uplink::shift2depth(depthBuffer, depthWidth * depthHeight);
						server().receive(depthBuffer, colorBuffer);
					}
					//count++;
				}

				// Send ping-pong feedback image.
				// FIXME: This const-cast sucks.
				//if (!cameraFrame.colorImage.isEmpty())
					//sendImage(const_cast<uplink::Image&>(cameraFrame.colorImage));
				if (m_bSendFeedbackImage) {
					const uplink::Image& feedback = server().getFeedbackImage();
					if (!feedback.isEmpty()) {
						//FreeImageWrapper::saveImage("test.png", ColorImageR8G8B8(feedback.height, feedback.width, (vec3uc*)feedback.planes[0].buffer));
						sendImage(const_cast<uplink::Image&>(feedback));
					}
				}
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
	bool m_bSendFeedbackImage;
};


//------------------------------------------------------------------------------

struct ExampleSessionSetup : uplink::SessionSetup
{
	ExampleSessionSetup()
	{
		if (GlobalBundlingState::get().s_widthSIFT == 1296 && GlobalBundlingState::get().s_heightSIFT == 968)
			addSetColorModeAction(uplink::ColorMode_1296x968);
		else if (GlobalBundlingState::get().s_widthSIFT == 640 && GlobalBundlingState::get().s_heightSIFT == 480)
			addSetColorModeAction(uplink::ColorMode_VGA);
		else
			throw MLIB_EXCEPTION("invalid sift dimensions for structure sensor frames");
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

		//addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_Locked);
		//addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_Locked);
		addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_ContinuousAuto);		//default unlocked, can lock with button
		addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_ContinuousAuto);

		addSetDepthCameraCodecAction(uplink::ImageCodecId_CompressedShifts);
		addSetColorCameraCodecAction(uplink::ImageCodecId_JPEG);
		addSetFeedbackImageCodecAction(uplink::ImageCodecId_JPEG);
	}
};

//------------------------------------------------------------------------------

struct ExampleServerDelegate : uplink::ServerDelegate
{
	void sendClearAllButtonsCommand()
	{
		_server->_currentSession->sendCustomCommand("button:clear:*");
	}

	void sendButtonCreateCommand(std::string buttonPngFilepath, std::string commandName)
	{
		uplink::CustomCommand customCommand;
		customCommand.command += "button:create:";
		customCommand.command += char(0);
		customCommand.command += commandName;
		customCommand.command += '\0';

		std::ifstream f(buttonPngFilepath, std::ios::binary);
		if (!f.is_open()) throw MLIB_EXCEPTION("could not open button path " + buttonPngFilepath);
		std::string imageBytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		customCommand.command.insert(customCommand.command.end(), imageBytes.begin(), imageBytes.end());

		_server->_currentSession->sendCustomCommand(customCommand);
	}

	virtual uplink::ServerSession* newSession(int socketDescriptor, uplink::Server* server)
	{
		_server = server;

		return new ExampleServerSession(socketDescriptor, server, GlobalBundlingState::get().s_sendUplinkFeedbackImage);
	}

	virtual void onConnect(uintptr_t sessionId)
	{
		sendClearAllButtonsCommand();
		sendButtonCreateCommand("Media/record-button.png", "RecordButtonPressed");
		sendButtonCreateCommand("Media/auto-level-button.png", "AutoLevelButtonPressed");

		_server->_currentSession->sendSessionSetup(
			ExampleSessionSetup()
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
		unsigned int bufferFrameSize = 3;	// depth/color buffer size //!!! TODO
		m_server.init(bufferFrameSize, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight); 
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

		//std::cout << "depth intrinsics: " << std::endl << m_depthIntrinsics << std::endl;
		//std::cout << "color intrinsics: " << std::endl << m_colorIntrinsics << std::endl;
		//std::cout << "depth extrinsics: " << std::endl << m_depthExtrinsics << std::endl;
		//std::cout << "color extrinsics: " << std::endl << m_colorExtrinsics << std::endl;
	}

	bool processDepth();

	// already in processDepth
	bool processColor() { return true; }

	std::string getSensorName() const {
		return "StructureSensor";
	}

	void startReceivingFrames() { m_bIsReceivingFrames = true; waitForRecord(); }
	void stopReceivingFrames() { m_bIsReceivingFrames = false; m_server.stopRecording(); }

	void updateFeedbackImage(BYTE* tex) {
		m_server.updateFeedbackImage(tex);
	}

private:

	void waitForConnection() {
		std::cout << "waiting for connection... ";
		m_server.startListening();
		while (!m_server.hasCalibration()) {
			Sleep(0); // wait for calibration
		}
		std::cout << "ready!" << std::endl;
	}
	void waitForRecord() {
		std::cout << "waiting for record..." << std::endl;
		while (!m_server.isRecordPressed()) {
			Sleep(0); // wait for start record button
		}
		std::cout << "server: start recording" << std::endl;
		m_server.startRecording();
	}

	ExampleServerDelegate m_serverDelegate;
	uplink::DesktopServer m_server;

	float* m_oldDepth;
	UCHAR* m_oldColor;
};

#endif