# pragma once

# include "./servers.h"

namespace uplink {

struct DesktopUI;

//------------------------------------------------------------------------------

struct DesktopServer : Server
{
public:
     DesktopServer (const std::string& serviceName, int servicePort, objc_weak ServerDelegate* serverDelegate);
    ~DesktopServer ();

	void init(unsigned int bufferSize, unsigned int feedbackWidth, unsigned int feedbackHeight);

public:
	//DesktopServerUI& ui() { return *_ui; }

	void updateCalibration(const uplink::CameraCalibration& calibrationDepth, const uplink::CameraCalibration& calibrationColor,
		unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight);
	bool getCalibration(uplink::CameraCalibration& calibrationDepth, uplink::CameraCalibration& calibrationColor,
		unsigned int& depthWidth, unsigned int& depthHeight, unsigned int& colorWidth, unsigned int& colorHeight);
	bool hasCalibration() { const MutexLocker _(m_mutex); return m_bHasCalibration; }

	void setRecordPressed(bool b) { m_bRecordPressed = b; }
	bool isRecordPressed() const { return m_bRecordPressed; }

	void startRecording() { if (m_bRecordPressed) m_bIsRecording = true; }
	void stopRecording() { m_bIsRecording = false; }
	bool isRecording() const { return m_bIsRecording; }

	std::pair<float*, uint8*> process(float* oldDepth, uint8* oldColor) {

		//std::cout << "[get]: " << m_depthFilledList.size() << ", " << m_depthEmptyList.size() << std::endl;
		if (m_depthFilledList.empty() && !m_bIsRecording) return std::pair<float*, uint8*>(NULL, NULL);
		while (m_depthFilledList.empty()) {
			//std::cerr << "waiting for frames" << std::endl;
			sleep(0.01f);
		}
		//std::cout << "[get]: " << m_depthFilledList.size() << ", " << m_depthEmptyList.size() << std::endl;

		const MutexLocker m(m_mutex);
		float* depth = m_depthFilledList.front();
		m_depthFilledList.pop_front();
		if (oldDepth != NULL) m_depthEmptyList.push_back(oldDepth);

		uint8* color = m_colorFilledList.front();
		m_colorFilledList.pop_front();
		if (oldColor != NULL) m_colorEmptyList.push_back(oldColor);

		//std::cout << "[get success]" << std::endl;
		return std::pair<float*, uint8*>(depth, color);
	}

	void receive(uint16* recDepth, uint8* recColor) {

		if (!m_bIsRecording || recColor == NULL || recDepth == NULL) return;
		//std::cout << "[receive]: " << m_depthFilledList.size() << ", " << m_depthEmptyList.size() << std::endl;
		while (m_depthEmptyList.empty()) {
			//std::cout << "list full -- frame lost" << std::endl;
			return;
		}

		float* depth = NULL;
		uint8* color = NULL;
		{
			const MutexLocker m(m_mutex);
			depth = m_depthEmptyList.front();
			m_depthEmptyList.pop_front();
			while (m_colorEmptyList.empty()) {
				std::cout << "ERROR: color/depth not synced" << std::endl;
				return;
			}
			color = m_colorEmptyList.front();
			m_colorEmptyList.pop_front();
		}
		// depth
		for (unsigned int i = 0; i < m_depthWidth*m_depthHeight; i++) {
			uint16 v = recDepth[i];
			if (v > 0 && v < shift2depth(0xffff)) {
				depth[i] = (float)v * 0.001f;
			}
			else {
				depth[i] = -std::numeric_limits<float>::infinity();
			}
		}
		// color
		for (unsigned int y = 0; y < m_colorHeight; y++) {
			for (unsigned int x = 0; x < m_colorWidth; x++) {
				unsigned int idx = y * m_colorWidth + x;
				color[idx*m_numColorChannels + 0] = recColor[idx * 3 + 0];
				color[idx*m_numColorChannels + 1] = recColor[idx * 3 + 1];
				color[idx*m_numColorChannels + 2] = recColor[idx * 3 + 2];
				color[idx*m_numColorChannels + 3] = 255;
			}
		}

		const MutexLocker m(m_mutex);
		m_depthFilledList.push_back(depth);
		m_colorFilledList.push_back(color);
	}

	void updateFeedbackImage(BYTE* data) {
		m_feedbackImage.format = uplink::ImageFormat_RGB;
		m_feedbackImage.width = m_feedbackWidth;
		m_feedbackImage.height = m_feedbackHeight;
		m_feedbackImage.planes[0].buffer = m_feedbackImageData;
		m_feedbackImage.planes[0].sizeInBytes = sizeof(uint8) * m_feedbackWidth * m_feedbackHeight;
		m_feedbackImage.planes[0].bytesPerRow = sizeof(uint8) * m_feedbackWidth;

#pragma omp parallel for
		for (int i = 0; i < m_feedbackImage.width * m_feedbackImage.height; i++) {
			m_feedbackImageData[i * 3 + 0] = data[i * 4 + 0];
			m_feedbackImageData[i * 3 + 1] = data[i * 4 + 1];
			m_feedbackImageData[i * 3 + 2] = data[i * 4 + 2];
		}
	}
	const Image& getFeedbackImage() const { return m_feedbackImage; }

private:
	//DesktopServerUI* _ui;

	std::list<float*> m_depthEmptyList;
	std::list<float*> m_depthFilledList;

	std::list<uint8*> m_colorEmptyList;
	std::list<uint8*> m_colorFilledList;

	std::vector<float*> m_depthBuffer; // data buffers
	std::vector<uint8*>  m_colorBuffer;

	//std::list<double> m_depthTimestamps;
	//std::list<double> m_colorTimestamps;

	Mutex				 m_mutex;

	unsigned int m_depthWidth;
	unsigned int m_depthHeight;
	unsigned int m_colorWidth;
	unsigned int m_colorHeight;
	unsigned int m_numColorChannels;

	uplink::CameraCalibration m_calibrationDepth;
	uplink::CameraCalibration m_calibrationColor;
	bool					  m_bHasCalibration;

	bool					  m_bIsRecording;
	bool					  m_bRecordPressed;

	Image m_feedbackImage; // sent back to app
	uint8* m_feedbackImageData;
	unsigned int m_feedbackWidth, m_feedbackHeight;
};

//------------------------------------------------------------------------------

struct DesktopServerSession : ServerSession
{
    DesktopServerSession (int socketDescriptor, Server* server)
        : ServerSession(socketDescriptor, server)
    {

    }

    DesktopServer& server () { return *downcast<DesktopServer>(&ServerSession::server()); }
};

//------------------------------------------------------------------------------

} // uplink namespace

# include "./desktop-server.hpp"
