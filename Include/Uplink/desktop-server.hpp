# include "./desktop-server.h"

# if UPLINK_HAS_DESKTOP_UI
#   include "./desktop-ui.h"
# endif

namespace uplink {

//------------------------------------------------------------------------------

inline
DesktopServer::DesktopServer (const std::string& serviceName, int servicePort, objc_weak ServerDelegate* serverDelegate)
        : Server(serviceName, servicePort, serverDelegate)
        //, _ui(0)
		, m_depthWidth(0), m_depthHeight(0), m_colorWidth(0), m_colorHeight(0), m_numColorChannels(4)
		, m_bHasCalibration(false)
		, m_bRecordPressed(false)
		, m_bIsRecording(false)
		, m_feedbackImageData(NULL), m_feedbackWidth(0), m_feedbackHeight(0)
    {
# if UPLINK_HAS_DESKTOP_UI
        _ui = new DesktopServerUI;
# endif
    }

inline
DesktopServer::~DesktopServer ()
{
    Server::clear();

    // No other threads should call us back, now.

# if UPLINK_HAS_DESKTOP_UI
    zero_delete(_ui);
# endif

	for (unsigned int i = 0; i < m_depthBuffer.size(); i++) {
		SAFE_DELETE_ARRAY(m_depthBuffer[i]);
		SAFE_DELETE_ARRAY(m_colorBuffer[i]);
	}
	SAFE_DELETE_ARRAY(m_feedbackImageData);
}

inline
void DesktopServer::init(unsigned int bufferSize, unsigned int feedbackWidth, unsigned int feedbackHeight)
{
	const MutexLocker _(m_mutex);
	m_depthBuffer.resize(bufferSize, NULL);
	m_colorBuffer.resize(bufferSize, NULL);

	m_depthEmptyList.clear();
	m_colorEmptyList.clear();
	m_depthFilledList.clear();
	m_colorFilledList.clear();

	//m_depthTimestamps.resize(bufferSize, -1.0);
	//m_colorTimestamps.resize(bufferSize, -1.0);

	m_feedbackWidth = feedbackWidth;
	m_feedbackHeight = feedbackHeight;
	m_feedbackImageData = new uplink::uint8[feedbackWidth * feedbackHeight * 3];
	m_feedbackImage.format = ImageFormat_Empty;
	m_feedbackImage.width = 0;
	m_feedbackImage.height = 0;
}

inline
void DesktopServer::updateCalibration(const uplink::CameraCalibration& calibrationDepth, const uplink::CameraCalibration& calibrationColor,
unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const MutexLocker _(m_mutex);
	m_calibrationDepth = calibrationDepth;
	m_calibrationColor = calibrationColor;

	//const float scaleFactor = 1.0f / (float)m_downsampleFactor;
	//// adapt intrinsics
	//m_calibrationColor.fx *= scaleFactor;	m_calibrationColor.cx *= scaleFactor;
	//m_calibrationColor.fy *= scaleFactor;	m_calibrationColor.cy *= scaleFactor;

	{ // initialize
		m_depthWidth = depthWidth;
		m_depthHeight = depthHeight;
		unsigned int depthSize = m_depthWidth * m_depthHeight;
		m_colorWidth = colorWidth;
		m_colorHeight = colorHeight;
		unsigned int colorSize = m_numColorChannels * m_colorWidth * m_colorHeight;
		for (unsigned int i = 0; i < m_depthBuffer.size(); i++) {
			m_depthBuffer[i] = new float[depthSize];
			m_colorBuffer[i] = new uint8[colorSize];
			// everything is empty
			m_depthEmptyList.push_back(m_depthBuffer[i]);
			m_colorEmptyList.push_back(m_colorBuffer[i]);
		}
	}
	if (!m_bHasCalibration) m_bHasCalibration = true;
}

inline
bool DesktopServer::getCalibration(uplink::CameraCalibration& calibrationDepth, uplink::CameraCalibration& calibrationColor,
unsigned int& depthWidth, unsigned int& depthHeight, unsigned int& colorWidth, unsigned int& colorHeight)
{
	const MutexLocker _(m_mutex);
	if (!m_bHasCalibration) return false;
	calibrationDepth = m_calibrationDepth;
	calibrationColor = m_calibrationColor;
	depthWidth = m_depthWidth;
	depthHeight = m_depthHeight;
	colorWidth = m_colorWidth;
	colorHeight = m_colorHeight;
	return true;
}

//------------------------------------------------------------------------------

} // uplink namespace
