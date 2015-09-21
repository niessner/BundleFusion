
#include "stdafx.h"

#include "StructureSensor.h"
#include <iomanip>
#ifdef STRUCTURE_SENSOR

//FrameTimer ExampleServerSession::s_timer;

bool StructureSensor::processDepth()
{
	std::pair<float*,UCHAR*> frames = m_server.process(m_oldDepth, m_oldColor);
	if (frames.first == NULL || frames.second == NULL) return false;

	// depth
	memcpy(m_depthFloat[m_currentRingBufIdx], frames.first, sizeof(float)*getDepthWidth()*getDepthHeight());

	// color
	memcpy(m_colorRGBX, (vec4uc*)frames.second, sizeof(vec4uc)*getColorWidth()*getColorHeight());

	m_oldDepth = frames.first;
	m_oldColor = frames.second;

	return true;
}


#endif