
#include "stdafx.h"

#include "SensorDataReader.h"
#include "GlobalAppState.h"
#include "MatrixConversion.h"

#ifdef SENSOR_DATA_READER

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>

#include <conio.h>

SensorDataReader::SensorDataReader()
{
	m_numFrames = 0;
	m_currFrame = 0;
	m_bHasColorData = false;
	//parameters are read from the calibration file

	m_sensorData = NULL;
	m_sensorDataCache = NULL;
}

SensorDataReader::~SensorDataReader()
{
	releaseData();
}


void SensorDataReader::createFirstConnected()
{
	releaseData();

	std::string filename = GlobalAppState::get().s_binaryDumpSensorFile;

	std::cout << "Start loading binary dump... ";
	m_sensorData = new SensorData;
	m_sensorData->loadFromFile(filename);
	std::cout << "DONE!" << std::endl;
	std::cout << *m_sensorData << std::endl;

	//std::cout << "intrinsics:" << std::endl;
//	std::cout << m_sensorData->m_calibrationDepth.m_intrinsic << std::endl;

	RGBDSensor::init(m_sensorData->m_depthWidth, m_sensorData->m_depthHeight, std::max(m_sensorData->m_colorWidth, 1u), std::max(m_sensorData->m_colorHeight, 1u), 1);
	initializeDepthIntrinsics(m_sensorData->m_calibrationDepth.m_intrinsic(0, 0), m_sensorData->m_calibrationDepth.m_intrinsic(1, 1), m_sensorData->m_calibrationDepth.m_intrinsic(0, 2), m_sensorData->m_calibrationDepth.m_intrinsic(1, 2));
	initializeColorIntrinsics(m_sensorData->m_calibrationColor.m_intrinsic(0, 0), m_sensorData->m_calibrationColor.m_intrinsic(1, 1), m_sensorData->m_calibrationColor.m_intrinsic(0, 2), m_sensorData->m_calibrationColor.m_intrinsic(1, 2));

	initializeDepthExtrinsics(m_sensorData->m_calibrationDepth.m_extrinsic);
	initializeColorExtrinsics(m_sensorData->m_calibrationColor.m_extrinsic);


	m_numFrames = (unsigned int)m_sensorData->m_frames.size();

	if (m_numFrames > 0 && m_sensorData->m_frames[0].getColorCompressed()) {
		m_bHasColorData = true;
	}
	else {
		m_bHasColorData = false;
	}

	const unsigned int cacheSize = 10;
	//m_sensorDataCache = new RGBDFrameCacheRead(m_sensorData, cacheSize);
}

bool SensorDataReader::processDepth()
{
	if (m_currFrame >= m_numFrames)
	{
		GlobalAppState::get().s_playData = false;
		//std::cout << "binary dump sequence complete - press space to run again" << std::endl;
		stopReceivingFrames();
		std::cout << "binary dump sequence complete - stopped receiving frames" << std::endl;
		m_currFrame = 0;
	}

	if (GlobalAppState::get().s_playData) {

		float* depth = getDepthFloat();
		//memcpy(depth, m_data.m_DepthImages[m_currFrame], sizeof(float)*getDepthWidth()*getDepthHeight());

		//TODO check why the frame cache is not used?
		//ml::RGBDFrameCacheRead::FrameState frameState = m_sensorDataCache->getNext();
		ml::SensorData::RGBDFrameCacheRead::FrameState frameState;
		frameState.m_colorFrame = m_sensorData->decompressColorAlloc(m_currFrame);
		frameState.m_depthFrame = m_sensorData->decompressDepthAlloc(m_currFrame);


		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
			depth[i] = (float)frameState.m_depthFrame[i] / m_sensorData->m_depthShift;
		}

		incrementRingbufIdx();

		if (m_bHasColorData) {
			//memcpy(m_colorRGBX, m_data.m_ColorImages[m_currFrame], sizeof(vec4uc)*getColorWidth()*getColorHeight());
			for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
				m_colorRGBX[i] = vec4uc(frameState.m_colorFrame[i]);
			}
		}
		frameState.free();

		//if (m_currFrame == 50) {
		//	m_sensorDataCache->endDecompression();
		//	std::cout << "END END" << std::endl;
		//	getchar();
		//}
		m_currFrame++;
		return true;
	}
	else {
		return false;
	} 
}

std::string SensorDataReader::getSensorName() const
{
	return m_sensorData->m_sensorName;
}

ml::mat4f SensorDataReader::getRigidTransform(int offset) const
{
	unsigned int idx = m_currFrame - 1 + offset;
	if (idx >= m_sensorData->m_frames.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(idx));
	const mat4f& transform = m_sensorData->m_frames[idx].getCameraToWorld();
	return transform;
	//return m_data.m_trajectory[idx];
}

void SensorDataReader::releaseData()
{
	m_currFrame = 0;
	m_bHasColorData = false;

	 
	SAFE_DELETE(m_sensorDataCache);
	if (m_sensorData) {
		m_sensorData->free();
		SAFE_DELETE(m_sensorData);
	}
}



#endif
