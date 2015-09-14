
#include "stdafx.h"

#include "BinaryDumpReader.h"
#include "GlobalAppState.h"

#ifdef BINARY_DUMP_READER

#include <algorithm>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <string>

#include <conio.h>

BinaryDumpReader::BinaryDumpReader()
{
	m_NumFrames = 0;
	m_CurrFrame = 0;
	m_bHasColorData = false;
	//parameters are read from the calibration file
}

BinaryDumpReader::~BinaryDumpReader()
{
	releaseData();
}


void BinaryDumpReader::createFirstConnected()
{
	releaseData();

	std::string filename = GlobalAppState::get().s_binaryDumpSensorFile;

	std::cout << "Start loading binary dump" << std::endl;
	//BinaryDataStreamZLibFile inputStream(filename, false);
	BinaryDataStreamFile inputStream(filename, false);
	inputStream >> m_data;
	std::cout << "Loading finished" << std::endl;
	std::cout << m_data << std::endl;

	std::cout << "intrinsics:" << std::endl;
	std::cout << m_data.m_CalibrationDepth.m_Intrinsic << std::endl;

	RGBDSensor::init(m_data.m_DepthImageWidth, m_data.m_DepthImageHeight, std::max(m_data.m_ColorImageWidth,1u), std::max(m_data.m_ColorImageHeight,1u), 1);
	initializeDepthIntrinsics(m_data.m_CalibrationDepth.m_Intrinsic(0,0), m_data.m_CalibrationDepth.m_Intrinsic(1,1), m_data.m_CalibrationDepth.m_Intrinsic(0,2), m_data.m_CalibrationDepth.m_Intrinsic(1,2));
	initializeColorIntrinsics(m_data.m_CalibrationColor.m_Intrinsic(0,0), m_data.m_CalibrationColor.m_Intrinsic(1,1), m_data.m_CalibrationColor.m_Intrinsic(0,2), m_data.m_CalibrationColor.m_Intrinsic(1,2));

	initializeDepthExtrinsics(m_data.m_CalibrationDepth.m_Extrinsic);
	initializeColorExtrinsics(m_data.m_CalibrationColor.m_Extrinsic);


	m_NumFrames = m_data.m_DepthNumFrames;
	assert(m_data.m_ColorNumFrames == m_data.m_DepthNumFrames || m_data.m_ColorNumFrames == 0);		
		
	if (m_data.m_ColorImages.size() > 0) {
		m_bHasColorData = true;
	} else {
		m_bHasColorData = false;
	}
}

bool BinaryDumpReader::processDepth()
{
	if(m_CurrFrame >= m_NumFrames)
	{
		GlobalAppState::get().s_playData = false;
		std::cout << "binary dump sequence complete - press space to run again" << std::endl;
		m_CurrFrame = 0;
	}

	if(GlobalAppState::get().s_playData) {

		float* depth = getDepthFloat();
		memcpy(depth, m_data.m_DepthImages[m_CurrFrame], sizeof(float)*getDepthWidth()*getDepthHeight());

		incrementRingbufIdx();

		if (m_bHasColorData) {
			memcpy(m_colorRGBX, m_data.m_ColorImages[m_CurrFrame], sizeof(vec4uc)*getColorWidth()*getColorHeight());
		}

		m_CurrFrame++;
		return true;
	} else {
		return false;
	}
}

void BinaryDumpReader::releaseData()
{
	m_CurrFrame = 0;
	m_bHasColorData = false;
	m_data.deleteData();
}



#endif
