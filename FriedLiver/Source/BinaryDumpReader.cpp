
#include "stdafx.h"

#include "BinaryDumpReader.h"
#include "GlobalAppState.h"
#include "PoseHelper.h"

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

	//default
	//m_data.m_CalibrationDepth.m_Intrinsic = mat4f(
	//	525.0f, 0.0f, 319.5f, 0.0f,
	//	0.0f, 525.0f, 239.5f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationDepth.m_IntrinsicInverse = m_data.m_CalibrationDepth.m_Intrinsic.getInverse();
	//m_data.m_CalibrationColor.m_Intrinsic = mat4f(
	//	525.0f, 0.0f, 319.5f, 0.0f,
	//	0.0f, 525.0f, 239.5f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationColor.m_IntrinsicInverse = m_data.m_CalibrationColor.m_Intrinsic.getInverse();
	//fr1
	//m_data.m_CalibrationDepth.m_Intrinsic = mat4f(
	//	591.1f, 0.0f, 331.0f, 0.0f,
	//	0.0f, 590.1f, 234.0f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationDepth.m_IntrinsicInverse = m_data.m_CalibrationDepth.m_Intrinsic.getInverse();
	//m_data.m_CalibrationColor.m_Intrinsic = mat4f(
	//	517.3f, 0.0f, 318.6f, 0.0f,
	//	0.0f, 516.5f, 255.3f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationColor.m_IntrinsicInverse = m_data.m_CalibrationColor.m_Intrinsic.getInverse();
	//fr2
	//m_data.m_CalibrationDepth.m_Intrinsic = mat4f(		// fr2 
	//	580.8f, 0.0f, 308.8f, 0.0f,
	//	0.0f, 581.8f, 253.0f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationDepth.m_IntrinsicInverse = m_data.m_CalibrationDepth.m_Intrinsic.getInverse();
	//m_data.m_CalibrationColor.m_Intrinsic = mat4f(		// fr2 
	//	520.9f, 0.0f, 325.1f, 0.0f,
	//	0.0f, 521.0f, 249.7f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationColor.m_IntrinsicInverse = m_data.m_CalibrationColor.m_Intrinsic.getInverse();
	//fr3
	//m_data.m_CalibrationDepth.m_Intrinsic = mat4f(		// fr3 
	//	567.6f, 0.0f, 324.7f, 0.0f,
	//	0.0f, 570.2f, 250.1f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationDepth.m_IntrinsicInverse = m_data.m_CalibrationDepth.m_Intrinsic.getInverse();
	//m_data.m_CalibrationColor.m_Intrinsic = mat4f(		// fr3 
	//	535.4f, 0.0f, 320.1f, 0.0f,
	//	0.0f, 539.2f, 247.6f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//m_data.m_CalibrationColor.m_IntrinsicInverse = m_data.m_CalibrationColor.m_Intrinsic.getInverse();

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
		//std::cout << "binary dump sequence complete - press space to run again" << std::endl;
		stopReceivingFrames();
		std::cout << "binary dump sequence complete - stopped receiving frames" << std::endl;
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

void BinaryDumpReader::evaluateTrajectory(const std::vector<mat4f>& trajectory) const
{
	std::vector<mat4f> referenceTrajectory = m_data.m_trajectory;
	const size_t numTransforms = std::min(trajectory.size(), referenceTrajectory.size());
	// make sure reference trajectory starts at identity
	mat4f offset = referenceTrajectory.front().getInverse();
	for (unsigned int i = 0; i < referenceTrajectory.size(); i++) referenceTrajectory[i] = offset * referenceTrajectory[i];

	const auto transErr = PoseHelper::evaluateAteRmse(trajectory, referenceTrajectory);
	std::cout << "*********************************" << std::endl;
	std::cout << "ate rmse = " << transErr.first << ", " << transErr.second << std::endl;
	std::cout << "*********************************" << std::endl;
	//{
	//	std::vector<mat4f> optTrajectory = trajectory;
	//	optTrajectory.resize(numTransforms);
	//	referenceTrajectory.resize(numTransforms);
	//	PoseHelper::saveToPoseFile("debug/opt.txt", optTrajectory);
	//	PoseHelper::saveToPoseFile("debug/gt.txt", referenceTrajectory);
	//}
}



#endif
