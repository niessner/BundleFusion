
#include "stdafx.h"

#include "SensorDataReader.h"
#include "GlobalAppState.h"
#include "MatrixConversion.h"
#include "PoseHelper.h"

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

	//std::cout << "depth intrinsics:" << std::endl;
	//std::cout << m_sensorData->m_calibrationDepth.m_intrinsic << std::endl;
	//std::cout << "color intrinsics:" << std::endl;
	//std::cout << m_sensorData->m_calibrationColor.m_intrinsic << std::endl;

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
	m_sensorDataCache = new ml::SensorData::RGBDFrameCacheRead(m_sensorData, cacheSize);
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

		//TODO check why the frame cache is not used?
		ml::SensorData::RGBDFrameCacheRead::FrameState frameState = m_sensorDataCache->getNext();
		//ml::SensorData::RGBDFrameCacheRead::FrameState frameState;
		//frameState.m_colorFrame = m_sensorData->decompressColorAlloc(m_currFrame);
		//frameState.m_depthFrame = m_sensorData->decompressDepthAlloc(m_currFrame);


		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
			if (frameState.m_depthFrame[i] == 0) depth[i] = -std::numeric_limits<float>::infinity();
			else depth[i] = (float)frameState.m_depthFrame[i] / m_sensorData->m_depthShift;
		}

		incrementRingbufIdx();

		if (m_bHasColorData) {
			for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
				m_colorRGBX[i] = vec4uc(frameState.m_colorFrame[i]);
			}
		}
		frameState.free();

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

void SensorDataReader::saveToFile(const std::string& filename, const std::vector<mat4f>& trajectory) const
{
	const unsigned int numFrames = (unsigned int)std::min(trajectory.size(), m_sensorData->m_frames.size());
	for (unsigned int i = 0; i < numFrames; i++) {
		m_sensorData->m_frames[i].setCameraToWorld(trajectory[i]);
	}
	//fill in rest invalid
	mat4f invalidTransform; invalidTransform.setZero(-std::numeric_limits<float>::infinity());
	for (unsigned int i = (unsigned int)trajectory.size(); i < m_sensorData->m_frames.size(); i++) {
		m_sensorData->m_frames[i].setCameraToWorld(invalidTransform);
	}

	m_sensorData->saveToFile(filename);
}

void SensorDataReader::evaluateTrajectory(const std::vector<mat4f>& trajectory) const
{
	std::vector<mat4f> referenceTrajectory;
	for (const auto& f : m_sensorData->m_frames) referenceTrajectory.push_back(f.getCameraToWorld());
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
