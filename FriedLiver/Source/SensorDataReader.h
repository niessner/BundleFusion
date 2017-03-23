#pragma once


/************************************************************************/
/* Reads sensor data files from .sens files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"
#include "stdafx.h"

#ifdef SENSOR_DATA_READER

//namespace ml {
//	class SensorData;
//	class RGBDFrameCacheRead;
//}

class SensorDataReader : public RGBDSensor
{
public:

	//! Constructor
	SensorDataReader();

	//! Destructor; releases allocated ressources
	~SensorDataReader();

	//! initializes the sensor
	void createFirstConnected();

	//! reads the next depth frame
	bool processDepth();


	bool processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return true;
	}

	std::string getSensorName() const;

	mat4f getRigidTransform(int offset) const;


	void stopReceivingFrames() { m_bIsReceivingFrames = false; }

	//kind of a hack
	void saveToFile(const std::string& filename, const std::vector<mat4f>& trajectory) const;

	void evaluateTrajectory(const std::vector<mat4f>& trajectory) const;
	void getTrajectory(std::vector<mat4f>& trajectory) const;

private:
	//! deletes all allocated data
	void releaseData();

	ml::SensorData* m_sensorData;
	ml::SensorData::RGBDFrameCacheRead* m_sensorDataCache;

	unsigned int	m_numFrames;
	unsigned int	m_currFrame;
	bool			m_bHasColorData;

};


#endif	//sensor data reader
