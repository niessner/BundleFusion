#pragma once


/************************************************************************/
/* Reads binary dump data from .sensor files                            */
/************************************************************************/

#include "GlobalAppState.h"
#include "RGBDSensor.h"
#include "stdafx.h"

#ifdef BINARY_DUMP_READER

class BinaryDumpReader : public RGBDSensor
{
public:

	//! Constructor
	BinaryDumpReader();

	//! Destructor; releases allocated ressources
	~BinaryDumpReader();

	//! initializes the sensor
	void createFirstConnected();

	//! reads the next depth frame
	bool processDepth();
	

	bool processColor()	{
		//everything done in process depth since order is relevant (color must be read first)
		return true;
	}

	std::string getSensorName() const {
		//return "BinaryDumpReader";
		return m_data.m_SensorName;
	}

	mat4f getRigidTransform() const {
		if (m_CurrFrame-1 >= m_data.m_trajectory.size()) throw MLIB_EXCEPTION("invalid trajectory index " + std::to_string(m_CurrFrame-1));
		return m_data.m_trajectory[m_CurrFrame-1];
	}

	unsigned int getNumTotalFrames() const {
		return m_NumFrames;
	}

	void stopReceivingFrames() { m_bIsReceivingFrames = false; }

	void evaluateTrajectory(const std::vector<mat4f>& trajectory) const;

private:
	//! deletes all allocated data
	void releaseData();

	CalibratedSensorData m_data;

	unsigned int	m_NumFrames;
	unsigned int	m_CurrFrame;
	bool			m_bHasColorData;

};


#endif
