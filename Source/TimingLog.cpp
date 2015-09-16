
#include "stdafx.h"
#include "TimingLog.h"


std::vector<TimingLog::FrameTiming> TimingLog::m_localFrameTimings;
std::vector<TimingLog::FrameTiming> TimingLog::m_globalFrameTimings;
std::vector<double> TimingLog::m_totalFrameTimings;
