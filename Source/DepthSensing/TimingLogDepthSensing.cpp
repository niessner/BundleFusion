
#include "stdafx.h"

#include "TimingLogDepthSensing.h"

double TimingLogDepthSensing::totalTimeHoleFilling = 0.0;
unsigned int TimingLogDepthSensing::countTimeHoleFilling = 0;

double TimingLogDepthSensing::totalTimeRenderMain = 0.0;
unsigned int TimingLogDepthSensing::countTimeRenderMain = 0;

double TimingLogDepthSensing::totalTimeOptimizer = 0.0;
unsigned int TimingLogDepthSensing::countTimeOptimizer = 0;

double TimingLogDepthSensing::totalTimeFilterColor = 0.0;
unsigned int TimingLogDepthSensing::countTimeFilterColor = 0;

double TimingLogDepthSensing::totalTimeFilterDepth = 0.0;
unsigned int TimingLogDepthSensing::countTimeFilterDepth = 0;

double TimingLogDepthSensing::totalTimeRGBDAdapter = 0.0;
unsigned int TimingLogDepthSensing::countTimeRGBDAdapter = 0;

double TimingLogDepthSensing::totalTimeClusterColor = 0.0;
unsigned int TimingLogDepthSensing::countTimeClusterColor = 0;

double TimingLogDepthSensing::totalTimeEstimateLighting = 0.0;
unsigned int TimingLogDepthSensing::countTimeEstimateLighting = 0;

double TimingLogDepthSensing::totalTimeRemapDepth = 0.0;
unsigned int TimingLogDepthSensing::countTimeRemapDepth = 0;

double TimingLogDepthSensing::totalTimeSegment = 0.0;
unsigned int TimingLogDepthSensing::countTimeSegment = 0;

double TimingLogDepthSensing::totalTimeTracking = 0.0;
unsigned int TimingLogDepthSensing::countTimeTracking = 0;

double TimingLogDepthSensing::totalTimeSFS = 0.0;
unsigned int TimingLogDepthSensing::countTimeSFS = 0;

double TimingLogDepthSensing::totalTimeRayCast = 0.0;
unsigned int TimingLogDepthSensing::countTimeRayCast = 0;

double TimingLogDepthSensing::totalTimeRayIntervalSplatting = 0.0;
unsigned int TimingLogDepthSensing::countTimeRayIntervalSplatting = 0;

//double DepthSenginTimingLog::totalTimeRayIntervalSplattingCUDA = 0.0;
//unsigned int DepthSenginTimingLog::countTimeRayIntervalSplattingCUDA = 0;
//
//double DepthSenginTimingLog::totalTimeRayIntervalSplattingDX11 = 0.0;
//unsigned int DepthSenginTimingLog::countTimeRayIntervalSplattingDX11 = 0;

double TimingLogDepthSensing::totalTimeCompactifyHash = 0.0;
unsigned int TimingLogDepthSensing::countTimeCompactifyHash = 0;

double TimingLogDepthSensing::totalTimeAlloc = 0.0;
unsigned int TimingLogDepthSensing::countTimeAlloc = 0;

double TimingLogDepthSensing::totalTimeIntegrate = 0.0;
unsigned int TimingLogDepthSensing::countTimeIntegrate = 0;

double TimingLogDepthSensing::totalTimeDeIntegrate = 0.0;
unsigned int TimingLogDepthSensing::countTimeDeIntegrate = 0;

/////////////
// benchmark
/////////////

double TimingLogDepthSensing::totalTimeAllAvgArray[BENCHMARK_SAMPLES];
unsigned int TimingLogDepthSensing::countTotalTimeAll = 0;
double TimingLogDepthSensing::totalTimeAllWorst = 0.0;
double TimingLogDepthSensing::totalTimeAllMaxAvg = 0.0;
double TimingLogDepthSensing::totalTimeAllMinAvg = 0.0;
double TimingLogDepthSensing::totalTimeAll = 0.0;
double TimingLogDepthSensing::totalTimeSquaredAll = 0.0;
