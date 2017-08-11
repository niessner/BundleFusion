


#include "GlobalAppState.h"

#ifdef KINECT
#pragma comment(lib, "Kinect10.lib")
#endif

#ifdef KINECT_ONE
#pragma comment(lib, "Kinect20.lib")
#endif

#ifdef OPEN_NI
#pragma comment(lib, "OpenNI2.lib")
#endif

#ifdef INTEL_SENSOR
#ifdef _DEBUG
#pragma comment(lib, "DSAPI.dbg.lib")
#else
#pragma comment(lib, "DSAPI.lib")
#endif
#endif

#ifdef REAL_SENSE
#ifdef _DEBUG
#pragma comment(lib, "libpxc_d.lib")
#pragma comment(lib, "libpxcutils_d.lib")
#else
#pragma comment(lib, "libpxc.lib")
#pragma comment(lib, "libpxcutils.lib")
#endif
#endif

#ifdef STRUCTURE_SENSOR
#pragma comment (lib, "Ws2_32.lib")
#pragma comment(lib, "gdiplus.lib")
#endif

#include "RGBDSensor.h"
#include "BinaryDumpReader.h"
//TODO add other sensors here
#include "PrimeSenseSensor.h"
#include "KinectSensor.h"
#include "KinectOneSensor.h"
#include "StructureSensor.h"
#include "SensorDataReader.h"


#include "GlobalBundlingState.h"
#include "TimingLog.h"

#include "SiftGPU/MatrixConversion.h"
#include "SiftGPU/CUDATimer.h"
#include "SiftGPU/SIFTMatchFilter.h"
#include "CUDAImageManager.h"

#include "ConditionManager.h"
#include "DualGPU.h"
#include "OnlineBundler.h"
#include "DepthSensing/DepthSensing.h"


