#pragma once

#include "RGBDSensor.h"
#include "TrajectoryManager.h"
#include "OnlineBundler.h"
#include "ConditionManager.h"

int startDepthSensing(OnlineBundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager);


