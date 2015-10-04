#pragma once

#include "RGBDSensor.h"
#include "Bundler.h"
#include "ConditionManager.h"

int startDepthSensing(Bundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager);


