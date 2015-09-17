#pragma once

#include "RGBDSensor.h"
#include "Bundler.h"

int startDepthSensing(Bundler* bundler, RGBDSensor* sensor, CUDAImageManager* imageManager);


