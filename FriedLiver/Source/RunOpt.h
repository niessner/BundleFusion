#pragma once

#include "GlobalBundlingState.h"
#include "GlobalDefines.h"

class SIFTImageManager;


class RunOpt {
public:
	RunOpt() {}
	~RunOpt() {}

	static void run(const std::string& sensFile, const std::string& siftFile, const std::string& trajectoryFile);

private:
	static void loadFromSensFile(const std::string& sensFile, unsigned int submapSize,
		std::vector<DepthImage32>& depthImages, std::vector<ColorImageR8G8B8>& colorImages,
		mat4f& depthIntrinsics, mat4f& colorIntrinsics) 
	{
		if (!(util::getFileExtension(sensFile) == "sens")) throw MLIB_EXCEPTION("invalid file type " + sensFile + " for sensorData");

		std::cout << "loading frames from sens... ";
		SensorData sensorData;
		sensorData.loadFromFile(sensFile);
		SensorData::RGBDFrameCacheRead sensorDataCache(&sensorData, 10);
		const unsigned int numOrigFrames = (unsigned int)sensorData.m_frames.size();
		const unsigned int numFrames = numOrigFrames / submapSize;

		depthIntrinsics = sensorData.m_calibrationDepth.m_intrinsic;
		colorIntrinsics = sensorData.m_calibrationColor.m_intrinsic;

		colorImages.resize(numFrames);
		depthImages.resize(numFrames);
		for (unsigned int i = 0; i < numFrames; i++) {
			const unsigned int oldIndex = i * submapSize;
			MLIB_ASSERT(oldIndex < numOrigFrames);
			const auto& frame = sensorData.m_frames[oldIndex];
			vec3uc* colordata = sensorData.decompressColorAlloc(frame);
			unsigned short* depthdata = sensorData.decompressDepthAlloc(frame);
			colorImages[i] = ColorImageR8G8B8(sensorData.m_colorWidth, sensorData.m_colorHeight, colordata);
			depthImages[i] = DepthImage32(DepthImage16(sensorData.m_depthWidth, sensorData.m_depthHeight, depthdata));
			std::free(colordata);	std::free(depthdata);
		}

		std::cout << "done! (" << colorImages.size() << " of " << sensorData.m_frames.size() << ")" << std::endl;
		std::cout << "depth intrinsics:" << std::endl << depthIntrinsics << std::endl;
		std::cout << "color intrinsics:" << std::endl << colorIntrinsics << std::endl;
	}

};