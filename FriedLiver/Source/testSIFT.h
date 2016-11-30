#pragma once

#include "GlobalBundlingState.h"
#include "GlobalDefines.h"

class SIFTImageManager;

class TestSIFT {
public:

	TestSIFT() {
		free();
	}
	~TestSIFT() {}

	void loadFromSensor(const std::string& filename, unsigned int siftWidth = (unsigned int)-1, unsigned int siftHeight = (unsigned int)-1) {
		std::cout << "loading from file: " << filename << "... ";
		free();

		SensorData sd;
		sd.loadFromFile(filename);
		if (sd.m_frames.size() > 2500) sd.m_frames.resize(2500);

		m_colorIntrinsics = sd.m_calibrationColor.m_intrinsic;
		m_depthIntrinsics = sd.m_calibrationDepth.m_intrinsic;
		if (siftWidth == (unsigned int)-1) siftWidth = sd.m_colorWidth;
		if (siftHeight == (unsigned int)-1) siftHeight = sd.m_colorHeight;

		for (size_t i = 0; i < sd.m_frames.size(); i++) {
			const auto& frame = sd.m_frames[i];
			vec3uc* color = sd.decompressColorAlloc(frame); unsigned short* depth = sd.decompressDepthAlloc(frame);

			m_colorImages.push_back(ColorImageR8G8B8(sd.m_colorWidth, sd.m_colorHeight, color));
			m_depthImages.push_back(DepthImage32(DepthImage16(sd.m_colorWidth, sd.m_colorHeight, depth)));
			std::free(color); std::free(depth);
		}
		setImageSize(siftWidth, siftHeight);
		std::cout << "done!" << std::endl;
	}

	void setImageSize(unsigned int siftWidth, unsigned int siftHeight) {
		if (m_colorImages.empty()) {
			std::cout << "[setImageSize] ERROR: no color images" << std::endl;
			return;
		}
		if (!m_intensityImages.empty() && m_intensityImages.front().getWidth() == siftWidth && m_intensityImages.front().getHeight() == siftHeight) {
			//no need to resize
			return;
		}
		m_intensityImages.clear();
		const unsigned int colorWidth = m_colorImages.front().getWidth();
		const unsigned int colorHeight = m_colorImages.front().getHeight();
		m_intensityIntrinsics = m_colorIntrinsics;
		if (siftWidth != colorWidth || siftHeight != colorHeight) { // adapt intrinsics
			m_intensityIntrinsics._m00 *= (float)siftWidth / (float)colorWidth;
			m_intensityIntrinsics._m11 *= (float)siftHeight / (float)colorHeight;
			 m_intensityIntrinsics._m02 *= (float)(siftWidth-1) / (float)(colorWidth-1);
			m_intensityIntrinsics._m12 *= (float)(siftHeight-1) / (float)(colorHeight-1);
		}

		for (size_t i = 0; i < m_colorImages.size(); i++) {
			ColorImageR32 intensity(colorWidth, colorHeight);
			for (const auto& p : m_colorImages[i]) {
				float v = (0.299f*p.value.x + 0.587f*p.value.y + 0.114f*p.value.z);
				intensity(p.x, p.y) = v;
			}
			if (siftWidth == colorWidth && siftHeight == colorHeight) m_intensityImages.push_back(intensity);
			else m_intensityImages.push_back(intensity.getResized(siftWidth, siftHeight));
		}
	}
	void setTargetNumKeys(unsigned int num) { m_targetNumKeypoints = num; }
	void setMinKeyScale(float s) { m_minKeyScale = s; }

	void test();

private:

	void free() {
		m_intensityImages.clear();
		m_colorImages.clear();
		m_depthImages.clear();

		m_colorIntrinsics.setZero(-std::numeric_limits<float>::infinity());
		m_intensityIntrinsics.setZero(-std::numeric_limits<float>::infinity());
		m_depthIntrinsics.setZero(-std::numeric_limits<float>::infinity());
		m_targetNumKeypoints = 0;
		m_minKeyScale = 0.0f;
	}
	
	std::vector<ColorImageR8G8B8>	m_colorImages;
	std::vector<ColorImageR32>		m_intensityImages;
	std::vector<DepthImage32>		m_depthImages;

	mat4f							m_colorIntrinsics;
	mat4f							m_intensityIntrinsics;
	mat4f							m_depthIntrinsics;

	unsigned int					m_targetNumKeypoints;
	float							m_minKeyScale;
};