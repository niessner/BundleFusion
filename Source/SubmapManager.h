#pragma once
#ifndef SUBMAP_MANAGER_H
#define SUBMAP_MAnAGER_H

class SubmapManager {
public:
	CUDACache* currentLocalCache;
	CUDACache* nextLocalCache;
	CUDACache* globalCache; 

	SIFTImageManager* currentLocal;
	SIFTImageManager* nextLocal;
	SIFTImageManager* global;

	std::vector<mat4f> globalTrajectory;
	std::vector<mat4f> completeTrajectory;

	std::vector< std::vector<mat4f> > localTrajectories;

	SubmapManager() {
		currentLocal = NULL;
		nextLocal = NULL;
		global = NULL;
		m_numTotalFrames = 0;
		m_submapSize = 0;
	}
	void init(unsigned int maxNumGlobalImages, unsigned int maxNumLocalImages, unsigned int maxNumKeysPerImage,
		unsigned int submapSize, unsigned int numTotalFrames = (unsigned int)-1)
	{
		// cache
		const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
		const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;
		currentLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages);
		nextLocalCache = new CUDACache(downSampWidth, downSampHeight, maxNumLocalImages);
		globalCache = new CUDACache(downSampWidth, downSampHeight, maxNumGlobalImages);

		// sift manager
		currentLocal = new SIFTImageManager(maxNumLocalImages, maxNumKeysPerImage);
		nextLocal = new SIFTImageManager(maxNumLocalImages, maxNumKeysPerImage);
		global = new SIFTImageManager(maxNumGlobalImages, maxNumKeysPerImage);

		globalTrajectory.push_back(mat4f::identity()); // first transform is the identity

		m_numTotalFrames = numTotalFrames;
		m_submapSize = submapSize;
	}
	void setTotalNumFrames(unsigned int n) {
		m_numTotalFrames = n;
	}
	~SubmapManager() {
		SAFE_DELETE(currentLocal);
		SAFE_DELETE(nextLocal);
		SAFE_DELETE(global);

		SAFE_DELETE(currentLocalCache);
		SAFE_DELETE(nextLocalCache);
		SAFE_DELETE(globalCache);
	}

	// update complete trajectory with new global trajectory info
	void updateTrajectory() {
		MLIB_ASSERT(globalTrajectory.size() > 0);

		unsigned int totalNum = 0;
		if (localTrajectories.back().size() > m_submapSize) totalNum = (unsigned int)globalTrajectory.size() * m_submapSize;
		else totalNum = (unsigned int)(globalTrajectory.size() - 1) * m_submapSize + (unsigned int)localTrajectories.back().size();
		completeTrajectory.resize(totalNum);

		unsigned int idx = 0;
		for (unsigned int i = 0; i < globalTrajectory.size(); i++) {
			const mat4f& baseTransform = globalTrajectory[i];
			for (unsigned int t = 0; t < std::min(m_submapSize, (unsigned int)localTrajectories[i].size()); t++) { // overlap frame
				completeTrajectory[idx++] = baseTransform * localTrajectories[i][t];
			}
		}
	}

	void switchLocal() {
		currentLocal->reset();
		SIFTImageManager* tmp = currentLocal;
		currentLocal = nextLocal;
		nextLocal = tmp;

		currentLocalCache->reset();
		CUDACache* tmpCache = currentLocalCache;
		currentLocalCache = nextLocalCache;
		nextLocalCache = tmpCache;
	}

	bool isLastFrame(unsigned int curFrame) const { return (curFrame + 1) == m_numTotalFrames; }
	bool isLastLocalFrame(unsigned int curFrame) const { return (curFrame >= m_submapSize && (curFrame % m_submapSize) == 0); }
	unsigned int getCurrLocalIdx(unsigned int curFrame) const {
		const unsigned int curLocalIdx = (curFrame + 1 == m_numTotalFrames) ? (curFrame / m_submapSize) : (curFrame / m_submapSize) - 1; // adjust for endframe
		return curLocalIdx;
	}

private:

	unsigned int m_numTotalFrames;
	unsigned int m_submapSize;
};

#endif