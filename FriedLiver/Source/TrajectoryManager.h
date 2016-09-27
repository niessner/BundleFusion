
#pragma  once

#include "CUDAImageManager.h"
#include "PoseHelper.h"

class TrajectoryManager {
public:
	struct TrajectoryFrame {
		enum TYPE {
			Integrated = 0,
			NotIntegrated_NoTransform = 1,
			NotIntegrated_WithTransform = 2,
			Invalid = 3,
			ReIntegration = 4
		};

		TYPE type;
		unsigned int frameIdx;
		mat4f integratedTransform;		//camera-to-world transform
		mat4f& optimizedTransform;		//bundling optimized (ref to global array)
		float dist;	//distance between optimized and integrated transform
	};

	TrajectoryManager(unsigned int numMaxImage);


	void addFrame(TrajectoryFrame::TYPE what, const mat4f& transform, unsigned int idx);


	//! called by Bundler; whenever the optimization finishes
	void updateOptimizedTransform(const float4x4* d_trajectory, unsigned int numFrames);

	void generateUpdateLists();


	void confirmIntegration(unsigned int frameIdx);

	bool getTopFromReIntegrateList(mat4f& oldTransform, mat4f& newTransform, unsigned int& frameIdx);
	bool getTopFromIntegrateList(mat4f& trans, unsigned int& frameIdx);
	bool getTopFromDeIntegrateList(mat4f& trans, unsigned int& frameIdx);


	const std::vector<TrajectoryFrame>& getFrames() const;
	unsigned int getNumOptimizedFrames() const;
	unsigned int getNumAddedFrames() const;
	unsigned int getNumActiveOperations() const;


	void getOptimizedTransforms(std::vector<mat4f>& transforms) {
		m_mutexUpdateTransforms.lock();

		unsigned int numFrames = std::min(m_numAddedFrames, m_numOptimizedFrames);
		transforms.resize(numFrames);

		for (unsigned int i = 0; i < numFrames; i++) {
			const auto&f = m_frames[i];
			assert(f.frameIdx == i);
			if (f.type == TrajectoryManager::TrajectoryFrame::Invalid) {
				for (unsigned int k = 0; k < 16; k++) {
					transforms[i][k] = -std::numeric_limits<float>::infinity();
				}
			}
			else {
				transforms[i] = f.optimizedTransform;
			}
		}
		m_mutexUpdateTransforms.unlock();
	}
	void lockUpdateTransforms() {
		m_mutexUpdateTransforms.lock();
	}

	void unlockUpdateTransforms() {
		m_mutexUpdateTransforms.unlock();
	}

	////for debugging
	//void getIntegratedTransforms(std::vector<mat4f>& transforms) {
	//	m_mutexUpdateTransforms.lock();
	//
	//	unsigned int numFrames = m_numAddedFrames;
	//	transforms.resize(numFrames);
	//
	//	for (unsigned int i = 0; i < numFrames; i++) {
	//		const auto&f = m_frames[i];
	//		assert(f.frameIdx == i);
	//		if (f.type == TrajectoryManager::TrajectoryFrame::Invalid) {
	//			for (unsigned int k = 0; k < 16; k++) {
	//				transforms[i][k] = -std::numeric_limits<float>::infinity();
	//			}
	//		}
	//		else {
	//			transforms[i] = f.integratedTransform;
	//		}
	//	}
	//	m_mutexUpdateTransforms.unlock();
	//}
private:
	void invalidateFrame(unsigned int frameIdx);

	std::mutex m_mutexUpdateTransforms;

	std::vector<mat4f> m_optmizedTransforms;
	std::vector<TrajectoryFrame>	m_frames;
	std::vector<TrajectoryFrame*>	m_framesSort;
	unsigned int m_numAddedFrames;
	unsigned int m_numOptimizedFrames;

	std::list<TrajectoryFrame*> m_toDeIntegrateList;
	std::list<TrajectoryFrame*> m_toIntegrateList;
	std::list<TrajectoryFrame*> m_toReIntegrateList;


	unsigned int	m_topNActive;				//only keep up to N
	float			m_minPoseDistSqrt;				//only change if value is larger than this
	float			m_featureRescaleRotToTrans;	//multiply the angle in the distance metric by this factor
};