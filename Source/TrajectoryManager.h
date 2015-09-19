
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

	TrajectoryManager(unsigned int numMaxImage) {
		m_optmizedTransforms.resize(numMaxImage);
		m_frames.reserve(numMaxImage);
		m_framesSort.reserve(numMaxImage);
		for (unsigned int i = 0; i < numMaxImage; i++) {
			m_frames.push_back(TrajectoryFrame { TrajectoryFrame::NotIntegrated_NoTransform, (unsigned int)-1, mat4f::zero(), m_optmizedTransforms[i] });
		}
		m_numAddedFrames = 0;
		m_numOptimizedFrames = 0;

		//TODO move this to some sort of parameter file
		m_topNActive = 10;
		m_minPoseDist = 0.02f*0.02f;
		m_featureRescaleRotToTrans = 2.0f;

	}


	void addFrame(TrajectoryFrame::TYPE what, const mat4f& transform, unsigned int idx)
	{
		m_frames[idx].type = what;
		m_frames[idx].frameIdx = idx;
		m_frames[idx].integratedTransform = transform;
		m_frames[idx].optimizedTransform = transform;
		m_framesSort.push_back(&m_frames[idx]);
		m_numAddedFrames++;
	}


	//! called by Bundler; whenever the optimization finishes
	void updateOptimizedTransform(float4x4* d_trajectory, unsigned int numFrames) {
		m_numOptimizedFrames = numFrames;
		numFrames = std::min(numFrames, m_numAddedFrames);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_optmizedTransforms.data(), d_trajectory, sizeof(mat4f)*numFrames, cudaMemcpyDeviceToHost));
	}

	void generateUpdateLists()
	{
		unsigned int numFrames = std::min(m_numOptimizedFrames, m_numAddedFrames);

		for (unsigned int i = 0; i < numFrames; i++) {
			TrajectoryFrame& f = m_frames[i];
			if (f.optimizedTransform[0] == -std::numeric_limits<float>::infinity()) {
				//f.type = TrajectoryFrame::Invalid;
				invalidateFrame(i);	//adding it to the deIntragte list
			}
			else {
				//has a valid transform now
				if (f.type == TrajectoryFrame::NotIntegrated_NoTransform)	{
					//adding it to the integrate list
					f.type = TrajectoryFrame::NotIntegrated_WithTransform;
					m_toIntegrateList.push_back(&f);
				}

				vec6f poseOptimized = PoseHelper::MatrixToPose(f.optimizedTransform);
				poseOptimized[0] *= m_featureRescaleRotToTrans;
				poseOptimized[1] *= m_featureRescaleRotToTrans;
				poseOptimized[2] *= m_featureRescaleRotToTrans;

				vec6f poseIntegrated = PoseHelper::MatrixToPose(f.integratedTransform);
				poseIntegrated[0] *= m_featureRescaleRotToTrans;
				poseIntegrated[1] *= m_featureRescaleRotToTrans;
				poseIntegrated[2] *= m_featureRescaleRotToTrans;
				f.dist = (poseIntegrated - poseOptimized) | (poseIntegrated - poseOptimized);
			}
		}

		//TODO could remove Invalids from m_framesSort();
		//TODO only generate a topN list
		auto s = [](const TrajectoryFrame *left, const TrajectoryFrame *right) {
			if (left->type == TrajectoryFrame::Integrated && right->type != TrajectoryFrame::Integrated)	return true;
			if (left->type != TrajectoryFrame::Integrated && right->type == TrajectoryFrame::Integrated)	return false;
			return left->dist > right->dist;
		};
		std::sort(m_framesSort.begin(), m_framesSort.end(), s);
		//m_framesSort.sort(s);

		for (unsigned int i = 0; i < numFrames; i++) {
			const auto *f = m_framesSort[i];
			std::cout << "[" << f->frameIdx << "]" << " " << f->dist;
			std::cout << "\t type: " << f->type;
			std::cout << std::endl;
		}

		for (unsigned int i = (unsigned int)m_toReIntegrateList.size(); i < m_topNActive; i++) {
			auto* f = m_framesSort[i];
			if (f->dist > m_minPoseDist && f->type == TrajectoryFrame::Integrated) {
				f->type = TrajectoryFrame::ReIntegration;
				m_toReIntegrateList.push_back(f);
			}
			else {
				break;
			}
		}
	}


	void confirmIntegration(unsigned int frameIdx) {
		m_frames[frameIdx].type = TrajectoryFrame::Integrated;
	}

	bool getTopFromReIntegrateList(mat4f& oldTransform, mat4f& newTransform, unsigned int& frameIdx) {
		if (m_toReIntegrateList.empty())	return false;
		assert(m_toReIntegrateList.front()->type == TrajectoryFrame::ReIntegration);

		newTransform = m_toReIntegrateList.front()->optimizedTransform;
		frameIdx = m_toReIntegrateList.front()->frameIdx;
		oldTransform = m_toReIntegrateList.front()->integratedTransform;
		m_toReIntegrateList.front()->integratedTransform = newTransform;
		m_toReIntegrateList.pop_front();
		return true;
	}


	bool getTopFromIntegrateList(mat4f& trans, unsigned int& frameIdx) {
		if (m_toIntegrateList.empty())	return false;
		assert(m_toIntegrateList.front()->type == TrajectoryFrame::NotIntegrated_WithTransform);

		trans = m_toIntegrateList.front()->optimizedTransform;
		frameIdx = m_toIntegrateList.front()->frameIdx;
		m_toIntegrateList.front()->integratedTransform = trans;
		m_toIntegrateList.pop_front();
		return true;
	}

	bool getTopFromDeIntegrateList(mat4f& trans, unsigned int& frameIdx) {
		if (m_toDeIntegrateList.empty())	return false;
		assert(m_toDeIntegrateList.front()->type == TrajectoryFrame::Invalid);

		trans = m_toDeIntegrateList.front()->optimizedTransform;
		frameIdx = m_toDeIntegrateList.front()->frameIdx;
		m_toDeIntegrateList.pop_front();
		return true;
	}



private:
	void invalidateFrame(unsigned int frameIdx) {
		assert(m_frames[frameIdx].type != TrajectoryFrame::Invalid);
		TrajectoryFrame::TYPE typeBefore = m_frames[frameIdx].type;
		m_frames[frameIdx].type = TrajectoryFrame::Invalid;

		if (typeBefore == TrajectoryFrame::Integrated) {
			m_toDeIntegrateList.push_back(&m_frames[frameIdx]);
		}
	}

	std::vector<mat4f> m_optmizedTransforms;
	std::vector<TrajectoryFrame>	m_frames;
	std::vector<TrajectoryFrame*>	m_framesSort;
	unsigned int m_numAddedFrames;
	unsigned int m_numOptimizedFrames;

	std::list<TrajectoryFrame*> m_toDeIntegrateList;
	std::list<TrajectoryFrame*> m_toIntegrateList;
	std::list<TrajectoryFrame*> m_toReIntegrateList;


	unsigned int	m_topNActive;				//only keep up to N
	float			m_minPoseDist;				//only change if value is larger than this
	float			m_featureRescaleRotToTrans;	//multiply the angle in the distance metric by this factor
};