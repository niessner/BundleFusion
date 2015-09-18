
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
			Invalid = 3
		};

		TYPE type;
		unsigned int frameIdx;
		mat4f integratedTransform;		//camera-to-world transform
		mat4f& optimizedTransform;		//bundling optimized (ref to global array)
		vec6f poseIntegrated;
		float dist;	//distance between optimized and integrated transform
	};

	TrajectoryManager(unsigned int numMaxImage) {
		m_optmizedTransforms.resize(numMaxImage);
		for (unsigned int i = 0; i < numMaxImage; i++) {
			m_frames.push_back(TrajectoryFrame { TrajectoryFrame::NotIntegrated_NoTransform, i, mat4f::zero(), m_optmizedTransforms[i] });
		}
	}


	void addFrame(TrajectoryFrame::TYPE what, const mat4f& transform, unsigned int idx)
	{
		m_frames[idx].type = what;
		m_frames[idx].frameIdx = idx;
		m_frames[idx].integratedTransform = transform;
		m_frames[idx].optimizedTransform = transform;
		m_frames[idx].poseIntegrated = PoseHelper::MatrixToPose(transform);
		m_framesSort.push_back(&m_frames[idx]);
	}


	//! called by Bundler; whenever the optimization finishes
	void updateOptimizedTransform(float4x4* d_trajectory, unsigned int numFrames) {
		if (m_frames.size() < numFrames) throw MLIB_EXCEPTION("We need equal or more frames on the CPU");

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_optmizedTransforms.data(), d_trajectory, sizeof(mat4f)*numFrames, cudaMemcpyDeviceToHost));

		for (unsigned int i = 0; i < numFrames; i++) {
			TrajectoryFrame& f = m_frames[i];
			if (f.optimizedTransform[0] == -std::numeric_limits<float>::infinity()) {
				f.type = TrajectoryFrame::Invalid;
			}
			else {
				//has a valid transform now
				if (f.type == TrajectoryFrame::NotIntegrated_NoTransform)	f.type = TrajectoryFrame::NotIntegrated_WithTransform; //TODO can just add to the integrate list

				vec6f poseOptimized = PoseHelper::MatrixToPose(f.optimizedTransform);
				//TODO think about some scaling here
				f.dist = (f.poseIntegrated - poseOptimized) | (f.poseIntegrated - poseOptimized);
			}			
		}

		//TODO could remove Invalids from m_framesSort();

		auto s = [](const TrajectoryFrame *left, const TrajectoryFrame *right) {
			if (left->type == TrajectoryFrame::NotIntegrated_WithTransform)	return true;		//prioritize not integrated, but with transform
			if (right->type == TrajectoryFrame::NotIntegrated_WithTransform)	return false;	//prioritize not integrated, but with transform
			if (left->type == TrajectoryFrame::Invalid || left->type == TrajectoryFrame::NotIntegrated_NoTransform) return false;		//penalize invalids
			if (right->type == TrajectoryFrame::Invalid || right->type == TrajectoryFrame::NotIntegrated_NoTransform) return true;		//penalize invalids
			
			return left->dist > right->dist;
		};
		m_framesSort.sort(s);
	}

	void invalidateFrame(unsigned int idx) {
		assert(m_frames[idx].type != TrajectoryFrame::Invalid);
		TrajectoryFrame::TYPE typeBefore = m_frames[idx].type;
		m_frames[idx].type = TrajectoryFrame::Invalid;

		if (typeBefore == TrajectoryFrame::Integrated) {
			m_toDeIntegrateList.push_back(&m_frames[idx]);
		}
	}
private:
	std::vector<mat4f> m_optmizedTransforms;
	std::vector<TrajectoryFrame> m_frames;
	std::list<TrajectoryFrame*>  m_framesSort;

	std::list<TrajectoryFrame*> m_toDeIntegrateList;
	std::list<TrajectoryFrame*> m_toIntegrateList;
};