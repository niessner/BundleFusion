
#include "stdafx.h"

#include "TrajectoryManager.h"     
#include "GlobalAppState.h"


TrajectoryManager::TrajectoryManager(unsigned int numMaxImage)
{
	m_optmizedTransforms.resize(numMaxImage);
	m_frames.reserve(numMaxImage);
	m_framesSort.reserve(numMaxImage);
	for (unsigned int i = 0; i < numMaxImage; i++) {
		m_frames.push_back(TrajectoryFrame{ TrajectoryFrame::NotIntegrated_NoTransform, (unsigned int)-1, mat4f::zero(-std::numeric_limits<float>::infinity()), m_optmizedTransforms[i] });
	}
	m_numAddedFrames = 0;
	m_numOptimizedFrames = 0;

	m_topNActive = GlobalAppState::get().s_topNActive;
	m_minPoseDistSqrt = GlobalAppState::get().s_minPoseDistSqrt;
	m_featureRescaleRotToTrans = 2.0f;
}

void TrajectoryManager::addFrame(TrajectoryFrame::TYPE what, const mat4f& transform, unsigned int idx)
{
	m_frames[idx].type = what;
	m_frames[idx].frameIdx = idx;
	m_frames[idx].integratedTransform = transform;
	m_frames[idx].optimizedTransform = transform;
	m_framesSort.push_back(&m_frames[idx]);
	m_numAddedFrames++;
}

void TrajectoryManager::updateOptimizedTransform(const float4x4* d_trajectory, unsigned int numFrames)
{
	m_mutexUpdateTransforms.lock();

	m_numOptimizedFrames = numFrames;
	numFrames = std::min(numFrames, m_numAddedFrames);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_optmizedTransforms.data(), d_trajectory, sizeof(mat4f)*numFrames, cudaMemcpyDeviceToHost));

	m_mutexUpdateTransforms.unlock();
}

void TrajectoryManager::generateUpdateLists()
{
	m_mutexUpdateTransforms.lock();

	unsigned int numFrames = std::min(m_numOptimizedFrames, m_numAddedFrames);

	for (unsigned int i = 0; i < numFrames; i++) {
		TrajectoryFrame& f = m_frames[i];

		if (f.optimizedTransform[0] == -std::numeric_limits<float>::infinity()) {
			//f.type = TrajectoryFrame::Invalid;
			invalidateFrame(i);	//adding it to the deIntragte list
		}
		else {
			//has a valid transform now / re-validate
			if (f.type == TrajectoryFrame::NotIntegrated_NoTransform || f.type == TrajectoryFrame::Invalid)	{
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
		if (left->type != TrajectoryFrame::Integrated) return false; //needs a strict less than comparison function // && right->type == TrajectoryFrame::Integrated)	return false;
		return left->dist > right->dist;
	};
	std::sort(m_framesSort.begin(), m_framesSort.begin() + numFrames, s);
	//m_framesSort.sort(s);

	//for (unsigned int i = 0; i < std::min(5u, numFrames); i++) {
	//	const auto *f = m_framesSort[i];
	//	std::cout << "[" << f->frameIdx << "]" << " " << f->dist;
	//	std::cout << "\t type: " << f->type;
	//	std::cout << std::endl;
	//}

	for (unsigned int i = (unsigned int)m_toReIntegrateList.size(); i < m_topNActive && i < numFrames; i++) {
		auto* f = m_framesSort[i];
		if (f->dist > m_minPoseDistSqrt && f->type == TrajectoryFrame::Integrated) {
			f->type = TrajectoryFrame::ReIntegration;
			m_toReIntegrateList.push_back(f);
		}
		else {
			break;
		}
	}

	m_mutexUpdateTransforms.unlock();
}

void TrajectoryManager::confirmIntegration(unsigned int frameIdx)
{
	assert(m_frames[frameIdx].type != TrajectoryFrame::Integrated);
	m_frames[frameIdx].type = TrajectoryFrame::Integrated;
}

bool TrajectoryManager::getTopFromReIntegrateList(mat4f& oldTransform, mat4f& newTransform, unsigned int& frameIdx)
{
	if (m_toReIntegrateList.empty())	return false;

	m_mutexUpdateTransforms.lock();
	while (!m_toReIntegrateList.empty()) { // some may have been invalidated in the meantime by updateOptimizedTransforms
		TrajectoryFrame* f = m_toReIntegrateList.front();
		newTransform = f->optimizedTransform;
		frameIdx = f->frameIdx;
		oldTransform = f->integratedTransform;
		m_toReIntegrateList.pop_front();
		if (newTransform[0] != -std::numeric_limits<float>::infinity()) {
			assert(f->type == TrajectoryFrame::ReIntegration);
			f->integratedTransform = newTransform;
			break;
		} // otherwise will be added to the deintegrate list next time
	}
	m_mutexUpdateTransforms.unlock();
	return true;
}

bool TrajectoryManager::getTopFromIntegrateList(mat4f& trans, unsigned int& frameIdx)
{
	if (m_toIntegrateList.empty())	return false;
	if (m_toIntegrateList.front()->type != TrajectoryFrame::NotIntegrated_WithTransform) {
		std::cout << "ERROR NEED TO CHECK FOR INVALIDATE INTEGRATE LIST ELEMENTS" << std::endl;
		getchar();
	}
	assert(m_toIntegrateList.front()->type == TrajectoryFrame::NotIntegrated_WithTransform);

	m_mutexUpdateTransforms.lock();
	trans = m_toIntegrateList.front()->optimizedTransform;
	frameIdx = m_toIntegrateList.front()->frameIdx;
	m_toIntegrateList.front()->integratedTransform = trans;
	m_toIntegrateList.pop_front();
	m_mutexUpdateTransforms.unlock();
	return true;
}

bool TrajectoryManager::getTopFromDeIntegrateList(mat4f& trans, unsigned int& frameIdx)
{
	if (m_toDeIntegrateList.empty())	return false;
	assert(m_toDeIntegrateList.front()->type == TrajectoryFrame::Invalid);

	m_mutexUpdateTransforms.lock();
	trans = m_toDeIntegrateList.front()->integratedTransform;
	frameIdx = m_toDeIntegrateList.front()->frameIdx;
	m_toDeIntegrateList.pop_front();
	m_mutexUpdateTransforms.unlock();
	return true;
}

const std::vector<TrajectoryManager::TrajectoryFrame>& TrajectoryManager::getFrames() const
{
	return m_frames;
}

unsigned int TrajectoryManager::getNumOptimizedFrames() const
{
	return m_numOptimizedFrames;
}

unsigned int TrajectoryManager::getNumAddedFrames() const
{
	return m_numAddedFrames;
}

unsigned int TrajectoryManager::getNumActiveOperations() const
{
	return
		(unsigned int)m_toDeIntegrateList.size() +
		(unsigned int)m_toIntegrateList.size() +
		(unsigned int)m_toReIntegrateList.size();
}

void TrajectoryManager::invalidateFrame(unsigned int frameIdx)
{
	if (m_frames[frameIdx].type == TrajectoryFrame::Invalid) return;
	TrajectoryFrame::TYPE typeBefore = m_frames[frameIdx].type;
	m_frames[frameIdx].type = TrajectoryFrame::Invalid;

	if (typeBefore == TrajectoryFrame::Integrated) {
		m_toDeIntegrateList.push_back(&m_frames[frameIdx]);
	}
}
