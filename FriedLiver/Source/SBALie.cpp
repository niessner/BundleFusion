#include "stdafx.h"


#include "SBALie.h"
#ifdef USE_LIE_SPACE
//WARNING broken: needs residual unpacking

#include "TimingLog.h"

LieEnergyTermHelper* SBA::s_energyTermHelperGlobal = NULL;

#define POSESIZE 6

void LieEnergyTermHelper::computeEnergy(const std::vector<mat4f>& transforms)
{
	// set to zero
	m_lastEnergy = 0.0f;

	int size = (int)m_correspondences.size();
	MLIB_ASSERT(size > 0);
	m_residuals.resize(size);

	//#pragma omp parallel for num_threads(7) //TODO undo
	for (int t = 0; t < size; t++)
	{
		const EntryJ& corr = m_correspondences[t];
		if (corr.isValid()) {
			vec3f cpos_i(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z);
			vec3f cpos_j(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
			vec3f wpos_i = transforms[corr.imgIdx_i] * cpos_i; // world space
			vec3f wpos_j = transforms[corr.imgIdx_j] * cpos_j; // world space
			vec3f dist = wpos_i - wpos_j;

			m_residuals[t * 3 + 0] = std::make_pair((unsigned int)t * 3 + 0, dist.x);
			m_residuals[t * 3 + 1] = std::make_pair((unsigned int)t * 3 + 1, dist.y);
			m_residuals[t * 3 + 2] = std::make_pair((unsigned int)t * 3 + 2, dist.z);
		}
		else {
			m_residuals[t * 3 + 0] = std::make_pair((unsigned int)t * 3 + 0, 0.0f);
			m_residuals[t * 3 + 1] = std::make_pair((unsigned int)t * 3 + 1, 0.0f);
			m_residuals[t * 3 + 2] = std::make_pair((unsigned int)t * 3 + 2, 0.0f);
		}
	}

	for (unsigned int i = 0; i < m_numResiduals; i++)
		m_lastEnergy += m_residuals[i].second *  m_residuals[i].second;
}

void LieEnergyTermHelper::computeEnergyAndJacobian(const std::vector<mat4f>& transforms, const std::vector<Pose>& poses)
{
	// set to zero
	m_lastEnergy = 0.0f;
	m_JTJ.setZero();
	m_JTr.setZero();

	int size = (int)m_correspondences.size();
	MLIB_ASSERT(size > 0);
	m_residuals.resize(size);

	//#pragma omp parallel for num_threads(7) //!!!TODO FIX
	for (int t = 0; t < size; t++)
	{
		const EntryJ& corr = m_correspondences[t];
		if (corr.isValid()) {
			vec3f cpos_i(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z);
			vec3f cpos_j(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
			vec3f wpos_i = transforms[corr.imgIdx_i] * cpos_i; // world space
			vec3f wpos_j = transforms[corr.imgIdx_j] * cpos_j; // world space
			vec3f dist = wpos_i - wpos_j;

			int varIdxBase_i = -1;
			int varIdxBase_j = -1;
			JacobianBlock jacobianBlockRow_i, jacobianBlockRow_j;
			if (corr.imgIdx_i > 0) {
				varIdxBase_i = (corr.imgIdx_i - 1) * POSESIZE;
				computeJacobianBlockRow_i(dist, cpos_i, transforms[corr.imgIdx_i], jacobianBlockRow_i);
			}
			if (corr.imgIdx_j > 0) {
				varIdxBase_j = (corr.imgIdx_j - 1) * POSESIZE;
				computeJacobianBlockRow_j(dist, cpos_j, transforms[corr.imgIdx_j], jacobianBlockRow_j);
			}
			addToLocalSystem(jacobianBlockRow_i, jacobianBlockRow_j, varIdxBase_i, varIdxBase_j, dist);

			m_residuals[t * 3 + 0] = std::make_pair((unsigned int)t * 3 + 0, dist.x);
			m_residuals[t * 3 + 1] = std::make_pair((unsigned int)t * 3 + 1, dist.y);
			m_residuals[t * 3 + 2] = std::make_pair((unsigned int)t * 3 + 2, dist.z);

		}
		else {
			m_residuals[t * 3 + 0] = std::make_pair((unsigned int)t * 3 + 0, 0.0f);
			m_residuals[t * 3 + 1] = std::make_pair((unsigned int)t * 3 + 1, 0.0f);
			m_residuals[t * 3 + 2] = std::make_pair((unsigned int)t * 3 + 2, 0.0f);
		}
	} // each pair of images

	for (unsigned int i = 0; i < m_numResiduals; i++)
		m_lastEnergy += m_residuals[i].second *  m_residuals[i].second;
}

void LieEnergyTermHelper::addToLocalSystem(const JacobianBlock& jacobianBlockRow_i, const JacobianBlock& jacobianBlockRow_j,
	int varIdxBase_i, int varIdxBase_j, const vec3f& residual)
{
	for (unsigned int res = 0; res < 3; res++) {
		for (unsigned int i = 0; i < POSESIZE; i++) {
			for (unsigned int j = i; j < POSESIZE; j++) {

				if (varIdxBase_i >= 0) {
					float dii = jacobianBlockRow_i.data[i][res] * jacobianBlockRow_i.data[j][res];
					m_JTJ(varIdxBase_i + i, varIdxBase_i + j) += dii;
					if (i != j) m_JTJ(varIdxBase_i + j, varIdxBase_i + i) += dii;
				}
				if (varIdxBase_j >= 0) {
					float djj = jacobianBlockRow_j.data[i][res] * jacobianBlockRow_j.data[j][res];
					m_JTJ(varIdxBase_j + i, varIdxBase_j + j) += djj;
					if (i != j) m_JTJ(varIdxBase_j + j, varIdxBase_j + i) += djj;
				}
				if (varIdxBase_i >= 0 && varIdxBase_j >= 0) {
					float dij = jacobianBlockRow_i.data[i][res] * jacobianBlockRow_j.data[j][res];
					m_JTJ(varIdxBase_i + i, varIdxBase_j + j) += dij;
					m_JTJ(varIdxBase_j + j, varIdxBase_i + i) += dij;
					if (i != j) {
						float dji = jacobianBlockRow_i.data[j][res] * jacobianBlockRow_j.data[i][res];
						m_JTJ(varIdxBase_i + j, varIdxBase_j + i) += dji;
						m_JTJ(varIdxBase_j + i, varIdxBase_i + j) += dji;
					}
				}

			} // j
			if (varIdxBase_i >= 0) m_JTr(varIdxBase_i + i) += jacobianBlockRow_i.data[i][res] * residual[res];
			if (varIdxBase_j >= 0) m_JTr(varIdxBase_j + i) += jacobianBlockRow_j.data[i][res] * residual[res];
		} // i
	} // res
}

void LieEnergyTermHelper::computejacobianBlockRow_i(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_i, const mat4f& transform_i, JacobianBlock& jacPart_i)
{
	float m;
	if (dist == 0.0f) m = 0.0f;
	else m = 1.0f / dist;
	vec3f diff = wpos_i - wpos_j;

	vec3f px(1.0f, 0.0f, 0.0f);
	vec3f py(0.0f, 1.0f, 0.0f);
	vec3f pz(0.0f, 0.0f, 1.0f);
	vec3f pa(0.0f, -wpos_i.z, wpos_i.y);
	vec3f pb(wpos_i.z, 0.0f, -wpos_i.x);
	vec3f pc(-wpos_i.y, wpos_i.x, 0.0f);

	jacPart_i[0] = m * (diff | px); // d trans x
	jacPart_i[1] = m * (diff | py); // d trans y
	jacPart_i[2] = m * (diff | pz); // d trans z
	jacPart_i[3] = m * (diff | pa); // d rot a
	jacPart_i[4] = m * (diff | pb); // d rot b
	jacPart_i[5] = m * (diff | pc); // d rot c
}

void LieEnergyTermHelper::computejacobianBlockRow_j(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_j, const mat4f& transform_j, JacobianBlock& jacPart_j)
{
	float m;
	if (dist == 0.0f) m = 0.0f;
	else m = 1.0f / dist;
	vec3f diff = wpos_i - wpos_j;

	vec3f px(1.0f, 0.0f, 0.0f);
	vec3f py(0.0f, 1.0f, 0.0f);
	vec3f pz(0.0f, 0.0f, 1.0f);
	vec3f pa(0.0f, -wpos_j.z, wpos_j.y);
	vec3f pb(wpos_j.z, 0.0f, -wpos_j.x);
	vec3f pc(-wpos_j.y, wpos_j.x, 0.0f);

	jacPart_j[0] = -m * (diff | px); // d trans x
	jacPart_j[1] = -m * (diff | py); // d trans y
	jacPart_j[2] = -m * (diff | pz); // d trans z
	jacPart_j[3] = -m * (diff | pa); // d rot a
	jacPart_j[4] = -m * (diff | pb); // d rot b
	jacPart_j[5] = -m * (diff | pc); // d rot c
}


bool LieEnergyTermHelper::useVerification()
{
	//const float maxRes = GlobalAppState::get().s_verifyOptDistThresh;
	//const float percentageThresh = GlobalAppState::get().s_verifyOptPercentThresh;
	//std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
	//	return fabs(left.second) > fabs(right.second);
	//});
	//unsigned int count = 0;
	//for (unsigned int i = 0; i < m_residuals.size(); i++) {
	//	if (fabs(m_residuals[i].second) >= maxRes) count++;
	//	else break;
	//}

	//std::cout << "[ useVerification ] " << count << " / " << m_numResiduals << " = " << (float)count / m_numResiduals << " vs " << percentageThresh << std::endl;
	//if ((float)count / m_numResiduals >= percentageThresh) return true;
	//return false;
	return false;
}

bool LieEnergyTermHelper::findAndRemoveOutliers()
{
	const float MAX_RESIDUAL = 0.05f;

	// sort residuals ascending and determine threshold
	std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
		return fabs(left.second) < fabs(right.second);
	});
	const std::pair<unsigned int, float> &highest = m_residuals.back();
	const vec2ui& id = resIdxToMatchIdx[highest.first / 3];
	if (fabs(highest.second) > MAX_RESIDUAL) { // remove!
		m_bundle->invalidateImagePairMatch(id.x);
		m_bundle->filterFramesByMatches(m_numFrames); // need to re-adjust for removed matches
		m_numFrames = m_bundle->getNumImages();
		return true;
	}
	//std::cout << "\tmax residual from images (" << m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices() << "): " << highest.second << std::endl;
	//m_bundle->getMatches().visualizeMatch("debug/matches/", id.x);

	return false;
}

void SBA::alignInternal(std::vector<mat4f>& transforms, unsigned int maxNumIters)
{
	float eps = 0.00002f;
	float epsg = 0.000001f;

	float prev = std::numeric_limits<float>::infinity();
	unsigned int numVars = m_energyTermHelper->getNumUnknowns();
	unsigned int numResiduals = m_energyTermHelper->getNumResiduals();
	//std::cout << "#vars = " << numVars << ", #residuals = " << numResiduals << std::endl;

	unsigned int iter = 0;
	while (iter < maxNumIters) {

		std::vector<Pose> poses(transforms.size());
		poses[0] = Pose(0.0f); // first is identity
		for (unsigned int i = 1; i < transforms.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(transforms[i]);

		m_energyTermHelper->computeEnergyAndJacobian(transforms, poses);
		float curEnergy = m_energyTermHelper->getLastEnergy();

		std::cout << "[ iteration " << iter << " ] " << curEnergy << std::endl;
		const Eigen::MatrixXf& JTJ = m_energyTermHelper->getJTJ();
		const Eigen::VectorXf& JTr = m_energyTermHelper->getJTr();
		Eigen::VectorXf delta;
		// direct solve
		Eigen::JacobiSVD<Eigen::MatrixXf> SVD(JTJ, Eigen::ComputeFullU | Eigen::ComputeFullV);
		delta = SVD.solve(JTr);

		// update
		std::vector<Pose> curPoses = poses;
		//for (unsigned int i = 1; i < transforms.size(); i++) {
		//	mat4f update = PoseHelper::PoseToMatrix(Pose(delta((i - 1)*POSESIZE + 0),
		//		delta((i - 1)*POSESIZE + 1),
		//		delta((i - 1)*POSESIZE + 2),
		//		delta((i - 1)*POSESIZE + 3),
		//		delta((i - 1)*POSESIZE + 4),
		//		delta((i - 1)*POSESIZE + 5)));
		//	transforms[i] = update * transforms[i];
		//	//transforms[i] = transforms[i] * update;
		//}
		for (unsigned int i = 1; i < transforms.size(); i++) {
			for (unsigned int j = 0; j < POSESIZE; j++)
				poses[i][j] -= delta((i - 1) * 6 + j);
			transforms[i] = PoseHelper::PoseToMatrix(poses[i]);
		}

		float maxDelta = delta(0);
		for (unsigned int i = 0; i < numVars; i++)
			if (delta(i) > maxDelta) maxDelta = delta(i);
		//std::cout << "\tmax delta = " << maxDelta << std::endl;

		if (curEnergy < eps)
			break;
		if (math::abs(prev - curEnergy) < epsg)
			break;
		if (maxDelta < epsg)
			break;
		prev = curEnergy;
		iter++;
	}
}

void SBA::align(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useVerify)
{
	m_bVerify = false;

	setEnergyHelperGlobal(m_energyTermHelper);
	MLIB_ASSERT(transforms.size() == s_energyTermHelperGlobal->getNumFrames());
	std::cout << "[ align ]" << std::endl;

	m_maxResidual = -1.0f;

	do {
		alignInternal(transforms, maxNumIters);
		float finalEnergy = s_energyTermHelperGlobal->getLastEnergy();
		if (finalEnergy == 0) std::cout << "FAILED!\n";
		std::cout << "last energy: " << finalEnergy << std::endl;
	} while (s_energyTermHelperGlobal->findAndRemoveOutliers());
	if (useVerify) m_bVerify = m_energyTermHelper->useVerification();
}
#endif