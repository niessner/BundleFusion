#include "stdafx.h"
#include "SBA_CPU.h"

#ifdef USE_CPU_SOLVE

SBAEnergyTermHelper* SBA::s_energyTermHelperGlobal = NULL;
Timer SBA::m_timer;

#define POSESIZE 6

void SBAEnergyTermHelper::computeEnergy(const std::vector<mat4f>& transforms)
{
	// set to zero
	m_lastEnergy = 0.0f;

	const Matches& matches = m_bundle->getMatches();
	int size = (int)matches.size();
	MLIB_ASSERT(size > 0);

#ifdef DEBUG_JAC_RES
	for (unsigned int r = 0; r < m_numResiduals; r++)
		d_residuals(r) = 0.0f;
#endif

#pragma omp parallel for num_threads(7)
	for (int t = 0; t < size; t++)
	{
		const ImagePairMatchSet& set = matches.getImagePairMatchSet(t);
		const vec2i& imageIndices = set.getImageIndices();
		if (imageIndices.x < (int)m_numFrames && imageIndices.y < (int)m_numFrames) {
			for (unsigned int m = 0; m < set.getNumMatches(); m++)
			{
				// back-project keypoints
				vec2ui keyPointIndices = set.getKeyPointIndices(m);

				vec3f cpos_i = m_bundle->getKeyPointCameraSpacePosition(imageIndices.x, keyPointIndices.x); // camera space
				vec3f cpos_j = m_bundle->getKeyPointCameraSpacePosition(imageIndices.y, keyPointIndices.y); // camera space
				vec3f wpos_i = transforms[imageIndices.x] * cpos_i; // world space
				vec3f wpos_j = transforms[imageIndices.y] * cpos_j; // world space
				vec3f dist = wpos_i - wpos_j;

				unsigned int resIdx = matchIdxToResIdx[vec2ui(t, m)] * 3;
				m_residuals[resIdx + 0] = std::make_pair(resIdx, dist.x);
				m_residuals[resIdx + 1] = std::make_pair(resIdx, dist.y);
				m_residuals[resIdx + 2] = std::make_pair(resIdx, dist.z);

#ifdef DEBUG_JAC_RES
				d_residuals[resIdx + 0] = dist.x;
				d_residuals[resIdx + 1] = dist.y;
				d_residuals[resIdx + 2] = dist.z;
#endif
			} // keypoints in track
		} // if frame to optimize
	} // each pair of images

	for (unsigned int i = 0; i < m_numResiduals; i++)
		m_lastEnergy += m_residuals[i].second *  m_residuals[i].second;
}

void SBAEnergyTermHelper::computeEnergyAndJacobian(const std::vector<mat4f>& transforms, const std::vector<Pose>& poses)
{
	// set to zero
	m_lastEnergy = 0.0f;
	m_JTJ.setZero();
	m_JTr.setZero();

#ifndef USE_LIE_SPACE
	std::vector<MatrixDerivatives> derivs(transforms.size());
	for (unsigned int i = 0; i < transforms.size(); i++)
		derivs[i].compute(poses[i]);
#endif

	const Matches& matches = m_bundle->getMatches();
	int size = (int)matches.size();
	MLIB_ASSERT(size > 0);

#ifdef DEBUG_JAC_RES
	for (unsigned int r = 0; r < m_numResiduals; r++) {
		for (unsigned int v = 0; v < m_numVars; v++)
			d_jacobian(r, v) = 0.0f;
		d_residuals(r) = 0.0f;
	}
#endif

	//#pragma omp parallel for num_threads(7) //!!!TODO FIX
	for (int t = 0; t < size; t++)
	{
		const ImagePairMatchSet& set = matches.getImagePairMatchSet(t);
		const vec2i& imageIndices = set.getImageIndices();
		if (imageIndices.x < (int)m_numFrames && imageIndices.y < (int)m_numFrames) {
			for (unsigned int m = 0; m < set.getNumMatches(); m++)
			{
				// back-project keypoints
				const vec2ui keyPointIndices = set.getKeyPointIndices(m);

				const vec3f cpos_i = m_bundle->getKeyPointCameraSpacePosition(imageIndices.x, keyPointIndices.x); // camera space
				const vec3f cpos_j = m_bundle->getKeyPointCameraSpacePosition(imageIndices.y, keyPointIndices.y); // camera space
				const vec3f wpos_i = transforms[imageIndices.x] * cpos_i; // world space
				const vec3f wpos_j = transforms[imageIndices.y] * cpos_j; // world space
				vec3f dist = wpos_i - wpos_j;

				int varIdxBase_i = -1;
				int varIdxBase_j = -1;
				JacobianBlock jacobianBlockRow_i, jacobianBlockRow_j;
				if (imageIndices.x > 0) {
					varIdxBase_i = (imageIndices.x - 1) * POSESIZE;
#ifdef USE_LIE_SPACE
					computeJacobianBlockRow_i(dist, cpos_i, transforms[imageIndices.x], jacobianBlockRow_i);
#else
					computeJacobianBlockRow_i(cpos_i, dist, derivs[imageIndices.x], jacobianBlockRow_i);
#endif
				}
				if (imageIndices.y > 0) {
					varIdxBase_j = (imageIndices.y - 1) * POSESIZE;
#ifdef USE_LIE_SPACE
					computeJacobianBlockRow_j(dist, cpos_j, transforms[imageIndices.y], jacobianBlockRow_j);
#else				   
					computeJacobianBlockRow_j(cpos_j, dist, derivs[imageIndices.y], jacobianBlockRow_j);
#endif
				}
				addToLocalSystem(jacobianBlockRow_i, jacobianBlockRow_j, varIdxBase_i, varIdxBase_j, dist);

				unsigned int resIdx = matchIdxToResIdx[vec2ui(t, m)] * 3;
				m_residuals[resIdx + 0] = std::make_pair(resIdx, dist.x);
				m_residuals[resIdx + 1] = std::make_pair(resIdx, dist.y);
				m_residuals[resIdx + 2] = std::make_pair(resIdx, dist.z);

#ifdef DEBUG_JAC_RES
				if (varIdxBase_i >= 0)
					for (unsigned int i = 0; i < POSESIZE; i++) for (unsigned int r = 0; r < 3; r++) d_jacobian(resIdx + r, varIdxBase_i + i) = jacobianBlockRow_i.data[i][r];
				if (varIdxBase_j >= 0)
					for (unsigned int i = 0; i < POSESIZE; i++) for (unsigned int r = 0; r < 3; r++) d_jacobian(resIdx + r, varIdxBase_j + i) = jacobianBlockRow_j.data[i][r];
				d_residuals[resIdx + 0] = dist.x;
				d_residuals[resIdx + 1] = dist.y;
				d_residuals[resIdx + 2] = dist.z;
#endif
			} // keypoints in track
		} // if frame to optimize
	} // each pair of images

	for (unsigned int i = 0; i < m_numResiduals; i++)
		m_lastEnergy += m_residuals[i].second *  m_residuals[i].second;

	//computeJTJStatistics("debug/convergence/j_stats.txt");
	//std::cout << "waiting..." << std::endl;
	//getchar();
	//std::vector< std::vector<float> > debugJTJ(m_numVars);
	//std::vector<float> debugJtr(m_numVars);
	//for (unsigned int i = 0; i < m_numVars; i++) {
	//	debugJTJ[i].resize(m_numVars);
	//	for (unsigned int jali = 0; j < m_numVars; j++)
	//		debugJTJ[i][j] = m_JTJ(i, j);
	//	debugJtr[i] = m_JTr(i);
	//}
	//int a = 5;
}

void SBAEnergyTermHelper::addToLocalSystem(const JacobianBlock& jacobianBlockRow_i, const JacobianBlock& jacobianBlockRow_j, int varIdxBase_i, int varIdxBase_j, const vec3f& residual)
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

#ifdef USE_LIE_SPACE
void SBAEnergyTermHelper::computejacobianBlockRow_i(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_i, const mat4f& transform_i, JacobianBlock& jacPart_i)
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

void SBAEnergyTermHelper::computejacobianBlockRow_j(const vec3f& wpos_i, const vec3f& wpos_j, float dist, const vec3f& cpos_j, const mat4f& transform_j, JacobianBlock& jacPart_j)
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
#else
void SBAEnergyTermHelper::computeJacobianBlockRow_i(const vec3f& cpos_i, const vec3f& dist, MatrixDerivatives& deriv_i, JacobianBlock& jacPart_i)
{
	for (unsigned int i = 0; i < POSESIZE; i++)
		jacPart_i.data[i] = deriv_i.dPose[i] * cpos_i;
}

void SBAEnergyTermHelper::computeJacobianBlockRow_j(const vec3f& cpos_j, const vec3f& dist, MatrixDerivatives& deriv_j, JacobianBlock& jacPart_j)
{
	for (unsigned int i = 0; i < POSESIZE; i++)
		jacPart_j.data[i] = -(deriv_j.dPose[i] * cpos_j);
}
#endif

bool SBAEnergyTermHelper::useVerification()
{
	const float maxRes = GlobalAppState::get().s_verifyOptDistThresh;
	const float percentageThresh = GlobalAppState::get().s_verifyOptPercentThresh;
	std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
		return fabs(left.second) > fabs(right.second);
	});
	unsigned int count = 0;
	for (unsigned int i = 0; i < m_residuals.size(); i++) {
		if (fabs(m_residuals[i].second) >= maxRes) count++;
		else break;
	}

	std::cout << "[ useVerification ] " << count << " / " << m_numResiduals << " = " << (float)count / m_numResiduals << " vs " << percentageThresh << std::endl;
	if ((float)count / m_numResiduals >= percentageThresh) return true;
	return false;
}

bool SBAEnergyTermHelper::findAndRemoveOutliers()
{
	const float MAX_RESIDUAL = 0.05f;

	// sort residuals ascending and determine threshold
	std::sort(m_residuals.begin(), m_residuals.end(), [](const std::pair<unsigned int, float> &left, const std::pair<unsigned int, float> &right) {
		return fabs(left.second) < fabs(right.second);
	});
	const std::pair<unsigned int, float> &highest = m_residuals.back();
	const vec2ui& id = resIdxToMatchIdx[highest.first / 3];
	TimingLog::maxComponentResiduals.push_back(std::make_pair(m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices(), highest.second));
	if (fabs(highest.second) > MAX_RESIDUAL) { // remove!
		//!!!
		//if (m_bundle->getMatches().isValid(id.x)) {
		//	std::cout << "\timages (" << m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices() << "): invalid match " << highest.second << std::endl;
		//	m_bundle->getMatches().visualizeMatch("debug/invalidated/", id.x);
		//}
		//!!!
		m_bundle->invalidateImagePairMatch(id.x);
		m_bundle->filterFramesByMatches(m_numFrames); // need to re-adjust for removed matches
		m_numFrames = m_bundle->getNumImages();
		return true;
	}
	//std::cout << "\tmax residual from images (" << m_bundle->getMatches().getImagePairMatchSet(id.x).getImageIndices() << "): " << highest.second << std::endl;
	//m_bundle->getMatches().visualizeMatch("debug/matches/", id.x);

	return false;
}

void SBAEnergyTermHelper::computeJTJStatistics(const std::string& filename) const
{
	unsigned int numVarFrames = m_numVars / POSESIZE;
	std::vector<unsigned int> numCorrPerVariable(numVarFrames, 0);
	const Matches& matches = m_bundle->getMatches();
	for (unsigned int i = 0; i < matches.size(); i++) {
		const ImagePairMatchSet& set = matches.getImagePairMatchSet(i);
		if (set.getNumMatches() > 0) {
			const vec2i& imageIndices = set.getImageIndices();
			numCorrPerVariable[imageIndices.x] += set.getNumMatches();
			numCorrPerVariable[imageIndices.y] += set.getNumMatches();
		}
	}
	float avgNumCorrPerVariable = (float)std::accumulate(numCorrPerVariable.begin(), numCorrPerVariable.end(), 0) / (float)numCorrPerVariable.size();
	std::sort(numCorrPerVariable.begin(), numCorrPerVariable.end());
	unsigned int minNumCorrPerVariable = numCorrPerVariable.front(); unsigned int maxNumCorrPerVariable = numCorrPerVariable.back();
	unsigned int medNumCorrPerVariable = numCorrPerVariable[numCorrPerVariable.size() / 2];

	std::vector<unsigned int> numNonZeroPerRow(m_numVars, 0);
	unsigned int numNonZero = 0;
	std::vector<float> sumNonDiagonalPerRow(m_numVars, 0.0f);
	for (unsigned int i = 0; i < m_numVars; i++) {
		for (unsigned int j = 0; j < m_numVars; j++) {
			if (m_JTJ(i, j) != 0) {
				numNonZero++;
				numNonZeroPerRow[i]++;
			}
			if (i != j) sumNonDiagonalPerRow[i] += fabs(m_JTJ(i, j));
		}
	}
	std::vector<float> diagRatio(m_numVars); unsigned int diagCount = 0;
	for (unsigned int i = 0; i < m_numVars; i++) {
		float d = fabs(m_JTJ(i, i));
		diagRatio[i] = d / sumNonDiagonalPerRow[i];
		if (diagRatio[i] >= 1) diagCount++;
	}
	std::ofstream s(filename);
	s << "#frames = " << m_bundle->getNumImages() << ", #vars = " << m_numVars << ", #residuals = " << m_numResiduals << " (" << m_numResiduals / 3 << ")" << std::endl;
	// #non-zero
	s << "#non-zeros = " << numNonZero << " of " << m_numVars << "^2 = " << (m_numVars*m_numVars) <<  " | ratio: " << (float)numNonZero/(float)(m_numVars*m_numVars) << std::endl;
	s << "#non-zeros per row:" << std::endl;
	std::sort(numNonZeroPerRow.begin(), numNonZeroPerRow.end());
	s << "\tavg: " << (float)std::accumulate(numNonZeroPerRow.begin(), numNonZeroPerRow.end(), 0) / (float)numNonZeroPerRow.size() << std::endl;
	s << "\tmax: " << numNonZeroPerRow.back() << std::endl;
	s << "\tmin: " << numNonZeroPerRow.front() << std::endl;
	s << "\tmed: " << numNonZeroPerRow[numNonZeroPerRow.size() / 2] << std::endl;
	s << "\t"; for (unsigned int i = 0; i < numNonZeroPerRow.size(); i++) s << numNonZeroPerRow[i] << " ";
	s << std::endl << std::endl;;
	// diagonal dominance
	s << "diagonal dominance:" << std::endl;
	s << "\t#greater: " << diagCount << std::endl;
	s << "\tratios:"; for (unsigned int i = 0; i < diagRatio.size(); i++) s << diagRatio[i] << " ";
	s << std::endl;
	s << "\tsum non-diag per row:"; for (unsigned int i = 0; i < sumNonDiagonalPerRow.size(); i++) s << sumNonDiagonalPerRow[i] << " ";
	s << std::endl;
	s << "\tdiagonal:"; for (unsigned int i = 0; i < m_numVars; i++) s << m_JTJ(i, i) << " ";
	s << std::endl << std::endl;
	// #corr per frame
	s << "#corr per frame/var (jacobian):" << std::endl;
	s << "\tavg: " << avgNumCorrPerVariable << std::endl;
	s << "\tmax: " << maxNumCorrPerVariable << std::endl;
	s << "\tmin: " << minNumCorrPerVariable << std::endl;
	s << "\tmed: " << medNumCorrPerVariable << std::endl;
	s << "\t"; for (unsigned int i = 0; i < numCorrPerVariable.size(); i++) s << numCorrPerVariable[i] << " ";
	s << std::endl << std::endl;
	s.close();
}


void SBA::alignInternal(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useCPUPCG)
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
		if (useCPUPCG) {
			//Eigen::ConjugateGradient<Eigen::MatrixXf, Eigen::Lower, Eigen::DiagonalPreconditioner<float>> cg;
			Eigen::ConjugateGradient<Eigen::MatrixXf> cg;
			cg.compute(JTJ);
			//cg.setMaxIterations(10);
			delta = cg.solve(JTr);
			std::cout << "#iterations:     " << cg.iterations() << std::endl;
			//std::cout << "estimated error: " << cg.error() << std::endl;
			//getchar();
		}
		else { // direct solve
			Eigen::JacobiSVD<Eigen::MatrixXf> SVD(JTJ, Eigen::ComputeFullU | Eigen::ComputeFullV);
			delta = SVD.solve(JTr);
		}

		// update
		std::vector<Pose> curPoses = poses;
#ifdef USE_LIE_SPACE
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
#else
		for (unsigned int i = 1; i < transforms.size(); i++) {
			for (unsigned int j = 0; j < POSESIZE; j++)
				poses[i][j] -= delta((i - 1) * POSESIZE + j);
			transforms[i] = PoseHelper::PoseToMatrix(poses[i]);
		}
#endif
		float maxDelta = delta(0);
		for (unsigned int i = 0; i < numVars; i++)
			if (delta(i) > maxDelta) maxDelta = delta(i);
		//std::cout << "\tmax delta = " << maxDelta << std::endl;

		TimingLog::energies.back().push_back(curEnergy);

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

void SBA::alignGradientDescent(std::vector<mat4f>& transforms, unsigned int maxNumIters)
{
#ifdef USE_LIE_SPACE
	throw MLIB_EXCEPTION("lie not supported!");
#endif

	setEnergyHelperGlobal(m_energyTermHelper);
	std::cout << "[ align gradient descent ]" << std::endl;
	TimingLog::energies.push_back(std::vector<float>());
	m_timer.start();

	float eps = 0.0001f;
	float lambda = 0.01f;

	float prev = std::numeric_limits<float>::infinity();
	unsigned int numVars = m_energyTermHelper->getNumUnknowns();

	unsigned int iter = 0;
	while (iter < maxNumIters) {

		std::vector<Pose> poses(transforms.size());
		poses[0] = Pose(0.0f); // first is identity
		for (unsigned int i = 1; i < transforms.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(transforms[i]);

		m_energyTermHelper->computeEnergyAndJacobian(transforms, poses);
		float curEnergy = m_energyTermHelper->getLastEnergy();
		std::cout << "[ iteration " << iter << " ] " << curEnergy << std::endl;
		const Eigen::VectorXf& JTr = m_energyTermHelper->getJTr();

		// update
		for (unsigned int i = 1; i < transforms.size(); i++) {
			for (unsigned int j = 0; j < POSESIZE; j++)
				poses[i][j] -= lambda * JTr((i - 1) * POSESIZE + j);
			transforms[i] = PoseHelper::PoseToMatrix(poses[i]);
		}

		TimingLog::energies.back().push_back(curEnergy);

		if (curEnergy < eps)
			break;
		prev = curEnergy;
		iter++;
	}
	float finalEnergy = m_energyTermHelper->getLastEnergy();
	if (finalEnergy == 0) std::cout << "FAILED!\n";
	std::cout << "last energy: " << finalEnergy << std::endl;

	m_timer.stop();
	TimingLog::timeBundleAdjustment = m_timer.getElapsedTime();
	std::cout << "[ align gradient descent ] " << TimingLog::timeBundleAdjustment << " sec" << std::endl;
}

#ifdef DEBUG_JAC_RES
void SBA::alignNumeric(std::vector<mat4f>& transforms, unsigned int maxNumIters)
{
	setEnergyHelperGlobal(m_energyTermHelper);
	std::cout << "[ alignNumeric ]" << std::endl;
	m_timer.start();

	float eps = 0.0f0001;
	float epsg = 0.0f00001;

	float prev = std::numeric_limits<float>::infinity();
	unsigned int numVars = m_energyTermHelper->getNumUnknowns();

	unsigned int iter = 0;
	while (iter < maxNumIters) {

		std::vector<Pose> poses(transforms.size());
		poses[0] = Pose(0.0f); // first is identity
		for (unsigned int i = 1; i < transforms.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(transforms[i]);

		m_energyTermHelper->computeEnergy(transforms);
		float curEnergy = m_energyTermHelper->getLastEnergy();
		m_energyTermHelper->computeNumericJacobian(poses);

		std::cout << "[ iteration " << iter << " ] " << curEnergy << std::endl;
		const Eigen::MatrixXf& jacobian = m_energyTermHelper->getJacobian();
		const Eigen::VectorXf& residual = m_energyTermHelper->getResidualVector();
		Eigen::MatrixXf JTJ = jacobian.transpose() * jacobian;
		Eigen::VectorXf JTr = jacobian.transpose() * residual;
		Eigen::JacobiSVD<Eigen::MatrixXf> SVD(JTJ, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::VectorXf delta = SVD.solve(JTr);

		for (unsigned int i = 0; i < numVars; i++) {
			if (isnan(delta(i))) {
				std::cout << "invalid x!" << std::endl;
				for (unsigned int j = 0; j < numVars; j++)
					std::cout << delta(j) << " ";
				std::cout << std::endl;
				getchar();
			}
		}

		// update
		for (unsigned int i = 1; i < transforms.size(); i++) {
			for (unsigned int j = 0; j < POSESIZE; j++)
				poses[i][j] -= delta((i - 1) * POSESIZE + j);
			transforms[i] = PoseHelper::PoseToMatrix(poses[i]);
		}

		TimingLog::energies.push_back(curEnergy);

		if (curEnergy < eps)
			break;
		if (math::abs(prev - curEnergy) < epsg)
			break;
		prev = curEnergy;
		iter++;
	}
	float finalEnergy = m_energyTermHelper->getLastEnergy();
	if (finalEnergy == 0) std::cout << "FAILED!\n";
	std::cout << "last energy: " << finalEnergy << std::endl;

	m_timer.stop();
	TimingLog::timeBundleAdjustment = m_timer.getElapsedTime();
	std::cout << "[ align ] " << TimingLog::timeBundleAdjustment << " sec" << std::endl;
}
#endif

void SBA::align(std::vector<mat4f>& transforms, unsigned int maxNumIters, bool useGPUSolve, unsigned int numPCGits, bool useVerify)
{
	m_bVerify = false;

	setEnergyHelperGlobal(m_energyTermHelper);
	MLIB_ASSERT(transforms.size() == s_energyTermHelperGlobal->getNumFrames());
	std::cout << "[ align ]" << std::endl;
	TimingLog::numFrames = (unsigned int)transforms.size();

	m_maxResidual = -1.0f;

	m_timer.start();

	if (useGPUSolve) {
		bool removed = false;
		const unsigned int maxIts = GlobalAppState::get().s_maxNumResidualsRemoved;
		unsigned int curIt = 0;
		do {
			TimingLog::numFramesPerOpt.push_back(m_energyTermHelper->getImageBundle()->getNumImages());

			removed = alignCUDA(transforms, maxNumIters, numPCGits, useVerify);
			TimingLog::energies.push_back(m_solver->getConvergenceAnalysis());
			TimingLog::numIterations += (unsigned int)TimingLog::energies.back().size();
			curIt++;
		} while (removed && curIt < maxIts);

		if (GlobalAppState::get().s_enableDetailedTimings) m_solver->evaluateTimings();
	}
	else {
		do {
			TimingLog::numFramesPerOpt.push_back(m_energyTermHelper->getImageBundle()->getNumImages());

			TimingLog::energies.push_back(std::vector<float>());
			alignInternal(transforms, maxNumIters, false);
			float finalEnergy = s_energyTermHelperGlobal->getLastEnergy();
			if (finalEnergy == 0) std::cout << "FAILED!\n";
			std::cout << "last energy: " << finalEnergy << std::endl;
			TimingLog::numCorrespondencesPerOpt.push_back(s_energyTermHelperGlobal->getNumResiduals() / 3);
			TimingLog::numIterations += (unsigned int)TimingLog::energies.back().size();
		} while (s_energyTermHelperGlobal->findAndRemoveOutliers());
		if (useVerify) m_bVerify = m_energyTermHelper->useVerification();
	}

	m_timer.stop();
	TimingLog::timeBundleAdjustment = m_timer.getElapsedTime();
	std::cout << "[ align Time:] " << TimingLog::timeBundleAdjustment << " sec" << std::endl;

}

bool SBA::alignCUDA(std::vector<mat4f>& transforms, unsigned int numNonLinearIterations, unsigned int numLinearIterations, bool useVerify)
{
	// correspondences
	const ImageBundle* bundle = m_energyTermHelper->getImageBundle();
	const Matches& matches = bundle->getMatches();
	m_numCorrespondences = matches.getNumCorrespondences();
	TimingLog::numCorrespondencesPerOpt.push_back(m_numCorrespondences);
	std::vector<Correspondence> correspondences(m_numCorrespondences);
	unsigned int corrIdx = 0;
	std::vector<unsigned int> numCorrPerImage(bundle->getNumImages(), 0);
	for (unsigned int i = 0; i < matches.size(); i++) {
		const ImagePairMatchSet& set = matches.getImagePairMatchSet(i);
		const vec2i& imageIndices = set.getImageIndices();
		for (unsigned int m = 0; m < set.getNumMatches(); m++) {
			const vec2ui keyIndices = set.getKeyPointIndices(m);
			correspondences[corrIdx].idx0 = imageIndices.x;
			correspondences[corrIdx].idx1 = imageIndices.y;
			correspondences[corrIdx].p0 = make_float3(bundle->getKeyPointCameraSpacePosition(imageIndices.x, keyIndices.x));
			correspondences[corrIdx].p1 = make_float3(bundle->getKeyPointCameraSpacePosition(imageIndices.y, keyIndices.y));
			corrIdx++;
			numCorrPerImage[imageIndices.x]++;
			numCorrPerImage[imageIndices.y]++;
		}
	}
	std::cout << "corrIdx = " << corrIdx << std::endl;
	unsigned int maxNumCorrPerImage = 0;
	for (unsigned int i = 0; i < numCorrPerImage.size(); i++)
		if (numCorrPerImage[i] > maxNumCorrPerImage) maxNumCorrPerImage = numCorrPerImage[i];
	cutilSafeCall(cudaMemcpy(d_correspondences, correspondences.data(), sizeof(Correspondence) * m_numCorrespondences, cudaMemcpyHostToDevice));

	// transforms
	unsigned int numImages = bundle->getNumImages();
	std::vector<Pose> poses = PoseHelper::convertToPoses(transforms);
	std::vector<vec3f> rotations(numImages), translations(numImages);
	for (unsigned int i = 0; i < numImages; i++) {
		rotations[i] = vec3f(poses[i][0], poses[i][1], poses[i][2]);
		translations[i] = vec3f(poses[i][3], poses[i][4], poses[i][5]);
	}
	cutilSafeCall(cudaMemcpy(d_xRot, rotations.data(), sizeof(float3) * numImages, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_xTrans, translations.data(), sizeof(float3) * numImages, cudaMemcpyHostToDevice));

	std::cout << "#correspondences = " << m_numCorrespondences << std::endl;
	std::cout << "max #corr per image = " << maxNumCorrPerImage << std::endl;
	if (maxNumCorrPerImage >= GlobalAppState::get().s_maxNumCorrPerImage) {
		std::cout << "ERROR: TOO MANY CORRESPONDENCES PER IMAGE!" << std::endl;
		getchar();
	}
	m_solver->solve(d_correspondences, m_numCorrespondences, numImages, numNonLinearIterations, numLinearIterations, d_xRot, d_xTrans);

	cutilSafeCall(cudaMemcpy(rotations.data(), d_xRot, sizeof(float3) * numImages, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(translations.data(), d_xTrans, sizeof(float3) * numImages, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < numImages; i++)
		poses[i] = Pose(rotations[i].x, rotations[i].y, rotations[i].z, translations[i].x, translations[i].y, translations[i].z);
	transforms = PoseHelper::convertToMatrices(poses);

	bool removed = removeMaxResidualCUDA(d_correspondences);

	if (!removed && useVerify) m_bVerify = m_solver->useVerification(d_correspondences, m_numCorrespondences);

	return removed;
}

bool SBA::removeMaxResidualCUDA(Correspondence* d_correspondences)
{
#ifndef CONVERGENCE_ANALYSIS
	ImageBundle* bundle = m_energyTermHelper->getImageBundle();
	vec2ui imageIndices;
	bool remove = m_solver->getMaxResidual(d_correspondences, imageIndices, m_maxResidual);
	TimingLog::maxComponentResiduals.push_back(std::make_pair(imageIndices, m_maxResidual));
	if (remove) {
		//!!!
		std::cout << "\timages (" << imageIndices << "): invalid match " << m_maxResidual << std::endl;
		//bundle->getMatches().visualizeMatch("debug/invalidated/", imageIndices);
		//getchar();
		//!!!
		bundle->invalidateImagePairMatch(imageIndices);
		bundle->filterFramesByMatches(); // need to re-adjust for removed matches
		return true;
	}
	std::cout << "\thighest residual " << m_maxResidual << " from images (" << imageIndices << ")" << std::endl;
	//bundle->getMatches().visualizeMatch("debug/matches/", imageIndices);
	return false;
#else
	ImageBundle* bundle = m_energyTermHelper->getImageBundle();
	vec2ui imageIndices;
	m_solver->getMaxResidual(d_correspondences, imageIndices, m_maxResidual);
	TimingLog::maxComponentResiduals.push_back(std::make_pair(imageIndices, m_maxResidual));
	std::cout << "\thighest residual " << m_maxResidual << " from images (" << imageIndices << ")" << std::endl;
	return false;
#endif
}
#endif