#include "stdafx.h"
#include "CorrespondenceEvaluator.h"

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
#include "CUDACache.h"
#include "SiftVisualization.h" //for debugging

const std::string CorrespondenceEvaluator::splitter = ",";

void CorrespondenceEvaluator::computeCachedData(const SIFTImageManager* siftManager, const CUDACache* cudaCache)
{
	const unsigned int curFrame = siftManager->getCurrentFrame();
	const unsigned int numFrames = siftManager->getNumImages();
	const float depthMin = GlobalBundlingState::get().s_denseDepthMin;
	const float depthMax = GlobalBundlingState::get().s_denseDepthMax;
	const float distThresh = GlobalBundlingState::get().s_projCorrDistThres;
	const float normalThresh = GlobalBundlingState::get().s_projCorrNormalThres;
	const float colorThresh = GlobalBundlingState::get().s_projCorrColorThresh;

	siftManager->getSIFTKeyPointsDEBUG(m_cachedKeys);

	m_cacheHasGTCorrByOverlap.resize(numFrames, false);
	unsigned int width = cudaCache->getWidth(), height = cudaCache->getHeight();
	const auto& cachedFrames = cudaCache->getCacheFrames();

	DepthImage32 curDepth(width, height);		ColorImageR32 curIntensity(width, height);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curDepth.getData(), cachedFrames[curFrame].d_depthDownsampled, sizeof(float)*curDepth.getNumPixels(), cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curIntensity.getData(), cachedFrames[curFrame].d_intensityDownsampled, sizeof(float)*curIntensity.getNumPixels(), cudaMemcpyDeviceToHost));

	DepthImage32 prvDepth(width, height);		ColorImageR32 prvIntensity(width, height);		mat4f transformCurToPrv;
	for (unsigned int i = 0; i < numFrames; i++) {
		if (i == curFrame) continue; 
		transformCurToPrv = m_referenceTrajectory[i].getInverse() * m_referenceTrajectory[curFrame]; //TODO still something wrong here?

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prvDepth.getData(), cachedFrames[i].d_depthDownsampled, sizeof(float)*prvDepth.getNumPixels(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prvIntensity.getData(), cachedFrames[i].d_intensityDownsampled, sizeof(float)*prvIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
		const vec2ui overlap = computeOverlap(curDepth, curIntensity, prvDepth, prvIntensity, transformCurToPrv, 
			cudaCache->getIntrinsics(), depthMin, depthMax, distThresh, normalThresh, colorThresh, m_minOverlapThreshForGTCorr,
			false);
			//(i == 0 && curFrame == 4));
		float o = (float)overlap.x / (float)overlap.y;
		if (overlap.y > 0 && o > m_minOverlapThreshForGTCorr) {
			//std::cout << "gt overlap frames (" << i << ", " << curFrame << "): " << o << " (" << overlap.x << "/" << overlap.y << ")" << std::endl;
			m_cacheHasGTCorrByOverlap[i] = true;
		}
		//else
		//	std::cout << "no gt overlap for frames (" << i << ", " << curFrame << "): " << o << " (" << overlap.x << "/" << overlap.y << ")" << std::endl;
	}

}

CorrEvaluation CorrespondenceEvaluator::evaluate(const SIFTImageManager* siftManager, const CUDACache* cudaCache,
	const mat4f& siftIntrinsicsInv, bool filtered, bool recomputeCache, bool clearCache, const std::string& corrType)
{
	//TODO: return the vector or the sum or evals?

	const unsigned int curFrame = siftManager->getCurrentFrame();
	const unsigned int numFrames = siftManager->getNumImages();
	const bool useLog = isLoggingToFile();

	if (recomputeCache) computeCachedData(siftManager, cudaCache);

	std::vector<uint2> keyPointIndices; std::vector<unsigned int> numMatches;
	siftManager->getCurrMatchKeyPointIndicesDEBUG(keyPointIndices, numMatches, filtered);
	const unsigned int offsetVal = filtered ? MAX_MATCHES_PER_IMAGE_PAIR_FILTERED : MAX_MATCHES_PER_IMAGE_PAIR_RAW;

	CorrEvaluation eval;
	//evaluate!
	for (unsigned int p = 0; p < numFrames; p++) {
		if (p == curFrame) continue;
		if (m_cacheHasGTCorrByOverlap[p]) {
			eval.numTotal++;
			if (numMatches[p] > 0) eval.numDetected++;
		}
		//check individual corrs
		unsigned int offset = offsetVal * p;
		float maxErr2 = 0.0f;
		for (unsigned int m = 0; m < numMatches[p]; m++) {
			const SIFTKeyPoint& k0 = m_cachedKeys[keyPointIndices[offset + m].x];
			const SIFTKeyPoint& k1 = m_cachedKeys[keyPointIndices[offset + m].y];
			const vec3f cp0 = depthToCamera(siftIntrinsicsInv, k0.pos.x, k0.pos.y, k0.depth);
			const vec3f cp1 = depthToCamera(siftIntrinsicsInv, k1.pos.x, k1.pos.y, k1.depth);
			vec3f err = m_referenceTrajectory[p] * cp0 - m_referenceTrajectory[curFrame] * cp1; 
			const float err2 = err.lengthSq();
			if (err2 > maxErr2) maxErr2 = err2;

			////debugging
			//if (err2 > m_maxProjErrorForCorrectCorr && filtered) {
			//	const mat4f transformCurToPrv = m_referenceTrajectory[p].getInverse() * m_referenceTrajectory[curFrame];
			//	vec3f posPrev = cp0;		vec3f posCur = transformCurToPrv * cp1;
			//	MeshDataf keyPrev = Shapesf::sphere(0.02f, posPrev, 10, 10, vec4f(1.0f, 0.0f, 0.0f, 1.0f)).computeMeshData();
			//	MeshDataf keyCur = Shapesf::sphere(0.02f, posCur, 10, 10, vec4f(0.0f, 1.0f, 0.0f, 1.0f)).computeMeshData();
			//	MeshIOf::saveToFile("debug/overlap/key_prev.ply", keyPrev); //prev
			//	MeshIOf::saveToFile("debug/overlap/key_cur.ply", keyCur); //cur
			//	static SensorData sd;
			//	if (sd.m_frames.empty()) sd.loadFromFile("../data/fr1_desk_from20.sens");
			//	SiftVisualization::printMatch("debug/overlap/match.png", sd, m_cachedKeys, numMatches, keyPointIndices,
			//		vec2ui(p*10, curFrame*10), offset, filtered, m);
			//	sd.saveFrameToPointCloud("debug/overlap/frame_prev.ply", p * 10, mat4f::identity());
			//	sd.saveFrameToPointCloud("debug/overlap/frame_cur.ply", curFrame * 10, transformCurToPrv);
			//	int a = 5;
			//}
			////debugging
		}
		if (numMatches[p] > 0) {
			if (maxErr2 < m_maxProjErrorForCorrectCorr) {
				if (m_cacheHasGTCorrByOverlap[p])
					eval.numCorrect++;
				//else {
				//	std::cout << "WARNING: found corr between frames (" << p << ", " << curFrame << ") but no gt overlap found" << std::endl;
				//}
			}
			else {
				if (useLog) m_outStreamIncorrect << numFrames << splitter << curFrame << splitter << p << splitter << corrType << splitter << std::sqrt(maxErr2) << std::endl;
			}
		}
		if (useLog && m_cacheHasGTCorrByOverlap[p] && numMatches[p] == 0) m_outStreamIncorrect << numFrames << splitter << curFrame << splitter << p << splitter << corrType << splitter << -1.0f << std::endl;
	} //each im-im corr for the current frame
	if (useLog) m_outStreamPerFrame << numFrames << splitter << curFrame << splitter << corrType << splitter << eval.getPrecision() << splitter << eval.getRecall() << splitter << eval.numCorrect << splitter << eval.numDetected << splitter << eval.numTotal << std::endl;

	if (clearCache) clearCachedData();

	return eval;
}

void CorrespondenceEvaluator::computeCorrespondences(const DepthImage32& depth0, const ColorImageR32& color0, 
	const DepthImage32& depth1, const ColorImageR32& color1, const mat4f& transform0to1, const mat4f& depthIntrinsics,
	float depthMin, float depthMax, float distThresh, float normalThresh, float colorThresh,
	float& sumResidual, float& sumWeight, unsigned int& numCorr, unsigned int& numValid, bool debugPrint)
{
	const float INVALID = -std::numeric_limits<float>::infinity();
	sumResidual = 0.0f;
	sumWeight = 0.0f;
	numCorr = 0;
	numValid = 0;

	//camera pos / normals
	const mat4f& depthIntrinsicsInverse = depthIntrinsics.getInverse();
	PointImage cameraPos0, normal0;
	computeCameraSpacePositions(depth0, depthIntrinsicsInverse, cameraPos0);
	computeNormals(cameraPos0, depthIntrinsicsInverse, normal0);
	PointImage cameraPos1, normal1;
	computeCameraSpacePositions(depth1, depthIntrinsicsInverse, cameraPos1);
	computeNormals(cameraPos1, depthIntrinsicsInverse, normal1);

	const std::string debugDir = "debug/overlap/";
	if (debugPrint) {
		ColorImageR32G32B32 cp0 = cameraPos0, n0 = normal0, cp1 = cameraPos1, n1 = normal1;
		float min0 = std::numeric_limits<float>::infinity(), min1 = std::numeric_limits<float>::infinity();
		float max0 = -std::numeric_limits<float>::infinity(), max1 = -std::numeric_limits<float>::infinity();
		for (unsigned int i = 0; i < cp0.getNumPixels(); i++) {
			const vec3f& c0 = cp0.getData()[i];
			if (c0.x != -std::numeric_limits<float>::infinity()) {
				if (c0.x < min0) min0 = c0.x;
				if (c0.y < min0) min0 = c0.y;
				if (c0.z < min0) min0 = c0.z;
				if (c0.x > max0) max0 = c0.x;
				if (c0.y > max0) max0 = c0.y;
				if (c0.z > max0) max0 = c0.z;
			}
			const vec3f& c1 = cp1.getData()[i];
			if (c1.x != -std::numeric_limits<float>::infinity()) {
				if (c1.x < min1) min1 = c1.x;
				if (c1.y < min1) min1 = c1.y;
				if (c1.z < min1) min1 = c1.z;
				if (c1.x > max1) max1 = c1.x;
				if (c1.y > max1) max1 = c1.y;
				if (c1.z > max1) max1 = c1.z;
			}
		}
		for (unsigned int i = 0; i < cp0.getNumPixels(); i++) {
			vec3f& c0 = cp0.getData()[i];
			if (c0.x != -std::numeric_limits<float>::infinity()) c0 = (c0 - min0) / (max0 - min0);
			vec3f& c1 = cp1.getData()[i];
			if (c1.x != -std::numeric_limits<float>::infinity()) c1 = (c1 - min1) / (max1 - min1);
			n0.getData()[i] = (n0.getData()[i] - 1.0f) / 2.0f;
			n1.getData()[i] = (n1.getData()[i] - 1.0f) / 2.0f;
		}
		FreeImageWrapper::saveImage(debugDir + "campos0.png", cp0);
		FreeImageWrapper::saveImage(debugDir + "normal0.png", n0);
		FreeImageWrapper::saveImage(debugDir + "campos1.png", cp1);
		FreeImageWrapper::saveImage(debugDir + "normal1.png", n1);
	}
	DepthImage32 corrImage(depth0.getWidth(), depth0.getHeight());
	corrImage.setPixels(-std::numeric_limits<float>::infinity());
	PointCloudf pc0Trans, pc1, pcCorr;

	const vec4f defaultColor(0.5f, 0.5f, 0.5f, 1.0f);

	for (unsigned int y = 0; y < depth0.getHeight(); y++) {
		for (unsigned int x = 0; x < depth0.getWidth(); x++) {
			if (debugPrint) {
				float d = depth1(x, y); const vec3f& cp = cameraPos1(x, y); const vec3f& n = normal1(x, y);
				if (d > depthMin && d < depthMax && cp.x != INVALID && n.x != INVALID) {
					pc1.m_points.push_back(cp);
					pc1.m_colors.push_back(vec4f(vec3f(color1(x, y)), 1.0f));
					pc1.m_colors.push_back(defaultColor);
				}
			}
			const vec4f p0 = vec4f(cameraPos0(x, y), 1.0f);
			const vec4f n0 = vec4f(normal0(x, y), 0.0f);
			const float cInput = color0(x, y);
			if (p0.x != INVALID && n0.x != INVALID) {
				if (depth0(x, y) > depthMin && depth0(x, y) < depthMax) numValid++;
				const vec4f pTransInput = transform0to1 * p0;
				const vec4f nTransInput = transform0to1 * n0;
				if (debugPrint) {
					pc0Trans.m_points.push_back(pTransInput.getVec3());
					pc0Trans.m_colors.push_back(vec4f(vec3f(color0(x, y)), 1.0f));
					pc0Trans.m_colors.push_back(defaultColor);
				}
				vec2f screenPosf = cameraToDepth(depthIntrinsics, pTransInput.getVec3());
				vec2i screenPos = math::round(screenPosf);
				if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < (int)depth0.getWidth() && screenPos.y < (int)depth0.getHeight()) {
					vec4f pTarget, nTarget; float cTarget;
					getBestCorrespondence1x1(screenPos, pTarget, nTarget, cTarget, cameraPos1, normal1, color1);

					if (pTarget.x != INVALID && nTarget.x != INVALID) {
						float d = (pTransInput - pTarget).length();
						float dNormal = nTransInput | nTarget;
						float c = (cInput - cTarget);

						float projInputDepth = pTransInput.z;//cameraToDepthZ(pTransInput);
						float tgtDepth = depth1(screenPos.x, screenPos.y);

						//bool b = ((tgtDepth != INVALID && projInputDepth < tgtDepth) && d > distThresh); // bad matches that are known
						if ((dNormal >= normalThresh && d <= distThresh /*&& c <= colorThresh*/) /*|| b*/) { // if normal/pos/color correspond or known bad match
							const float weight = std::max(0.0f, 0.5f*((1.0f - d / distThresh) + (1.0f - cameraToKinectProjZ(pTransInput.z, depthMin, depthMax)))); // for weighted ICP;

							sumResidual += d;	//residual
							sumWeight += weight;			//corr weight
							numCorr++;					//corr number
							if (debugPrint) {
								pcCorr.m_points.push_back(pTransInput.getVec3());
								pcCorr.m_colors.push_back(vec4f(vec3f(cInput), 1.0f));
								pcCorr.m_colors.push_back(defaultColor);
								corrImage(x, y) = d;
							}
						}
					} // projected to valid depth
				} // inside image
			}
		} // x
	} // y
	if (debugPrint) {
		PointCloudIOf::saveToFile(debugDir + "pc0trans.ply", pc0Trans);
		PointCloudIOf::saveToFile(debugDir + "pc1.ply", pc1);
		PointCloudIOf::saveToFile(debugDir + "pcCorr.ply", pcCorr);
		FreeImageWrapper::saveImage(debugDir + "corr.png", ColorImageR32G32B32(corrImage));
		int a = 5;
	}
}

ml::vec2ui CorrespondenceEvaluator::computeOverlap(const DepthImage32& depth0, const ColorImageR32& color0, 
	const DepthImage32& depth1, const ColorImageR32& color1, const mat4f transform0to1, const mat4f& depthIntrinsics,
	float depthMin, float depthMax, float distThresh, float normalThresh, float colorThresh, float earlyOutThresh, bool debugPrint)
{
	// 0 -> 1
	float sumResidual0 = 0.0f, sumWeight0 = 0.0f; unsigned int numCorr0 = 0, numValid0 = 0;
	computeCorrespondences(depth0, color0, depth1, color1, transform0to1,
		depthIntrinsics, depthMin, depthMax,
		distThresh, normalThresh, colorThresh,
		sumResidual0, sumWeight0, numCorr0, numValid0, debugPrint);
	float percent0 = (float)numCorr0 / (float)numValid0;
	if (percent0 > earlyOutThresh) return vec2ui(numCorr0, numValid0);

	unsigned int numCorr1 = 0, numValid1 = 0;
	computeCorrespondences(depth1, color1, depth0, color0, transform0to1.getInverse(),
		depthIntrinsics, depthMin, depthMax,
		distThresh, normalThresh, colorThresh,
		sumResidual0, sumWeight0, numCorr1, numValid1, debugPrint);
	float percent1 = (float)numCorr1 / (float)numValid1;

	if (percent0 > percent1)
		return vec2ui(numCorr0, numValid0);
	return vec2ui(numCorr1, numValid1);
}

void CorrespondenceEvaluator::computeNormals(const PointImage& cameraPos, const mat4f& intrinsicsInv, PointImage& normal)
{
	normal.allocate(cameraPos.getWidth(), cameraPos.getHeight());
	for (unsigned int y = 0; y < cameraPos.getHeight(); y++) {
		for (unsigned int x = 0; x < cameraPos.getWidth(); x++) {
			if (x > 0 && x + 1 < cameraPos.getWidth() && y > 0 && y + 1 < cameraPos.getHeight()) {
				const vec3f& CC = cameraPos(x + 0, y + 0);
				const vec3f& PC = cameraPos(x + 0, y + 1);
				const vec3f& CP = cameraPos(x + 1, y + 0);
				const vec3f& MC = cameraPos(x + 0, y - 1);
				const vec3f& CM = cameraPos(x - 1, y + 0);

				if (CC.x != -std::numeric_limits<float>::infinity() && PC.x != -std::numeric_limits<float>::infinity() &&
					CP.x != -std::numeric_limits<float>::infinity() && MC.x != -std::numeric_limits<float>::infinity() &&
					CM.x != -std::numeric_limits<float>::infinity())
				{
					const vec3f n = (PC - MC) ^ (CP - CM);
					const float l = n.length();
					if (l > 0.0f) normal(x, y) = n / -l;
					else normal(x, y) = vec3f(-std::numeric_limits<float>::infinity());
				}
			}
			else {
				normal(x, y) = vec3f(-std::numeric_limits<float>::infinity());
			}
		} //x
	} //y
}

void CorrespondenceEvaluator::computeCameraSpacePositions(const DepthImage32& depth, const mat4f& intrinsicsInv, PointImage& cameraPos)
{
	cameraPos.allocate(depth.getWidth(), depth.getHeight());
	for (unsigned int y = 0; y < depth.getHeight(); y++) {
		for (unsigned int x = 0; x < depth.getWidth(); x++) {
			const float d = depth(x, y);
			if (d == -std::numeric_limits<float>::infinity()) cameraPos(x, y) = vec3f(-std::numeric_limits<float>::infinity());
			else cameraPos(x, y) = (intrinsicsInv*vec4f((float)x*d, (float)y*d, d, d)).getVec3();
		} //x
	} //y
}

#endif