#pragma once
#include "GlobalBundlingState.h"

#ifdef EVALUATE_SPARSE_CORRESPONDENCES
#include "PoseHelper.h"
#include "SIFTImageManager.h"

class CUDACache;

struct CorrEvaluation {
	unsigned int numCorrect;	//#corrs found that are correct
	unsigned int numDetected;	//#corrs found 
	unsigned int numTotal;		//#corrs that could be found

	CorrEvaluation() {
		numCorrect = 0;
		numDetected = 0;
		numTotal = 0;
	}
	float getPrecision() const { 
		if (numDetected > 0) 
			return (float)numCorrect / (float)numDetected;
		return -std::numeric_limits<float>::infinity();
	}
	float getRecall() const {
		if (numTotal > 0)
			return (float)numDetected / (float)numTotal;
		return -std::numeric_limits<float>::infinity();
	}

	CorrEvaluation& operator+=(const CorrEvaluation& rhs) {                 
		numCorrect += rhs.numCorrect;
		numDetected += rhs.numDetected;
		numTotal += rhs.numTotal;
		return *this; 
	}
};

class CorrespondenceEvaluator
{
public:
	CorrespondenceEvaluator(const std::vector<mat4f>& referenceTrajectory, const std::string& logFilePrefix)
		: m_referenceTrajectory(referenceTrajectory), m_logFilePrefix(logFilePrefix)
	{
		m_minOverlapThreshForGTCorr = 0.1f;//0.3f;
		m_maxProjErrorForCorrectCorr = 0.2f;
		m_maxProjErrorForCorrectCorr2 = m_maxProjErrorForCorrectCorr * m_maxProjErrorForCorrectCorr;
		if (isLoggingToFile()) {
			m_outStreamPerFrame.open(m_logFilePrefix + "_frame.csv");
			m_outStreamIncorrect.open(m_logFilePrefix + "_wrong.csv");
			if (!m_outStreamPerFrame.is_open() || !m_outStreamIncorrect.is_open()) throw MLIB_EXCEPTION("[CorrespondenceEvaluator] failed to open log file(s): " + m_logFilePrefix);
			m_outStreamPerFrame << "numFrames" << splitter << "curFrame" << splitter << "type" << splitter << "precision" << splitter << "recall" << splitter << "numCorrect" << splitter << "numDetected" << splitter << "numTotal" << std::endl;
			m_outStreamIncorrect << "numFrames" << splitter << "curFrame" << splitter << "matchFrame" << splitter << "type" << splitter << "err" << std::endl;
		}
	}
	~CorrespondenceEvaluator() {}

	//recomputeCache for new set of keypoints, clearCache when done
	CorrEvaluation evaluate(const SIFTImageManager* siftManager, const CUDACache* cudaCache, const mat4f& siftIntrinsicsInv,
		bool filtered, bool recomputeCache, bool clearCache, const std::string& corrType);

	bool isLoggingToFile() const { return !m_logFilePrefix.empty(); }

	void finishLoggingToFile() {
		if (isLoggingToFile()) {
			m_outStreamPerFrame.close();
			m_outStreamIncorrect.close();
		}
	}

private:
	void computeCachedData(const SIFTImageManager* siftManager, const CUDACache* cudaCache);
	void clearCachedData()
	{
		m_cachedKeys.clear();
		m_cacheHasGTCorrByOverlap.clear();
	}

	//bidirectional
	//Note: does not use color!
	vec2ui computeOverlap(const DepthImage32& depth0, const ColorImageR32& color0,
		const DepthImage32& depth1, const ColorImageR32& color1, const mat4f transform0to1,
		const mat4f& depthIntrinsics, float depthMin, float depthMax,
		float distThresh, float normalThresh, float colorThresh, float earlyOutThresh, bool debugPrint);
	void computeCorrespondences(const DepthImage32& depth0, const ColorImageR32& color0,
		const DepthImage32& depth1, const ColorImageR32& color1, const mat4f& transform0to1,
		const mat4f& depthIntrinsics, float depthMin, float depthMax,
		float distThresh, float normalThresh, float colorThresh,
		float& sumResidual, float& sumWeight, unsigned int& numCorr, unsigned int& numValid,
		bool debugPrint);
	static void getBestCorrespondence1x1(
		const vec2i& screenPos, vec4f& pTarget, vec4f& nTarget, float& cTarget,
		const PointImage& targetCamPos, const PointImage& targetNormals, const ColorImageR32& targetColors)
	{
		pTarget = vec4f(targetCamPos(screenPos.x, screenPos.y), 1.0f);
		cTarget = targetColors(screenPos.x, screenPos.y); //intensity target color
		nTarget = vec4f(targetNormals(screenPos.x, screenPos.y), 0.0f);
	}
	static float cameraToKinectProjZ(float z, float depthMin, float depthMax) {
		return (z - depthMin) / (depthMax - depthMin);
	}
	vec2f cameraToDepth(const mat4f& depthIntrinsics, const vec3f& pos)
	{
		vec3f p = depthIntrinsics * pos;
		return vec2f(p.x / p.z, p.y / p.z);
	}
	vec3f depthToCamera(const mat4f& depthIntrinsincsinv, float x, float y, float d)
	{
		return depthIntrinsincsinv * (d * vec3f(x, y, 1.0f));
	}
	static void computeCameraSpacePositions(const DepthImage32& depth, const mat4f& intrinsicsInv, PointImage& cameraPos);
	static void computeNormals(const PointImage& cameraPos, const mat4f& intrinsicsInv, PointImage& normal);


	std::vector<mat4f> m_referenceTrajectory;
	float m_minOverlapThreshForGTCorr;
	float m_maxProjErrorForCorrectCorr;
	float m_maxProjErrorForCorrectCorr2;

	//logging (outputs csv)
	std::string m_logFilePrefix;		
	std::ofstream m_outStreamPerFrame;
	std::ofstream m_outStreamIncorrect;

	//cache data
	std::vector<SIFTKeyPoint> m_cachedKeys;
	std::vector<bool> m_cacheHasGTCorrByOverlap;

	static const std::string splitter; //for logging
};


#endif