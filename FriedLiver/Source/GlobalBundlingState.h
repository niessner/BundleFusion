#pragma once

#include "stdafx.h"

#include <vector>
#include <string>
#include <list>

#define USE_GLOBAL_DENSE_AT_END

//#define EVALUATE_SPARSE_CORRESPONDENCES
//#define PRINT_MEM_STATS


#define X_GLOBAL_BUNDLING_APP_STATE_FIELDS \
	X(bool, s_enableGlobalTimings) \
	X(bool, s_enablePerFrameTimings) \
	X(unsigned int, s_maxNumImages) \
	X(unsigned int, s_submapSize) \
	X(unsigned int, s_widthSIFT) \
	X(unsigned int, s_heightSIFT) \
	X(unsigned int, s_maxNumKeysPerImage) \
	X(unsigned int, s_numLocalNonLinIterations) \
	X(unsigned int, s_numLocalLinIterations) \
	X(unsigned int, s_numGlobalNonLinIterations) \
	X(unsigned int, s_numGlobalLinIterations) \
	X(unsigned int, s_downsampledWidth) \
	X(unsigned int, s_downsampledHeight) \
	X(float, s_verifySiftErrThresh) \
	X(float, s_verifySiftCorrThresh) \
	X(float, s_projCorrDistThres) \
	X(float, s_projCorrNormalThres) \
	X(float, s_projCorrColorThresh) \
	X(float, s_surfAreaPcaThresh) \
	X(bool, s_recordSolverConvergence) \
	X(bool, s_erodeSIFTdepth) \
	X(float, s_verifyOptErrThresh) \
	X(float, s_verifyOptCorrThresh) \
	X(bool, s_verbose) \
	X(bool, s_sendUplinkFeedbackImage) \
	X(float, s_depthSigmaD) \
	X(float, s_depthSigmaR) \
	X(bool, s_depthFilter) \
	X(unsigned int, s_minNumMatchesLocal) \
	X(unsigned int, s_minNumMatchesGlobal) \
	X(bool, s_useComprehensiveFrameInvalidation) \
	X(float, s_maxKabschResidual2) \
	X(float, s_minKeyScale) \
	X(float, s_siftMatchThresh) \
	X(float, s_siftMatchRatioMaxLocal) \
	X(float, s_siftMatchRatioMaxGlobal) \
	X(bool, s_useLocalVerify) \
	X(bool, s_useLocalDense) \
	X(unsigned int, s_numOptPerResidualRemoval) \
	X(float, s_colorDownSigma) \
	X(float, s_depthDownSigmaD) \
	X(float, s_depthDownSigmaR) \
	X(float, s_optMaxResThresh) \
	X(float, s_denseDistThresh) \
	X(float, s_denseNormalThresh) \
	X(float, s_denseColorThresh) \
	X(float, s_denseColorGradientMin) \
	X(float, s_denseDepthMin) \
	X(float, s_denseDepthMax) \
	X(unsigned int, s_denseOverlapCheckSubsampleFactor)

using namespace ml;

#ifndef VAR_NAME
#define VAR_NAME(x) #x
#endif

#define checkSizeArray(a, d)( (((sizeof a)/(sizeof a[0])) >= d))

class GlobalBundlingState
{
public:

#define X(type, name) type name;
	X_GLOBAL_BUNDLING_APP_STATE_FIELDS
#undef X

		//! sets the parameter file and reads
		void readMembers(const ParameterFile& parameterFile) {
		m_ParameterFile = parameterFile;
		readMembers();
	}

	//! reads all the members from the given parameter file (could be called for reloading)
	void readMembers() {
#define X(type, name) \
	if (!m_ParameterFile.readParameter(std::string(#name), name)) {MLIB_WARNING(std::string(#name).append(" ").append("uninitialized"));	name = type();}
		X_GLOBAL_BUNDLING_APP_STATE_FIELDS
#undef X


			m_bIsInitialized = true;
	}

	void print() const {
#define X(type, name) \
	std::cout << #name " = " << name << std::endl;
		X_GLOBAL_BUNDLING_APP_STATE_FIELDS
#undef X
	}

	static GlobalBundlingState& getInstance() {
		static GlobalBundlingState s;
		return s;
	}
	static GlobalBundlingState& get() {
		return getInstance();
	}


	//! constructor
	GlobalBundlingState() {
		m_bIsInitialized = false;
	}

	//! destructor
	~GlobalBundlingState() {
	}

	Timer	s_Timer;

private:
	bool			m_bIsInitialized;
	ParameterFile	m_ParameterFile;
};
