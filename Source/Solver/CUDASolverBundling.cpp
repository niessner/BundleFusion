
#include "stdafx.h"
#include "CUDASolverBundling.h"
#include "../GlobalBundlingState.h"
#include "../CUDACache.h"
#include "../SiftGPU/MatrixConversion.h"

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);
extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer);
extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, float* convergenceAnalysis, CUDATimer* timer);

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);

extern "C" void BuildDenseSystem(const SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);
extern "C" void convertLiePosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs);

extern "C" void VisualizeCorrespondences(const uint2& imageIndices, const SolverInput& input, SolverState& state, SolverParameters& parameters, float3* d_corrImage);

CUDASolverBundling::CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxNumResiduals)
	: m_maxNumberOfImages(maxNumberOfImages)
	, THREADS_PER_BLOCK(512) // keep consistent with the GPU
{
	m_timer = NULL;
	//if (GlobalBundlingState::get().s_enableDetailedTimings) m_timer = new CUDATimer();
	m_bRecordConvergence = GlobalBundlingState::get().s_recordSolverConvergence;

	//TODO PARAMS
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	m_verifyOptDistThresh = 0.02f;//GlobalAppState::get().s_verifyOptDistThresh;
	m_verifyOptPercentThresh = 0.05f;//GlobalAppState::get().s_verifyOptPercentThresh;

	const unsigned int numberOfVariables = maxNumberOfImages;
	m_maxCorrPerImage = maxNumResiduals / maxNumberOfImages;

	// State
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_deltaRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_deltaTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_rRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_rTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_zRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_zTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_pRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_pTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_Jp, sizeof(float3)*maxNumResiduals));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_Ap_XRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_Ap_XTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_scanAlpha, sizeof(float) * 2));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_rDotzOld, sizeof(float) *numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_precondionerRot, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_precondionerTrans, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_sumResidual, sizeof(float)));
	unsigned int n = (maxNumResiduals * 3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_maxResidual, sizeof(float) * n));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_maxResidualIndex, sizeof(int) * n));
	m_solverState.h_maxResidual = new float[n];
	m_solverState.h_maxResidualIndex = new int[n];

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_variablesToCorrespondences, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_numEntriesPerRow, sizeof(int)*m_maxNumberOfImages));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_countHighResidual, sizeof(int)));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_denseJtJ, sizeof(float) * 36 * numberOfVariables * numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_denseJtr, sizeof(float) * 6 * numberOfVariables));
	m_maxNumDenseImPairs = m_maxNumberOfImages * (m_maxNumberOfImages - 1) / 2;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_denseCorrCounts, sizeof(float) * m_maxNumDenseImPairs));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_denseOverlappingImages, sizeof(uint2) * m_maxNumDenseImPairs));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_numDenseOverlappingImages, sizeof(int)));

	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_corrCount, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_corrCountColor, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_sumResidualColor, sizeof(float)));

#ifdef USE_LIE_SPACE
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_xTransforms, sizeof(float4x4)*m_maxNumberOfImages));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&m_solverState.d_xTransformInverses, sizeof(float4x4)*m_maxNumberOfImages));
#else
	m_solverState.d_xTransforms = NULL;
	m_solverState.d_xTransformInverses = NULL;
#endif
	//!!!DEBUGGING
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_deltaRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_deltaTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_rRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_rTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_zRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_zTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_pRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_pTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_Jp, -1, sizeof(float3)*maxNumResiduals));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_Ap_XRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_Ap_XTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_scanAlpha, -1, sizeof(float) * 2));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_rDotzOld, -1, sizeof(float) *numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_precondionerRot, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_precondionerTrans, -1, sizeof(float3)*numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_sumResidual, -1, sizeof(float)));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_maxResidual, -1, sizeof(float) * n));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_maxResidualIndex, -1, sizeof(int) * n));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_variablesToCorrespondences, -1, sizeof(int)*m_maxNumberOfImages*m_maxCorrPerImage));
	MLIB_CUDA_SAFE_CALL(cudaMemset(d_numEntriesPerRow, -1, sizeof(int)*m_maxNumberOfImages));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_countHighResidual, -1, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_denseJtJ, -1, sizeof(float) * 36 * numberOfVariables * numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_denseJtr, -1, sizeof(float) * 6 * numberOfVariables));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_denseCorrCounts, -1, sizeof(float) * m_maxNumDenseImPairs));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_denseOverlappingImages, -1, sizeof(uint2) * m_maxNumDenseImPairs));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_numDenseOverlappingImages, -1, sizeof(int)));

	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_corrCount, -1, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_corrCountColor, -1, sizeof(int)));
	MLIB_CUDA_SAFE_CALL(cudaMemset(m_solverState.d_sumResidualColor, -1, sizeof(float)));

	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
	//!!!DEBUGGING
}

CUDASolverBundling::~CUDASolverBundling()
{
	if (m_timer) delete m_timer;

	// State
	MLIB_CUDA_SAFE_FREE(m_solverState.d_deltaRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_deltaTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_rRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_rTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_zRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_zTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_pRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_pTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_Jp);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_Ap_XRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_Ap_XTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_scanAlpha);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_rDotzOld);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_precondionerRot);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_precondionerTrans);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_sumResidual);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_maxResidual);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_maxResidualIndex);
	SAFE_DELETE_ARRAY(m_solverState.h_maxResidual);
	SAFE_DELETE_ARRAY(m_solverState.h_maxResidualIndex);

	MLIB_CUDA_SAFE_FREE(d_variablesToCorrespondences);
	MLIB_CUDA_SAFE_FREE(d_numEntriesPerRow);

	MLIB_CUDA_SAFE_FREE(m_solverState.d_countHighResidual);

	MLIB_CUDA_SAFE_FREE(m_solverState.d_denseJtJ);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_denseJtr);

	MLIB_CUDA_SAFE_FREE(m_solverState.d_xTransforms);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_xTransformInverses);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_denseOverlappingImages);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_numDenseOverlappingImages);

	MLIB_CUDA_SAFE_FREE(m_solverState.d_corrCount);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_sumResidualColor);
	MLIB_CUDA_SAFE_FREE(m_solverState.d_corrCountColor);
}

void CUDASolverBundling::solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, const int* d_validImages, unsigned int numberOfImages,
	unsigned int nNonLinearIterations, unsigned int nLinearIterations, const CUDACache* cudaCache,
	const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool usePairwiseDense,
	float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns,
	bool rebuildJT, bool findMaxResidual)
{
	MLIB_ASSERT(numberOfImages > 1 && nNonLinearIterations <= weightsSparse.size());
	if (numberOfCorrespondences > m_maxCorrPerImage*m_maxNumberOfImages) {
		//warning: correspondences will be invalidated AT RANDOM!
		std::cerr << "WARNING: #corr (" << numberOfCorrespondences << ") exceeded limit (" << m_maxCorrPerImage << "*" << m_maxNumberOfImages << "), please increase max #corr per image in the GAS" << std::endl;
	}

	float* convergence = NULL;
	if (m_bRecordConvergence) {
		m_convergence.resize(nNonLinearIterations + 1, -1.0f);
		convergence = m_convergence.data();
	}

	m_solverState.d_xRot = d_rotationAnglesUnknowns;
	m_solverState.d_xTrans = d_translationUnknowns;

	SolverParameters parameters;
	parameters.nNonLinearIterations = nNonLinearIterations;
	parameters.nLinIterations = nLinearIterations;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	parameters.weightSparse = weightsSparse.front();
	parameters.weightDenseDepth = weightsDenseDepth.front();
	parameters.weightDenseColor = weightsDenseColor.front();
	parameters.denseDistThresh = 0.1f; //TODO params
	parameters.denseNormalThresh = 0.97f;
	parameters.denseColorThresh = 0.1f;
	parameters.denseColorGradientMin = 0.005f;
	parameters.denseDepthMin = 0.5f;
	parameters.denseDepthMax = 3.5f; //TODO 
	parameters.useDense = (parameters.weightDenseDepth > 0 || parameters.weightDenseColor > 0);
	parameters.useDenseDepthAllPairwise = usePairwiseDense;
	parameters.denseOverlapCheckSubsampleFactor = 8; // for 160x120 -> 20x15 image

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = d_variablesToCorrespondences;
	solverInput.d_numEntriesPerRow = d_numEntriesPerRow;
	solverInput.numberOfImages = numberOfImages;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;
	solverInput.maxNumDenseImPairs = m_maxNumDenseImPairs;

	solverInput.weightsSparse = weightsSparse.data();
	solverInput.weightsDenseDepth = weightsDenseDepth.data();
	solverInput.weightsDenseColor = weightsDenseColor.data();
	solverInput.d_validImages = d_validImages;
	if (cudaCache) {
		solverInput.d_cacheFrames = cudaCache->getCacheFramesGPU();
		solverInput.denseDepthWidth = cudaCache->getWidth(); //TODO constant buffer for this?
		solverInput.denseDepthHeight = cudaCache->getHeight();
		mat4f intrinsics = cudaCache->getIntrinsics();
		solverInput.intrinsics = make_float4(intrinsics(0, 0), intrinsics(1, 1), intrinsics(0, 2), intrinsics(1, 2));
		MLIB_ASSERT(solverInput.denseDepthWidth / parameters.denseOverlapCheckSubsampleFactor > 8); //need enough samples
	}
	else {
		solverInput.d_cacheFrames = NULL;
		solverInput.denseDepthWidth = 0;
		solverInput.denseDepthHeight = 0;
		solverInput.intrinsics = make_float4(-std::numeric_limits<float>::infinity());
	}
	//!!!debugging
	if (false && parameters.weightDenseDepth > 0) {
		std::vector<int> validImages(solverInput.numberOfImages);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(validImages.data(), solverInput.d_validImages, sizeof(int)*solverInput.numberOfImages, cudaMemcpyDeviceToHost));
		convertLiePosesToMatricesCU(m_solverState.d_xRot, m_solverState.d_xTrans, solverInput.numberOfImages, m_solverState.d_xTransforms, m_solverState.d_xTransformInverses);

		//uint2 imageIndices = make_uint2(87, 88);
		//float3* d_corrImage = NULL; MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_corrImage, sizeof(float3)*solverInput.denseDepthWidth*solverInput.denseDepthHeight));
		//ColorImageR32G32B32 projImage(solverInput.denseDepthWidth, solverInput.denseDepthHeight); projImage.setPixels(vec3f(-std::numeric_limits<float>::infinity()));
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_corrImage, projImage.getPointer(), sizeof(float3)*solverInput.denseDepthWidth*solverInput.denseDepthHeight, cudaMemcpyHostToDevice));
		//VisualizeCorrespondences(imageIndices, solverInput, m_solverState, parameters, d_corrImage);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(projImage.getPointer(), d_corrImage, sizeof(float3)*solverInput.denseDepthWidth*solverInput.denseDepthHeight, cudaMemcpyDeviceToHost));
		//MLIB_CUDA_SAFE_FREE(d_corrImage);
		//FreeImageWrapper::saveImage("debug/imCorr.png", projImage);
		//unsigned int counter = 0;
		//for (unsigned int i = 0; i < projImage.getNumPixels(); i++) {
		//	const vec3f& p = projImage.getPointer()[i];
		//	if (p.x != -std::numeric_limits<float>::infinity()) counter++;
		//}
		//ColorImageR32G32B32A32 camPosImage(solverInput.denseDepthWidth, solverInput.denseDepthHeight);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPosImage.getPointer(), cudaCache->getCacheFrames()[imageIndices.x].d_cameraposDownsampled, sizeof(float4)*solverInput.denseDepthWidth*solverInput.denseDepthHeight, cudaMemcpyDeviceToHost));
		//FreeImageWrapper::saveImage("debug/imTgt.png", camPosImage);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPosImage.getPointer(), cudaCache->getCacheFrames()[imageIndices.y].d_cameraposDownsampled, sizeof(float4)*solverInput.denseDepthWidth*solverInput.denseDepthHeight, cudaMemcpyDeviceToHost));
		//FreeImageWrapper::saveImage("debug/imSrc.png", camPosImage);


		BuildDenseSystem(solverInput, m_solverState, parameters, NULL);
		int numOverlappingImages;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numOverlappingImages, m_solverState.d_numDenseOverlappingImages, sizeof(int), cudaMemcpyDeviceToHost));
		std::vector<float> imPairWeights(numOverlappingImages);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(imPairWeights.data(), m_solverState.d_denseCorrCounts, sizeof(float)*imPairWeights.size(), cudaMemcpyDeviceToHost));
		std::vector<vec2ui> imPairIndices(numOverlappingImages);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(imPairIndices.data(), m_solverState.d_denseOverlappingImages, sizeof(uint2)*imPairIndices.size(), cudaMemcpyDeviceToHost));

		//!!!debugging
		//std::vector<std::pair<vec2ui, unsigned int>> corrCounts;
		std::vector<std::pair<unsigned int, unsigned int>> connections(numberOfImages); for (unsigned int i = 0; i < numberOfImages; i++) connections[i] = std::make_pair(i, 0);
		for (unsigned int i = 0; i < imPairWeights.size(); i++) {
			if (imPairWeights[i] > 0) {
		//		corrCounts.push_back(std::make_pair(imPairIndices[i], (unsigned int)imPairWeights[i]));
				connections[imPairIndices[i].x].second++;
				connections[imPairIndices[i].y].second++;
			}
		}
		//std::sort(corrCounts.begin(), corrCounts.end(), [](const std::pair<vec2ui, float> &left, const std::pair<vec2ui, float> &right) {
		//	return fabs(left.second) < fabs(right.second);
		//});
		std::sort(connections.begin(), connections.end(), [](const std::pair<unsigned int, unsigned int> &left, const std::pair<unsigned int, unsigned int> &right) {
			return left.second < right.second;
		});
		int a = 5;
		//!!!debugging

		ColorImageR8G8B8 corrImage(numberOfImages, numberOfImages); unsigned int count = 0;
		for (unsigned int i = 0; i < imPairWeights.size(); i++) {
			if (imPairWeights[i] > 0) {
				vec3f c = BaseImageHelper::convertDepthToRGB(imPairWeights[i], 0.0f, 1.0f) * 255.0f;
				const vec2ui& images = imPairIndices[i];
				corrImage(images.y, images.x) = vec3uc(c);
				count++;
			}
		}
		// mark diagonal
		for (unsigned int i = 0; i < numberOfImages; i++) {
			corrImage(i, i) = vec3uc(255, 0, 0); // red
		}
		// mark invalid
		for (unsigned int i = 0; i < numberOfImages; i++) {
			if (validImages[i] == 0) {
				for (unsigned int k = 0; k < i; k++)
					corrImage(i, k) = vec3uc(0, 255, 0); //green
				for (unsigned int k = i + 1; k < numberOfImages; k++)
					corrImage(k, i) = vec3uc(0, 255, 0); //green
			}
		}
		const unsigned int query = 88; std::vector<unsigned int> queryConnections;
		for (unsigned int i = 0; i < imPairWeights.size(); i++) {
			if (imPairWeights[i] > 0) {
				if (imPairIndices[i].x == query) queryConnections.push_back(imPairIndices[i].y);
				else if (imPairIndices[i].y == query) queryConnections.push_back(imPairIndices[i].x);
			}
		}
		std::cout << "count = " << count << std::endl;
		FreeImageWrapper::saveImage("debug/corr.png", corrImage);
		std::cout << "waiting..." << std::endl; getchar();
	}
	//!!!debugging



	if (rebuildJT) {
		buildVariablesToCorrespondencesTable(d_correspondences, numberOfCorrespondences);
	}

	solveBundlingStub(solverInput, m_solverState, parameters, convergence, m_timer);

	if (findMaxResidual) {
		computeMaxResidual(solverInput, parameters);
	}
}

void CUDASolverBundling::buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	cutilSafeCall(cudaMemset(d_numEntriesPerRow, 0, sizeof(int)*m_maxNumberOfImages));

	if (numberOfCorrespondences > 0)
		buildVariablesToCorrespondencesTableCUDA(d_correspondences, numberOfCorrespondences, m_maxCorrPerImage, d_variablesToCorrespondences, d_numEntriesPerRow, m_timer);
}

void CUDASolverBundling::computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters)
{
	if (parameters.weightSparse > 0.0f) {
		evalMaxResidual(solverInput, m_solverState, parameters, m_timer);
		// copy to cpu
		unsigned int n = (solverInput.numberOfCorrespondences * 3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		cutilSafeCall(cudaMemcpy(m_solverState.h_maxResidual, m_solverState.d_maxResidual, sizeof(float) * n, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(m_solverState.h_maxResidualIndex, m_solverState.d_maxResidualIndex, sizeof(int) * n, cudaMemcpyDeviceToHost));
		// compute max
		float maxResidual = 0.0f; int maxResidualIndex = 0;
		for (unsigned int i = 0; i < n; i++) {
			if (maxResidual < m_solverState.h_maxResidual[i]) {
				maxResidual = m_solverState.h_maxResidual[i];
				maxResidualIndex = m_solverState.h_maxResidualIndex[i];
			}
		}
		m_solverState.h_maxResidual[0] = maxResidual;
		m_solverState.h_maxResidualIndex[0] = maxResidualIndex;
	}
	else {
		m_solverState.h_maxResidual[0] = 0.0f;
		m_solverState.h_maxResidualIndex[0] = 0;
	}
}

bool CUDASolverBundling::getMaxResidual(EntryJ* d_correspondences, ml::vec2ui& imageIndices, float& maxRes)
{
	const float MAX_RESIDUAL = 0.05f; // nonsquared residual
	if (m_timer) m_timer->startEvent(__FUNCTION__);

	maxRes = m_solverState.h_maxResidual[0];

	// for debugging get image indices regardless
	EntryJ h_corr;
	unsigned int imIdx = m_solverState.h_maxResidualIndex[0] / 3;
	cutilSafeCall(cudaMemcpy(&h_corr, d_correspondences + imIdx, sizeof(EntryJ), cudaMemcpyDeviceToHost));
	imageIndices = ml::vec2ui(h_corr.imgIdx_i, h_corr.imgIdx_j);

	if (m_timer) m_timer->endEvent();

	if (m_solverState.h_maxResidual[0] > MAX_RESIDUAL) { // remove!

		return true;
	}

	return false;
}

bool CUDASolverBundling::useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences)
{
	SolverParameters parameters;
	parameters.nNonLinearIterations = 0;
	parameters.nLinIterations = 0;
	parameters.verifyOptDistThresh = m_verifyOptDistThresh;
	parameters.verifyOptPercentThresh = m_verifyOptPercentThresh;

	SolverInput solverInput;
	solverInput.d_correspondences = d_correspondences;
	solverInput.d_variablesToCorrespondences = NULL;
	solverInput.d_numEntriesPerRow = NULL;
	solverInput.numberOfImages = 0;
	solverInput.numberOfCorrespondences = numberOfCorrespondences;

	solverInput.maxNumberOfImages = m_maxNumberOfImages;
	solverInput.maxCorrPerImage = m_maxCorrPerImage;

	unsigned int numHighResiduals = countHighResiduals(solverInput, m_solverState, parameters, m_timer);
	unsigned int total = solverInput.numberOfCorrespondences * 3;
	//std::cout << "\t[ useVerification ] " << numHighResiduals << " / " << total << " = " << (float)numHighResiduals / total << " vs " << parameters.verifyOptPercentThresh << std::endl;
	if ((float)numHighResiduals / total >= parameters.verifyOptPercentThresh) return true;
	return false;
}
