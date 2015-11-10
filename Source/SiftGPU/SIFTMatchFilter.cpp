#include "stdafx.h"
#include "SIFTMatchFilter.h"

#include "cuda_kabschReference.h"
//#include "cuda_kabsch.h"
#include "../GlobalBundlingState.h"

std::vector<std::vector<unsigned int>> SIFTMatchFilter::s_combinations;
bool SIFTMatchFilter::s_bInit;

// debug variables
DepthImage32 SIFTMatchFilter::s_debugCorr;

void SIFTMatchFilter::filterFrames(SIFTImageManager* siftManager)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	int connected = 0;
	const unsigned int curFrame = numImages - 1;

	std::vector<unsigned int> currNumFilteredMatchesPerImagePair(curFrame);
	cutilSafeCall(cudaMemcpy(currNumFilteredMatchesPerImagePair.data(), siftManager->d_currNumFilteredMatchesPerImagePair, sizeof(unsigned int) * curFrame, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames
		if (currNumFilteredMatchesPerImagePair[i] > 0) {
			connected = 1;
			break;
		}
	}

	if (!connected) {
		std::cout << "frame " << curFrame << " not connected to previous!" << std::endl;
		//getchar();
	}

	cutilSafeCall(cudaMemcpy(siftManager->d_validImages + curFrame, &connected, sizeof(int), cudaMemcpyHostToDevice));
}

void SIFTMatchFilter::filterKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	// current data
	const unsigned int curFrame = numImages - 1;
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	std::vector<float4x4> transforms(curFrame);
	std::vector<float4x4> transformsInv(curFrame);

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames

		std::vector<uint2> keyPointIndices;
		std::vector<float> matchDistances;
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

		float4x4 transform;
		unsigned int newNumMatches =
			filterImagePairKeyPointMatches(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv,
			minNumMatches, GlobalBundlingState::get().s_maxKabschResidual2, false);
		//std::cout << "(" << curFrame << ", " << i << "): " << newNumMatches << std::endl; 

		if (newNumMatches > 0) {
			transforms[i] = transform;
			transformsInv[i] = transform.getInverse();
		}
		else {
			transforms[i].setValue(0.0f);
			transformsInv[i].setValue(0.0f);
		}

		// copy back
		cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		if (newNumMatches > 0) {
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
		}
	}
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
}

unsigned int SIFTMatchFilter::filterImagePairKeyPointMatches(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform,
	const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool printDebug)
{
	unsigned int numRawMatches = (unsigned int)keyPointIndices.size();
	if (numRawMatches < minNumMatches) return 0;

	unsigned int numRef = filterKeyPointMatchesReference(keys.data(), keyPointIndices.data(), matchDistances.data(), numRawMatches, transform,
		siftIntrinsicsInv, minNumMatches, maxResThresh2, printDebug);
	return numRef;
}

void SIFTMatchFilter::filterBySurfaceArea(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames,
	const float4x4& siftIntrinsicsInv, unsigned int minNumMatches)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;

	// current data
	const unsigned int curFrame = numImages - 1;
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	ml::DepthImage32 curDepth(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curDepth.getPointer(), cachedFrames[curFrame].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames
		// get data
		ml::DepthImage32 prvDepth(downSampWidth, downSampHeight);
		cutilSafeCall(cudaMemcpy(prvDepth.getPointer(), cachedFrames[i].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

		std::vector<uint2> keyPointIndices;
		siftManager->getFiltKeyPointIndicesDEBUG(i, keyPointIndices);

		//std::cout << "(" << i << ", " << curFrame << "): ";
		bool valid =
			filterImagePairBySurfaceArea(keyPoints, prvDepth.getPointer(), curDepth.getPointer(), keyPointIndices, siftIntrinsicsInv, minNumMatches);

		if (!valid) {
			// invalidate
			unsigned int numMatches = 0;
			cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
	}
}

bool SIFTMatchFilter::filterImagePairBySurfaceArea(const std::vector<SIFTKeyPoint>& keys, float* depth0, float* depth1, const std::vector<uint2>& keyPointIndices,
	const float4x4& siftIntrinsicsInv, unsigned int minNumMatches)
{
	if (keyPointIndices.size() < minNumMatches) return false;

	const float minSurfaceAreaPca = 0.032f;
	float2 areas = computeSurfaceArea(keys.data(), keyPointIndices.data(), depth0, depth1, (unsigned int)keyPointIndices.size(), siftIntrinsicsInv);

	//std::cout << "areas = " << areas.x << " " << areas.y << std::endl;

	if (areas.x < minSurfaceAreaPca && areas.y < minSurfaceAreaPca) // invalid
		return false;
	return true;
}

float2 SIFTMatchFilter::computeSurfaceArea(const SIFTKeyPoint* keys, const uint2* keyPointIndices, float* depth0, float* depth1, unsigned int numMatches, const float4x4& siftIntrinsicsInv)
{
	ml::vec2f area(0.0f); ml::vec2f totalAreas(1.0f);

	std::vector<ml::vec3f> srcPts(numMatches);
	std::vector<ml::vec3f> tgtPts(numMatches);
	getKeySourceAndTargetPointsReference(keys, keyPointIndices, numMatches, (float3*)srcPts.data(), (float3*)tgtPts.data(), siftIntrinsicsInv);


	std::vector<ml::vec2f> srcPtsProj(numMatches);
	std::vector<ml::vec2f> tgtPtsProj(numMatches);

	auto srcAxes = ml::math::pointSetPCA(srcPts);
	ml::vec3f srcMean = std::accumulate(srcPts.begin(), srcPts.end(), ml::vec3f::origin) / (float)srcPts.size();
	// project points to plane(srcAxes[2].first, mean)
	const ml::vec3f& srcNormal = srcAxes[2].first;
	for (unsigned int i = 0; i < srcPts.size(); i++) {
		srcPts[i] = srcPts[i] - (srcNormal | (srcPts[i] - srcMean)) * srcNormal; // projected point (3d)
		ml::vec3f s = srcPts[i] - srcMean;
		srcPtsProj[i] = ml::vec2f(s | srcAxes[0].first, s | srcAxes[1].first); // projected point (2d plane basis)
	}
	// pca tgt pts
	auto tgtAxes = ml::math::pointSetPCA(tgtPts);
	ml::vec3f tgtMean = std::accumulate(tgtPts.begin(), tgtPts.end(), ml::vec3f::origin) / (float)tgtPts.size();
	// project points to plane(tgtAxes[2].first, mean)
	const ml::vec3f& tgtNormal = tgtAxes[2].first;
	for (unsigned int i = 0; i < tgtPts.size(); i++) {
		tgtPts[i] = tgtPts[i] - (tgtNormal | (tgtPts[i] - tgtMean)) * tgtNormal; // projected point (3d)
		ml::vec3f t = tgtPts[i] - tgtMean;
		tgtPtsProj[i] = ml::vec2f(t | tgtAxes[0].first, t | tgtAxes[1].first); // projected point (2d plane basis)
	}

	// compute areas
	ml::OBB2f srcOBB(srcPtsProj);
	ml::OBB2f tgtOBB(tgtPtsProj);
	area = ml::vec2f((float)srcOBB.getArea(), (float)tgtOBB.getArea());
	return make_float2(area.x, area.y);
}

void SIFTMatchFilter::filterByDenseVerify(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames, float depthMin, float depthMax)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;

	// current data
	const unsigned int curFrame = numImages - 1;
	ml::DepthImage32 curDepth(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curDepth.getPointer(), cachedFrames[curFrame].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR8G8B8A8 curColor(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curColor.getPointer(), cachedFrames[curFrame].d_colorDownsampled, sizeof(uchar4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 curCamPos(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curCamPos.getPointer(), cachedFrames[curFrame].d_cameraposDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 curNormals(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curNormals.getPointer(), cachedFrames[curFrame].d_normalsDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

	// transforms
	std::vector<float4x4> transforms(curFrame);
	cutilSafeCall(cudaMemcpy(transforms.data(), siftManager->d_currFilteredTransforms, sizeof(float4x4) * curFrame, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames
		// get data
		ml::DepthImage32 prvDepth(downSampWidth, downSampHeight);
		cutilSafeCall(cudaMemcpy(prvDepth.getPointer(), cachedFrames[i].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
		ml::ColorImageR8G8B8A8 prvColor(downSampWidth, downSampHeight);
		cutilSafeCall(cudaMemcpy(prvColor.getPointer(), cachedFrames[i].d_colorDownsampled, sizeof(uchar4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
		ml::ColorImageR32G32B32A32 prvCamPos(downSampWidth, downSampHeight);
		cutilSafeCall(cudaMemcpy(prvCamPos.getPointer(), cachedFrames[i].d_cameraposDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
		ml::ColorImageR32G32B32A32 prvNormals(downSampWidth, downSampHeight);
		cutilSafeCall(cudaMemcpy(prvNormals.getPointer(), cachedFrames[i].d_normalsDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

		std::vector<uint2> keyPointIndices;
		siftManager->getFiltKeyPointIndicesDEBUG(i, keyPointIndices);

		//std::cout << "(" << i << ", " << curFrame << "): ";
		bool valid =
			filterImagePairByDenseVerify(prvDepth.getPointer(), (float4*)prvCamPos.getPointer(), (float4*)prvNormals.getPointer(), (uchar4*)prvColor.getPointer(),
			curDepth.getPointer(), (float4*)curCamPos.getPointer(), (float4*)curNormals.getPointer(), (uchar4*)curColor.getPointer(),
			transforms[i], downSampWidth, downSampHeight, depthMin, depthMax);
		//if (valid) std::cout << "VALID" << std::endl;
		//else std::cout << "INVALID" << std::endl;

		if (!valid) {
			// invalidate
			unsigned int numMatches = 0;
			cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
	}
}

bool SIFTMatchFilter::filterImagePairByDenseVerify(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const uchar4* inputColor,
	const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const uchar4* modelColor,
	const float4x4& transform, unsigned int width, unsigned int height, float depthMin, float depthMax)
{

	const float verifySiftErrThresh = GlobalBundlingState::get().s_verifySiftErrThresh;
	const float verifySiftCorrThresh = GlobalBundlingState::get().s_verifySiftCorrThresh;
	float2 projErrors = computeProjectiveError(inputDepth, inputCamPos, inputNormals, inputColor, modelDepth, modelCamPos, modelNormals, modelColor,
		transform, width, height, depthMin, depthMax);

	//std::cout << "proj errors = " << projErrors.x << " " << projErrors.y << std::endl;
	if (projErrors.x == -std::numeric_limits<float>::infinity() || (projErrors.x > verifySiftErrThresh) || (projErrors.y < verifySiftCorrThresh)) { // tracking lost or bad match
		return false; // invalid
	}
	return true; // valid


}

float2 SIFTMatchFilter::computeProjectiveError(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const uchar4* inputColor,
	const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const uchar4* modelColor,
	const float4x4& transform, unsigned int width, unsigned int height, float depthMin, float depthMax)
{
	const unsigned int level = 2;
	const unsigned int subsampleFactor = 1 << level;

	//!!!TODO PARAMS
	const float sigmaD = 3.0f;
	const float sigmaR = 0.1f;
	const float distThres = 0.15f;
	const float normalThres = 0.97f;
	const float colorThresh = 0.1f;

	//jointBilateralFilter(width, height, color0, depth0, s_inputColor, sigmaD, sigmaR); // joint bilateral filter color (w/ depth)
	//jointBilateralFilter(width, newHeight, color1, depth1, s_modelColor, sigmaD, sigmaR); // joint bilateral filter color (w/ depth)}

	// input -> model
	float4x4 transformEstimate = transform;
	float sumResidual0, sumWeight0; unsigned int numCorr0;
	computeCorrespondences(width, height, inputDepth, inputCamPos, inputNormals, inputColor,
		modelDepth, modelCamPos, modelNormals, modelColor,
		transformEstimate, distThres, normalThres, colorThresh, level,
		depthMin, depthMax,	sumResidual0, sumWeight0, numCorr0);

	// model -> input
	transformEstimate = transform.getInverse();
	float sumResidual1, sumWeight1; unsigned int numCorr1;
	computeCorrespondences(width, height, modelDepth, modelCamPos, modelNormals, modelColor,
		inputDepth, inputCamPos, inputNormals, inputColor,
		transformEstimate, distThres, normalThres, colorThresh, level,
		depthMin, depthMax, sumResidual1, sumWeight1, numCorr1);

	float sumRes = (sumResidual0 + sumResidual1) * 0.5f;
	float sumWeight = (sumWeight0 + sumWeight1) * 0.5f;
	unsigned int numCorr = (numCorr0 + numCorr1) / 2;

	return make_float2(sumRes / sumWeight, (float)numCorr / (float)(width * height));
}

void SIFTMatchFilter::computeCameraSpacePositions(const float* depth, unsigned int width, unsigned int height, float4* out)
{
	std::cout << "ERROR hard-coded depth params in " << __FUNCTION__ << std::endl;
	getchar();
	//!!!TODO PARAMS
	const ml::mat4f depthIntrinsicsInverse(
		0.0017152658f, 0.0f, -0.548885047f, 0.0f,
		0.0f, 0.0017152658f, -0.4116638f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
	const unsigned int cameraDepthWidth = 640;
	const unsigned int cameraDepthHeight = 480;

	const float depthScaleWidth = (float)cameraDepthWidth / (float)width;
	const float depthScaleHeight = (float)cameraDepthHeight / (float)height;

	//!!!TOOD HACK
	ml::vec4f* mout = (ml::vec4f*)out;

	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {
			ml::vec2f coords(x * depthScaleWidth, y * depthScaleHeight);
			if (coords.x >= 0 && coords.y >= 0 && coords.x < (int)cameraDepthWidth && coords.y < (int)cameraDepthHeight) {
				float d = depth[y * width + x];
				if (d == -std::numeric_limits<float>::infinity()) mout[y * width + x] = ml::vec4f(-std::numeric_limits<float>::infinity());
				else mout[y * width + x] = ml::vec4f(depthIntrinsicsInverse * (d * ml::vec3f(coords.x, coords.y, 1.0f)), 1.0f);
			}
			else
				mout[y * width + x] = ml::vec4f(-std::numeric_limits<float>::infinity());
		} // x
	} // y
}

void SIFTMatchFilter::computeNormals(const float4* input, unsigned int width, unsigned int height, float4* out)
{
	//!!!TOOD HACK
	ml::vec4f* mout = (ml::vec4f*)out;
	ml::vec4f* minput = (ml::vec4f*)input;

	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {

			mout[y * width + x] = ml::vec4f(-std::numeric_limits<float>::infinity());
			if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
				const ml::vec4f& CC = minput[(y + 0) * width + (x + 0)];//input(x + 0, y + 0);
				const ml::vec4f& PC = minput[(y + 1) * width + (x + 0)];//input(x + 0, y + 1);
				const ml::vec4f& CP = minput[(y + 0) * width + (x + 1)];//input(x + 1, y + 0);
				const ml::vec4f& MC = minput[(y - 1) * width + (x + 0)];//input(x + 0, y - 1);
				const ml::vec4f& CM = minput[(y + 0) * width + (x - 1)];//input(x - 1, y + 0);

				if (CC.x != -std::numeric_limits<float>::infinity() && PC.x != -std::numeric_limits<float>::infinity() && CP.x != -std::numeric_limits<float>::infinity() &&
					MC.x != -std::numeric_limits<float>::infinity() && CM.x != -std::numeric_limits<float>::infinity()) {
					const ml::vec3f n = (PC.getVec3() - MC.getVec3()) ^ (CP.getVec3() - CM.getVec3());
					const float l = n.length();

					if (l > 0.0f)
						mout[y * width + x] = ml::vec4f(n / -l, 1.0f);
				}
			}
		} // x
	} // y
}

void SIFTMatchFilter::reSampleColor(const uchar4* input, unsigned int width, unsigned int height, unsigned int newWidth, unsigned int newHeight, float4* output)
{
	const float scaleWidthFactor = (float)width / (float)newWidth;
	const float scaleHeightFactor = (float)height / (float)newHeight;

	for (unsigned int i = 0; i < newHeight; i++) {
		for (unsigned int j = 0; j < newWidth; j++) {
			const unsigned int x = (unsigned int)std::round((float)j * scaleWidthFactor);
			const unsigned int y = (unsigned int)std::round((float)i * scaleHeightFactor);
			assert(y >= 0 && y < height && x >= 0 && x < width);
			output[i * newWidth + j] = make_float4(input[y * width + x].x / 255.0f, input[y * width + x].y / 255.0f, input[y * width + x].z / 255.0f, 1.0f);
		}
	}
}

void SIFTMatchFilter::jointBilateralFilter(unsigned int width, unsigned int height, const uchar4* color, const float* depth, uchar4* out, float sigmaD, float sigmaR)
{
	const int kernelRadius = (int)ceil(2.0f*sigmaD);

	const float INVALID_DEPTH = -std::numeric_limits<float>::infinity();

	//#pragma omp parallel for
	for (int y = 0; y < (int)height; y++) {
		for (int x = 0; x < (int)width; x++) {

			const unsigned int idx = y * width + x;
			out[idx] = color[idx];

			float3 sum = make_float3(0.0f, 0.0f, 0.0f);
			float sumWeight = 0.0f;
			const uchar4 center = color[idx];

			const float centerDepth = depth[idx];
			if (centerDepth != INVALID_DEPTH) {

				for (int mx = x - kernelRadius; mx <= x + kernelRadius; mx++) {
					for (int ny = y - kernelRadius; ny <= y + kernelRadius; ny++) {
						if (mx >= 0 && ny >= 0 && mx < (int)width && ny < (int)height) {
							const uchar4 cur = color[ny * width + mx];
							float depthDist = 0.0;
							const float curDepth = depth[ny * width + mx];
							if (centerDepth != INVALID_DEPTH && curDepth != INVALID_DEPTH) {
								depthDist = fabs(centerDepth - curDepth);

								if (depthDist < sigmaR) {
									const float weight = gaussD(sigmaD, mx - x, ny - y); // pixel distance

									sumWeight += weight;
									sum += weight*make_float3(cur.x, cur.y, cur.z);
								}
							}
						} // inside image bounds
					}
				} // kernel
			}
			if (sumWeight > 0) {
				float3 res = sum / sumWeight;
				out[idx] = make_uchar4((uchar)res.x, (uchar)res.y, (uchar)res.z, 255);
			}

		} // x
	} // y
}

float2 SIFTMatchFilter::cameraToDepth(const float4& pos)
{
	const ml::mat4f depthIntrinsics(583.0f, 0.0f, 320.0f, 0.0f,
		0.f, 583.0f, 240.f, 0.0f,
		0.f, 0.f, 1.f, 0.0f,
		0.f, 0.f, 0.f, 1.0f);
	ml::vec3f p = depthIntrinsics * ml::vec3f(pos.x, pos.y, pos.z);
	return make_float2(p.x / p.z, p.y / p.z);
}

float SIFTMatchFilter::cameraToDepthZ(const float4& pos)
{
	const ml::mat4f depthIntrinsics(572.0f, 0.0f, 320.0f, 0.0f,
		0.f, 572.0f, 240.f, 0.0f,
		0.f, 0.f, 1.f, 0.0f,
		0.f, 0.f, 0.f, 1.0f);

	return (depthIntrinsics * ml::vec3f(pos.x, pos.y, pos.z)).z;
}

void SIFTMatchFilter::computeCorrespondences(unsigned int width, unsigned int height,
	const float* inputDepth, const float4* input, const float4* inputNormals, const uchar4* inputColor,
	const float* modelDepth, const float4* model, const float4* modelNormals, const uchar4* modelColor,
	const float4x4& transform, float distThres, float normalThres, float colorThresh, unsigned int level,
	float depthMin, float depthMax,
	float& sumResidual, float& sumWeight, unsigned int& numCorr)
{
	const float INVALID = -std::numeric_limits<float>::infinity();
	sumResidual = 0.0f;
	sumWeight = 0.0f;
	numCorr = 0;

	float levelFactor = (float)(1 << level);
	//mean = vec3f(0.0f, 0.0f, 0.0f);
	//meanStDev = 1.0f;
	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {

			const unsigned int idx = y * width + x;

			float4 pInput = input[idx]; // point
			float4 nInput = inputNormals[idx]; nInput.w = 0.0f; // vector
			uchar4 cInput = inputColor[idx];

			if (pInput.x != INVALID && nInput.x != INVALID) {
				const float4 pTransInput = transform * pInput;
				const float4 nTransInput = transform * nInput;

				float2 screenPosf = cameraToDepth(pTransInput);
				//int2 screenPos = make_int2((int)(round(screenPosf.x) / levelFactor), (int)(round(screenPosf.y) / levelFactor)); // subsampled space
				int2 screenPos = make_int2((int)round(screenPosf.x / levelFactor), (int)round(screenPosf.y / levelFactor)); // subsampled space

				if (screenPos.x >= 0 && screenPos.y >= 0 && screenPos.x < (int)width && screenPos.y < (int)height) {
					float4 pTarget; float4 nTarget; uchar4 cTarget;
					getBestCorrespondence1x1(screenPos, pTarget, nTarget, cTarget, model, modelNormals, modelColor, width);

					if (pTarget.x != INVALID && nTarget.x != INVALID) {
						float d = length(pTransInput - pTarget);
						float dNormal = dot(make_float3(nTransInput.x, nTransInput.y, nTransInput.z), make_float3(nTarget.x, nTarget.y, nTarget.z)); // should be able to do dot(nTransInput, nTarget)
						//float c = length(make_float3((cInput.x - cTarget.x) / 255.0f, (cInput.y - cTarget.y) / 255.0f, (cInput.z - cTarget.z) / 255.0f));

						float projInputDepth = pTransInput.z;//cameraToDepthZ(pTransInput);
						float tgtDepth = modelDepth[screenPos.y * width + screenPos.x];

						bool b = ((tgtDepth != INVALID && projInputDepth < tgtDepth) && d > distThres); // bad matches that are known
						if ((dNormal >= normalThres && d <= distThres /*&& c <= colorThresh*/) || b) { // if normal/pos/color correspond or known bad match
							const float weight = std::max(0.0f, 0.5f*((1.0f - d / distThres) + (1.0f - cameraToKinectProjZ(pTransInput.z, depthMin, depthMax)))); // for weighted ICP;

							sumResidual += length(pTransInput - pTarget);	//residual
							sumWeight += weight;			//corr weight
							numCorr++;					//corr number
						}
					} // projected to valid depth
				} // inside image
			}
		} // x
	} // y

}

void SIFTMatchFilter::ransacKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2, bool debugPrint)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	if (!s_bInit) {
		std::cout << "warning: initializing combinations" << std::endl;
		Timer t;
		init();
		std::cout << "init: " << t.getElapsedTimeMS() << " ms" << std::endl;
	}

	// current data
	const unsigned int curFrame = numImages - 1;
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	std::vector<float4x4> transforms(curFrame);
	std::vector<float4x4> transformsInv(curFrame);

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames

		std::vector<uint2> keyPointIndices;
		std::vector<float> matchDistances;
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

		if (debugPrint) std::cout << "(" << i << ", " << curFrame << ")" << std::endl;
		float4x4 transform;
		unsigned int newNumMatches =
			filterImagePairKeyPointMatchesRANSAC(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv, minNumMatches,
			maxResThresh2, (unsigned int)s_combinations[0].size(), s_combinations, debugPrint);

		if (newNumMatches > 0) {
			transforms[i] = transform;
			transformsInv[i] = transform.getInverse();
		}
		else {
			transforms[i].setValue(0.0f);
			transformsInv[i].setValue(0.0f);
		}

		// copy back
		cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		if (newNumMatches > 0) {
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
		}
	}
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
}

#define REFINE_RANSAC
unsigned int SIFTMatchFilter::filterImagePairKeyPointMatchesRANSAC(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances,
	float4x4& transform, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxResThresh2,
	unsigned int k, const std::vector<std::vector<unsigned int>>& combinations, bool debugPrint)
{
	unsigned int numRawMatches = (unsigned int)keyPointIndices.size();
	if (numRawMatches < minNumMatches) return 0;

	//const unsigned int ransacMax = 100;
	//std::vector<float> errors(ransacMax);
	//std::vector<vec4uc> indices(ransacMax);

	//for (unsigned int i = 0; i < ransacMax; i++) {

	//} // tests


	unsigned int numValidCombs = 0;
	unsigned int numValidStarts = 0;

	// RANSAC TEST
	unsigned int maxNumInliers = 0;
	float bestMaxResidual = std::numeric_limits<float>::infinity();
	std::vector<unsigned int> bestCombinationIndices;
	for (unsigned int c = 0; c < combinations.size(); c++) {
		std::vector<unsigned int> indices = combinations[c];

		bool _DEBUGCOMB = debugPrint && indices[0] == 0 && indices[1] == 2 && indices[2] == 4 && indices[3] == 5;
		if (_DEBUGCOMB) std::cout << "combination at " << c << std::endl;

		// check if has combination
		bool validCombination = true;
		for (unsigned int i = 0; i < indices.size(); i++) {
			if (indices[i] >= numRawMatches) {
				validCombination = false;
				break;
			}
		}
		if (!validCombination) continue;
		numValidCombs++;

		std::vector<float3> srcPts(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED), tgtPts(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		getKeySourceAndTargetPointsByIndices(keys.data(), keyPointIndices.data(), indices.data(), k, srcPts.data(), tgtPts.data(), siftIntrinsicsInv);
		unsigned int curNumMatches = k;
		float3 eigenvalues; float4x4 transformEstimate;
		float curMaxResidual = computeKabschReprojError(srcPts.data(), tgtPts.data(), curNumMatches, eigenvalues, transformEstimate);
		if (_DEBUGCOMB) {
			std::cout << "src pts:" << std::endl;
			for (unsigned int i = 0; i < curNumMatches; i++) std::cout << "\t" << srcPts[i].x << " " << srcPts[i].y << " " << srcPts[i].z << std::endl;
			std::cout << "tgt pts:" << std::endl;
			for (unsigned int i = 0; i < curNumMatches; i++) std::cout << "\t" << tgtPts[i].x << " " << tgtPts[i].y << " " << tgtPts[i].z << std::endl;
			std::cout << "max res = " << curMaxResidual <<  "(cond = " << eigenvalues.x / eigenvalues.y << ")" << std::endl;
			getchar();
		}

		numValidStarts++;

		std::vector<bool> marker(numRawMatches, false);
		for (unsigned int i = 0; i < indices.size(); i++) marker[indices[i]] = true;
		// collect inliers
		for (unsigned int m = 0; m < numRawMatches; m++) {
			if (curNumMatches == MAX_MATCHES_PER_IMAGE_PAIR_FILTERED) break;
			if (marker[m]) continue;

			// don't add if within +/- 5 pix
			const float2& potential0 = keys[keyPointIndices[m].x].pos;
			const float2& potential1 = keys[keyPointIndices[m].y].pos;
			bool add = true;
			for (unsigned int p = 0; p < curNumMatches; p++) {
				if (length(potential0 - keys[keyPointIndices[indices[p]].x].pos) <= 5.0f ||
					length(potential1 - keys[keyPointIndices[indices[p]].y].pos) <= 5.0f) {
					add = false;
					break;
				}
			}
			if (!add) {
				if (_DEBUGCOMB) printf("match %d too close to previous\n", m);
				continue;
			}

			getKeySourceAndTargetPointsForIndex(keys.data(), keyPointIndices.data(), m, srcPts.data() + curNumMatches, tgtPts.data() + curNumMatches, siftIntrinsicsInv);
#ifdef REFINE_RANSAC
			// refine transform
			float curRes2 = computeKabschReprojError(srcPts.data(), tgtPts.data(), curNumMatches + 1, eigenvalues, transformEstimate);
			if (_DEBUGCOMB) {
				std::cout << m << std::endl;
				std::cout << "src pts:" << std::endl;
				for (unsigned int i = 0; i < curNumMatches + 1; i++) std::cout << "\t" << srcPts[i].x << " " << srcPts[i].y << " " << srcPts[i].z << std::endl;
				std::cout << "tgt pts:" << std::endl;
				for (unsigned int i = 0; i < curNumMatches + 1; i++) std::cout << "\t" << tgtPts[i].x << " " << tgtPts[i].y << " " << tgtPts[i].z << std::endl;
				std::cout << "max res = " << curRes2 << "(cond = " << eigenvalues.x / eigenvalues.y << ")" << std::endl;
				getchar();
			}
#else 
			float3 d = transformEstimate * srcPts[curNumMatches] - tgtPts[curNumMatches];
			float curRes2 = dot(d, d);
#endif
			//if (debugPrint && c == 0) {
			//	std::cout << "m = " << m << ", cur # matches = " << curNumMatches << ", new res = " << std::sqrt(curRes2) << std::endl;
			//}

			//if (eigenvalues.x / eigenvalues.y <= KABSCH_CONDITION_THRESH && curRes2 <= maxResThresh2) { // inlier
			if (curRes2 <= maxResThresh2) { // inlier
				curNumMatches++;
				indices.push_back(m);
#ifdef REFINE_RANSAC
				curMaxResidual = curRes2;
#else
				if (curRes2 > curMaxResidual) curMaxResidual = curRes2;
#endif
			}
		}
		if (_DEBUGCOMB) {
			printf("cur #matches %d vs %d, cur res %f vs %f\n", curNumMatches, maxNumInliers, curMaxResidual, bestMaxResidual);
			printf("cur match set:"); for (unsigned int i = 0; i < curNumMatches; i++) printf(" %d", indices[i]); printf("\n");
			printf("cur best match set:"); for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) printf(" %d", bestCombinationIndices[i]); printf("\n");
		}
		if (curNumMatches > maxNumInliers || (curNumMatches == maxNumInliers && curMaxResidual < bestMaxResidual)) {
			maxNumInliers = curNumMatches;
			bestCombinationIndices = indices;
			bestMaxResidual = curMaxResidual;

			if (_DEBUGCOMB) std::cout << "POTENTIAL MATCH" << std::endl;
			if (debugPrint) {
				std::cout << "adding potential match set (" << curNumMatches << ") res " << curMaxResidual << ", cond " << (eigenvalues.x / eigenvalues.y) << std::endl;
				std::cout << "\t";
				for (unsigned int i = 0; i < indices.size(); i++) std::cout << " " << indices[i];
				std::cout << std::endl;
			}
		}
	}

	if (debugPrint) {
		std::cout << "#valid com = " << numValidCombs << ", #valid starts = " << numValidStarts << std::endl;
		std::cout << "#raw matches = " << numRawMatches << ", max # inliers = " << maxNumInliers << std::endl;
	}

	if (maxNumInliers >= minNumMatches) {
		std::vector<float3> srcPts(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED), tgtPts(MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		getKeySourceAndTargetPointsByIndices(keys.data(), keyPointIndices.data(), bestCombinationIndices.data(), maxNumInliers, srcPts.data(), tgtPts.data(), siftIntrinsicsInv);
		float3 eigenvalues;
		transform = kabschReference(srcPts.data(), tgtPts.data(), maxNumInliers, eigenvalues);

		if (debugPrint) {
			std::cout << "potential indices: "; for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) std::cout << bestCombinationIndices[i] << " "; std::cout << std::endl;
			//std::cout << "final transform estimate:" << std::endl;
			//transform.print();
			//std::cout << "final cond = " << eigenvalues.x / eigenvalues.y << std::endl;
			std::cout << "key indices:" << std::endl;
			for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) {
				std::cout << "\t" << keyPointIndices[bestCombinationIndices[i]].x << " " << keyPointIndices[bestCombinationIndices[i]].y << std::endl;
			}
			getchar();
		}

		float c1 = eigenvalues.x / eigenvalues.y; // ok if coplanar
		eigenvalues = covarianceSVDReference(srcPts.data(), maxNumInliers);
		float cp = eigenvalues.x / eigenvalues.y; // ok if coplanar
		eigenvalues = covarianceSVDReference(tgtPts.data(), maxNumInliers);
		float cq = eigenvalues.x / eigenvalues.y; // ok if coplanar

		if (isnan(c1) || isnan(cp) || isnan(cq) || fabs(c1) > KABSCH_CONDITION_THRESH || fabs(cp) > KABSCH_CONDITION_THRESH || fabs(cq) > KABSCH_CONDITION_THRESH) {
			if (debugPrint) {
				std::cout << "failed condition test (" << c1 << ", " << cp << ", " << cq << ")" << std::endl;
				getchar();
			}
			return 0;
		}
		else {
			unsigned int maxBestComboIndex = 0;
			for (unsigned int i = 0; i < k; i++) {
				if (bestCombinationIndices[i] > maxBestComboIndex)
					maxBestComboIndex = bestCombinationIndices[i];
			}
			if (debugPrint) {
				std::cout << "max index = " << maxBestComboIndex << std::endl;
				std::cout << "\t#inliers = " << maxNumInliers << ", max res = " << bestMaxResidual << std::endl;
				std::cout << "final indices: "; for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) std::cout << bestCombinationIndices[i] << " "; std::cout << std::endl;
				getchar();
			}

			uint2 newKeyIndices[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED];
			float newMatchDists[MAX_MATCHES_PER_IMAGE_PAIR_FILTERED]; //TODO don't actually need this now
			for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) {
				newKeyIndices[i] = keyPointIndices[bestCombinationIndices[i]];
				newMatchDists[i] = matchDistances[bestCombinationIndices[i]];
			}
			for (unsigned int i = 0; i < bestCombinationIndices.size(); i++) {
				keyPointIndices[i] = newKeyIndices[i];
				matchDistances[i] = newMatchDists[i];
			}
			return maxNumInliers;
		}
	}
	return 0;
}

void SIFTMatchFilter::ransacKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv,
	unsigned int minNumMatches, float maxResThresh2, bool debugPrint)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	if (!s_bInit) {
		std::cout << "warning: initializing combinations" << std::endl;
		Timer t;
		init();
		std::cout << "init: " << t.getElapsedTimeMS() << " ms" << std::endl;
	}

	// current data
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	std::vector<float4x4> transforms(curFrame);
	std::vector<float4x4> transformsInv(curFrame);

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames

		std::vector<uint2> keyPointIndices;
		std::vector<float> matchDistances;
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

		//!!!DEBUGGING
		//if (curFrame == 125 && i == 104)
		//	debugPrint = true;
		//else debugPrint = false;
		//!!!DEBUGGING
		if (debugPrint) std::cout << "(" << i << ", " << curFrame << ")" << std::endl;
		float4x4 transform;
		unsigned int newNumMatches =
			filterImagePairKeyPointMatchesRANSAC(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv,
			minNumMatches, maxResThresh2, (unsigned int)s_combinations[0].size(), s_combinations, debugPrint);

		if (newNumMatches > 0) {
			transforms[i] = transform;
			transformsInv[i] = transform.getInverse();
		}
		else {
			transforms[i].setValue(0.0f);
			transformsInv[i].setValue(0.0f);
		}

		// copy back
		cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		if (newNumMatches > 0) {
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
		}
	}
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
}

void SIFTMatchFilter::filterKeyPointMatchesDEBUG(unsigned int curFrame, SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv,
	unsigned int minNumMatches, float maxResThresh2, bool printDebug)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	// current data
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	std::vector<float4x4> transforms(curFrame);
	std::vector<float4x4> transformsInv(curFrame);

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames

		std::vector<uint2> keyPointIndices;
		std::vector<float> matchDistances;
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

		//!!!DEBUGGING
		//if (curFrame == 125 && i == 104) {
		//	printDebug = true;
		//}
		//else printDebug = false;
		//!!!DEBUGGING

		float4x4 transform;
		unsigned int newNumMatches =
			filterImagePairKeyPointMatches(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv, minNumMatches, maxResThresh2, printDebug);
		//std::cout << "(" << curFrame << ", " << i << "): " << newNumMatches << std::endl; 

		if (newNumMatches > 0) {
			transforms[i] = transform;
			transformsInv[i] = transform.getInverse();
		}
		else {
			transforms[i].setValue(0.0f);
			transformsInv[i].setValue(0.0f);
		}

		// copy back
		cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		if (newNumMatches > 0) {
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
		}
	}
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransformsInv, transformsInv.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
}


void SIFTMatchFilter::visualizeProjError(SIFTImageManager* siftManager, const vec2ui& imageIndices, const std::vector<CUDACachedFrame>& cachedFrames, float depthMin, float depthMax)
{
	const unsigned int numImages = siftManager->getNumImages();
	MLIB_ASSERT(imageIndices.x < numImages && imageIndices.y < numImages);

	const unsigned int downSampWidth = GlobalBundlingState::get().s_downsampledWidth;
	const unsigned int downSampHeight = GlobalBundlingState::get().s_downsampledHeight;

	// current data
	ml::DepthImage32 curDepth(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curDepth.getPointer(), cachedFrames[imageIndices.y].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR8G8B8A8 curColor(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curColor.getPointer(), cachedFrames[imageIndices.y].d_colorDownsampled, sizeof(uchar4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 curCamPos(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curCamPos.getPointer(), cachedFrames[imageIndices.y].d_cameraposDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 curNormals(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(curNormals.getPointer(), cachedFrames[imageIndices.y].d_normalsDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

	// transforms
	float4x4 transform;
	cutilSafeCall(cudaMemcpy(&transform, siftManager->d_currFilteredTransforms + imageIndices.x, sizeof(float4x4), cudaMemcpyDeviceToHost));

	// prev data
	ml::DepthImage32 prvDepth(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(prvDepth.getPointer(), cachedFrames[imageIndices.x].d_depthDownsampled, sizeof(float) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR8G8B8A8 prvColor(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(prvColor.getPointer(), cachedFrames[imageIndices.x].d_colorDownsampled, sizeof(uchar4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 prvCamPos(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(prvCamPos.getPointer(), cachedFrames[imageIndices.x].d_cameraposDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));
	ml::ColorImageR32G32B32A32 prvNormals(downSampWidth, downSampHeight);
	cutilSafeCall(cudaMemcpy(prvNormals.getPointer(), cachedFrames[imageIndices.x].d_normalsDownsampled, sizeof(float4) * downSampWidth * downSampHeight, cudaMemcpyDeviceToHost));

	std::vector<uint2> keyPointIndices;
	siftManager->getFiltKeyPointIndicesDEBUG(imageIndices.x, keyPointIndices);

	const float verifyOptErrThresh = GlobalBundlingState::get().s_verifyOptErrThresh;
	const float verifyOptCorrThresh = GlobalBundlingState::get().s_verifyOptCorrThresh;

	//!!!TODO PARAMS
	const float distThres = 0.15f;
	const float normalThres = 0.97f;
	const float colorThresh = 0.1f;
	const unsigned int level = 2;

	//TODO HERE
	// input -> model
	float4x4 transformEstimate = transform;
	float sumResidual0, sumWeight0; unsigned int numCorr0;
	computeCorrespondencesDEBUG(downSampWidth, downSampHeight, curDepth.getPointer(), (float4*)curCamPos.getPointer(), (float4*)curNormals.getPointer(), (uchar4*)curColor.getPointer(),
		prvDepth.getPointer(), (float4*)prvCamPos.getPointer(), (float4*)prvNormals.getPointer(), (uchar4*)prvColor.getPointer(),
		transformEstimate, distThres, normalThres, colorThresh, level,
		depthMin, depthMax, sumResidual0, sumWeight0, numCorr0);

	FreeImageWrapper::saveImage("debug/projCorr-0.png", ColorImageR32G32B32(s_debugCorr));

	// model -> input
	transformEstimate = transform.getInverse();
	float sumResidual1, sumWeight1; unsigned int numCorr1;
	computeCorrespondencesDEBUG(downSampWidth, downSampHeight, prvDepth.getPointer(), (float4*)prvCamPos.getPointer(), (float4*)prvNormals.getPointer(), (uchar4*)prvColor.getPointer(),
		curDepth.getPointer(), (float4*)curCamPos.getPointer(), (float4*)curNormals.getPointer(), (uchar4*)curColor.getPointer(),
		transformEstimate, distThres, normalThres, colorThresh, level,
		depthMin, depthMax, sumResidual1, sumWeight1, numCorr1);

	FreeImageWrapper::saveImage("debug/projCorr-1.png", ColorImageR32G32B32(s_debugCorr));

	float sumRes = (sumResidual0 + sumResidual1) * 0.5f;
	float sumWeight = (sumWeight0 + sumWeight1) * 0.5f;
	unsigned int numCorr = (numCorr0 + numCorr1) / 2;

	float2 projErrors = make_float2(sumRes / sumWeight, (float)numCorr / (float)(downSampWidth * downSampHeight));

	bool valid = true;
	if (projErrors.x == -std::numeric_limits<float>::infinity() || (projErrors.x > verifyOptErrThresh) || (projErrors.y < verifyOptCorrThresh)) { // tracking lost or bad match
		valid = false; // invalid
	}
}