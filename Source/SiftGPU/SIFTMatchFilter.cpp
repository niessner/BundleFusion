#include "stdafx.h"
#include "SIFTMatchFilter.h"

#include "cuda_kabschReference.h"
//#include "cuda_kabsch.h"
#include "../GlobalBundlingState.h"

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

void SIFTMatchFilter::filterKeyPointMatches(SIFTImageManager* siftManager, const float4x4& siftIntrinsicsInv)
{
	const unsigned int numImages = siftManager->getNumImages();
	if (numImages <= 1) return;

	// current data
	const unsigned int curFrame = numImages - 1;
	std::vector<SIFTKeyPoint> keyPoints;
	siftManager->getSIFTKeyPointsDEBUG(keyPoints);
	std::vector<float4x4> transforms(curFrame);

	for (unsigned int i = 0; i < curFrame; i++) { // previous frames

		std::vector<uint2> keyPointIndices;
		std::vector<float> matchDistances;
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(i, keyPointIndices, matchDistances);

		float4x4 transform;
		unsigned int newNumMatches = 
			filterImagePairKeyPointMatches(keyPoints, keyPointIndices, matchDistances, transform, siftIntrinsicsInv);
		//std::cout << "(" << curFrame << ", " << i << "): " << newNumMatches << std::endl; 

		transforms[i] = transform;

		// copy back
		cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &newNumMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		if (newNumMatches > 0) {
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchKeyPointIndices + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, keyPointIndices.data(), sizeof(uint2) * newNumMatches, cudaMemcpyHostToDevice));
			cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredMatchDistances + i * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, matchDistances.data(), sizeof(float) * newNumMatches, cudaMemcpyHostToDevice));
		}
	}
	cutilSafeCall(cudaMemcpy(siftManager->d_currFilteredTransforms, transforms.data(), sizeof(float4x4) * curFrame, cudaMemcpyHostToDevice));
}

unsigned int SIFTMatchFilter::filterImagePairKeyPointMatches(const std::vector<SIFTKeyPoint>& keys, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances, float4x4& transform, const float4x4& siftIntrinsicsInv)
{
	unsigned int numRawMatches = (unsigned int)keyPointIndices.size();
	if (numRawMatches < MIN_NUM_MATCHES_FILTERED) return 0;

	//std::vector<SIFTKeyPoint> copy_keys0 = keys0;
	//std::vector<SIFTKeyPoint> copy_keys1 = keys1;
	//std::vector<uint2> copy_keyPointIndices = keyPointIndices;
	//std::vector<float> copy_matchDistances = matchDistances;
	//float4x4 copy_transform = transform;
	//unsigned int numTry = ::filterKeyPointMatches(keys0.data(),keys1.data(), keyPointIndices.data(), matchDistances.data(), numRawMatches, transform);
	//unsigned int numRef = filterKeyPointMatchesReference(copy_keys0.data(), copy_keys1.data(), copy_keyPointIndices.data(), copy_matchDistances.data(), numRawMatches, copy_transform);

	unsigned int numRef = filterKeyPointMatchesReference(keys.data(), keyPointIndices.data(), matchDistances.data(), numRawMatches, transform, siftIntrinsicsInv);
	return numRef;

	//unsigned int numTry = ::filterKeyPointMatches(keys.data(), keyPointIndices.data(), matchDistances.data(), numRawMatches, transform);
	//return numTry;
}

void SIFTMatchFilter::filterBySurfaceArea(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames, const float4x4& siftIntrinsicsInv)
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
			filterImagePairBySurfaceArea(keyPoints, prvDepth.getPointer(), curDepth.getPointer(), keyPointIndices, siftIntrinsicsInv);

		if (!valid) {
			// invalidate
			unsigned int numMatches = 0;
			cutilSafeCall(cudaMemcpy(siftManager->d_currNumFilteredMatchesPerImagePair + i, &numMatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
	}
}

bool SIFTMatchFilter::filterImagePairBySurfaceArea(const std::vector<SIFTKeyPoint>& keys, float* depth0, float* depth1, const std::vector<uint2>& keyPointIndices, const float4x4& siftIntrinsicsInv)
{
	if (keyPointIndices.size() < MIN_NUM_MATCHES_FILTERED) return false;

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

	////!!!TODO PARAMS
	//const ml::mat4f depthIntrinsicsInverse(
	//	0.0017152658f, 0.0f, -0.548885047f, 0.0f,
	//	0.0f, 0.0017152658f, -0.4116638f, 0.0f,
	//	0.0f, 0.0f, 1.0f, 0.0f,
	//	0.0f, 0.0f, 0.0f, 1.0f);
	//const unsigned int depthWidth = 640;
	//const unsigned int depthHeight = 480;
	//const unsigned int subDepthWidth = GlobalBundlingState::get().s_downsampledWidth;
	//const unsigned int subDepthHeight = GlobalBundlingState::get().s_downsampledHeight;
	//const float depthMin = 0.1f;
	//const float depthMax = 3.0f;
	//const unsigned int subSampleFactor = 2; 

	//const unsigned int newWidth = subDepthWidth / subSampleFactor;
	//const unsigned int newHeight = subDepthHeight / subSampleFactor;
	//const unsigned int numSampledPoints = newWidth * newHeight;
	//const float widthFactor = (float)depthWidth / (float)subDepthWidth;
	//const float heightFactor = (float)depthHeight / (float)subDepthHeight;
	////const ml::vec2f defaultSrcPt = srcPtsProj[0];// if invalid fill with this for pca... //!!!TODO check
	////const ml::vec2f defaultTgtPt = tgtPtsProj[0];// if invalid fill with this for pca... //!!!TODO check

	//const float maxDepth = 1.0f;

	//// all points projected (subsampled)
	//std::vector<ml::vec2f> allPointsSrc;//(numSampledPoints);
	//std::vector<ml::vec2f> allPointsTgt;//(numSampledPoints);
	//for (unsigned int y = 0; y < subDepthHeight; y += subSampleFactor) {
	//	for (unsigned int x = 0; x < subDepthWidth; x += subSampleFactor) {
	//		//const unsigned int idx = (y / subSampleFactor) * newWidth + (x / subSampleFactor);
	//		unsigned int yi = math::round(y * heightFactor);
	//		unsigned int xi = math::round(x * widthFactor);

	//		float sdepth = depth0[y * subDepthWidth + x];
	//		if (sdepth != -std::numeric_limits<float>::infinity() && sdepth >= depthMin && sdepth <= depthMax) {
	//			if (sdepth > maxDepth) sdepth = maxDepth;
	//			// depth to camera
	//			ml::vec3f cSrc = depthIntrinsicsInverse * (sdepth * ml::vec3f((float)xi, (float)yi, 1.0f));
	//			ml::vec3f cProj3dSrc = cSrc - (srcNormal | (cSrc - srcMean)) * srcNormal;
	//			ml::vec3f s = cProj3dSrc - srcMean;
	//			//allPointsSrc[idx] = ml::vec2f(s | srcAxes[0].first, s | srcAxes[1].first);
	//			allPointsSrc.push_back(ml::vec2f(s | srcAxes[0].first, s | srcAxes[1].first));
	//		}
	//		//else 
	//		//	allPointsSrc[idx] = defaultSrcPt;
	//		float tdepth = depth1[y * subDepthWidth + x];
	//		if (tdepth != -std::numeric_limits<float>::infinity() && tdepth >= depthMin && tdepth <= depthMax) {
	//			if (tdepth > maxDepth) tdepth = maxDepth;
	//			ml::vec3f cTgt = depthIntrinsicsInverse * (tdepth * ml::vec3f((float)xi, (float)yi, 1.0f));
	//			ml::vec3f cProj3dTgt = cTgt - (tgtNormal | (cTgt - tgtMean)) * tgtNormal;
	//			ml::vec3f s = cProj3dTgt - tgtMean;
	//			//allPointsTgt[idx] = ml::vec2f(s | tgtAxes[0].first, s | tgtAxes[1].first);
	//			allPointsTgt.push_back(ml::vec2f(s | tgtAxes[0].first, s | tgtAxes[1].first));
	//		}
	//		//else
	//		//	allPointsTgt[idx] = defaultTgtPt;
	//	}
	//}

	// compute areas
	ml::OBB2f srcOBB(srcPtsProj);
	ml::OBB2f tgtOBB(tgtPtsProj);
	area = ml::vec2f((float)srcOBB.getArea(), (float)tgtOBB.getArea());

	//ml::OBB2f srcAllOBB(allPointsSrc);
	//ml::OBB2f tgtAllOBB(allPointsTgt);
	//totalAreas = ml::vec2f((float)srcAllOBB.getArea(), (float)tgtAllOBB.getArea());

	//totalAreas = ml::math::max(totalAreas, 0.001f);
	////std::cout << "total areas = " << totalAreas.x << " " << totalAreas.y << std::endl;

	//return make_float2(area.x / totalAreas.x, area.y / totalAreas.y);
	return make_float2(area.x, area.y);
}

void SIFTMatchFilter::filterByDenseVerify(SIFTImageManager* siftManager, const std::vector<CUDACachedFrame>& cachedFrames)
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
			transforms[i], downSampWidth, downSampHeight);
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
	const float4x4& transform, unsigned int width, unsigned int height)
{

	//!!!TODO PARAMS
	const float verifySiftErrThresh = 0.075f;
	const float verifySiftCorrThresh = 0.02f;
	float2 projErrors = computeProjectiveError(inputDepth, inputCamPos, inputNormals, inputColor, modelDepth, modelCamPos, modelNormals, modelColor, transform, width, height);

	//std::cout << "proj errors = " << projErrors.x << " " << projErrors.y << std::endl;
	if (projErrors.x == -std::numeric_limits<float>::infinity() || (projErrors.x > verifySiftErrThresh) || (projErrors.y < verifySiftCorrThresh)) { // tracking lost or bad match
		return false; // invalid
	}
	return true; // valid


}

float2 SIFTMatchFilter::computeProjectiveError(const float* inputDepth, const float4* inputCamPos, const float4* inputNormals, const uchar4* inputColor,
	const float* modelDepth, const float4* modelCamPos, const float4* modelNormals, const uchar4* modelColor,
	const float4x4& transform, unsigned int width, unsigned int height)
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
		sumResidual0, sumWeight0, numCorr0);

	// model -> input
	transformEstimate = transform.getInverse();
	float sumResidual1, sumWeight1; unsigned int numCorr1;
	computeCorrespondences(width, height, modelDepth, modelCamPos, modelNormals, modelColor,
		inputDepth, inputCamPos, inputNormals, inputColor,
		transformEstimate, distThres, normalThres, colorThresh, level,
		sumResidual1, sumWeight1, numCorr1);

	float sumRes = (sumResidual0 + sumResidual1) * 0.5f;
	float sumWeight = (sumWeight0 + sumWeight1) * 0.5f;
	unsigned int numCorr = (numCorr0 + numCorr1) / 2;

	return make_float2(sumRes / sumWeight, (float)numCorr / (float)(width * height));
}

void SIFTMatchFilter::computeCameraSpacePositions(const float* depth, unsigned int width, unsigned int height, float4* out)
{
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

						float projInputDepth = cameraToDepthZ(pTransInput);
						float tgtDepth = modelDepth[screenPos.y * width + screenPos.x];

						bool b = ((tgtDepth != INVALID && projInputDepth < tgtDepth) && d > distThres); // bad matches that are known
						if ((dNormal >= normalThres && d <= distThres /*&& c <= colorThresh*/) || b) { // if normal/pos/color correspond or known bad match
							const float weight = std::max(0.0f, 0.5f*((1.0f - d / distThres) + (1.0f - cameraToKinectProjZ(pTransInput.z)))); // for weighted ICP;

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

