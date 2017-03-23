#include "stdafx.h"
#include "SiftVisualization.h"
#include "ImageHelper.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"
#include "CUDAImageManager.h"


void SiftVisualization::printKey(const std::string& filename, const CUDACache* cudaCache, const SIFTImageManager* siftManager, unsigned int frame)
{
	const std::vector<CUDACachedFrame>& frames = cudaCache->getCacheFrames();
	ColorImageR32 intensityImage(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityImage.getData(), frames[frame].d_intensityDownsampled, sizeof(float)*intensityImage.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 image(intensityImage);
	image.resize(GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT);

	printKey(filename, image, siftManager, frame);
}

void SiftVisualization::printKey(const std::string& filename, CUDAImageManager* cudaImageManager, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame)
{
	//TODO get color cpu for these functions
	CUDAImageManager::ManagedRGBDInputFrame& integrateFrame = cudaImageManager->getIntegrateFrame(allFrame);

	ColorImageR8G8B8A8 tmp(cudaImageManager->getIntegrationWidth(), cudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(tmp.getData(), integrateFrame.getColorFrameGPU(), sizeof(uchar4) * cudaImageManager->getIntegrationWidth() * cudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 im(cudaImageManager->getIntegrationWidth(), cudaImageManager->getIntegrationHeight());
	for (unsigned int i = 0; i < tmp.getNumPixels(); i++) im.getData()[i] = tmp.getData()[i].getVec3();
	im.resize(GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT);

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];
		RGBColor c = RGBColor::randomColor();
		vec3uc color(c.r, c.g, c.b);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}


void SiftVisualization::printKey(const std::string& filename, const ColorImageR8G8B8& image, const SIFTImageManager* siftManager, unsigned int frame)
{
	ColorImageR8G8B8 im = image;

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];
		RGBColor c = RGBColor::randomColor();
		vec3uc color(c.r, c.g, c.b);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}

void SiftVisualization::printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
	const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, float distMax, bool filtered, int maxNumMatches)
{
	// get data
	std::vector<SIFTKeyPoint> keys;
	siftManager->getSIFTKeyPointsDEBUG(keys); // prev frame

	std::vector<uint2> keyPointIndices;
	std::vector<float> matchDistances;
	if (filtered) {
		siftManager->getFiltKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	else {
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	if (keyPointIndices.size() == 0) return;

	ColorImageR32G32B32 matchImage(image1.getWidth() * 2, image1.getHeight());
	ColorImageR32G32B32 im1(image1);
	ColorImageR32G32B32 im2(image2);
	matchImage.copyIntoImage(im1, 0, 0);
	matchImage.copyIntoImage(im2, image1.getWidth(), 0);

	if (maxNumMatches < 0) maxNumMatches = (int)keyPointIndices.size();
	else maxNumMatches = std::min((int)keyPointIndices.size(), maxNumMatches);
	float maxMatchDistance = 0.0f;
	RGBColor lowColor = ml::RGBColor::Blue;
	RGBColor highColor = ml::RGBColor::Red;
	for (int i = 0; i < maxNumMatches; i++) {
		const SIFTKeyPoint& key1 = keys[keyPointIndices[i].x];
		const SIFTKeyPoint& key2 = keys[keyPointIndices[i].y];
		if (matchDistances[i] > maxMatchDistance) maxMatchDistance = matchDistances[i];

		RGBColor c = RGBColor::interpolate(lowColor, highColor, matchDistances[i] / distMax);
		vec3f color(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
		vec2i p0 = ml::math::round(ml::vec2f(key1.pos.x, key1.pos.y));
		vec2i p1 = ml::math::round(ml::vec2f(key2.pos.x + image1.getWidth(), key2.pos.y));
		ImageHelper::drawCircle(matchImage, p0, ml::math::round(key1.scale), color);
		ImageHelper::drawCircle(matchImage, p1, ml::math::round(key2.scale), color);
		ImageHelper::drawLine(matchImage, p0, p1, color);
	}
	//std::cout << "(" << imageIndices << "): max match distance = " << maxMatchDistance << std::endl;
	FreeImageWrapper::saveImage(filename, matchImage);
}

//void SiftVisualization::printMatch(const std::string& filename, const EntryJ& correspondence, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, const mat4f& colorIntrinsics)
//{
//	ColorImageR32G32B32 matchImage;
//	if (!util::fileExists(filename)) {
//		matchImage.allocate(image1.getWidth() * 2, image1.getHeight());
//		ColorImageR32G32B32 im1(image1);
//		ColorImageR32G32B32 im2(image2);
//		matchImage.copyIntoImage(im1, 0, 0);
//		matchImage.copyIntoImage(im2, image1.getWidth(), 0);
//	}
//	else {
//		FreeImageWrapper::loadImage(filename, matchImage);
//	}
//
//	const vec3f color = vec3f(0.0f, 0.0f, 1.0f); // blue
//
//	vec3f camPos0(correspondence.pos_i.x, correspondence.pos_i.y, correspondence.pos_i.z);
//	vec3f camPos1(correspondence.pos_j.x, correspondence.pos_j.y, correspondence.pos_j.z);
//
//	// project to image
//	vec3f projPos0 = colorIntrinsics * camPos0;
//	vec2i p0 = math::round(vec2f(projPos0.x / projPos0.z, projPos0.y / projPos0.z));
//	vec3f projPos1 = colorIntrinsics * camPos1;
//	vec2i p1 = math::round(vec2f(projPos1.x / projPos1.z, projPos1.y / projPos1.z));
//
//	p1 += vec2i(image1.getWidth(), 0);
//
//	const int radius = 3;
//	ImageHelper::drawCircle(matchImage, p0, radius, color);
//	ImageHelper::drawCircle(matchImage, p1, radius, color);
//	ImageHelper::drawLine(matchImage, p0, p1, color);
//	FreeImageWrapper::saveImage(filename, matchImage);
//}
//
//void SiftVisualization::printMatch(const std::string& filename, const vec2ui& imageIndices, const std::vector<EntryJ>& correspondences, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, const mat4f& colorIntrinsics)
//{
//	ColorImageR32G32B32 matchImage;
//	matchImage.allocate(image1.getWidth() * 2, image1.getHeight());
//	ColorImageR32G32B32 im1(image1);
//	ColorImageR32G32B32 im2(image2);
//	matchImage.copyIntoImage(im1, 0, 0);
//	matchImage.copyIntoImage(im2, image1.getWidth(), 0);
//
//	std::vector<vec2i> from, to;
//	for (unsigned int i = 0; i < correspondences.size(); i++) {
//		const EntryJ& corr = correspondences[i];
//		if (corr.isValid() && corr.imgIdx_i == imageIndices.x && corr.imgIdx_j == imageIndices.y) {
//			vec3f proj0 = colorIntrinsics * vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z);
//			vec3f proj1 = colorIntrinsics * vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
//			vec2i p0 = math::round(vec2f(proj0.x / proj0.z, proj0.y / proj0.z));
//			vec2i p1 = math::round(vec2f(proj1.x / proj1.z, proj1.y / proj1.z));
//			from.push_back(p0);			to.push_back(p1);
//		}
//	}
//	if (from.empty()) {
//		std::cout << "no matches to print for images " << imageIndices << std::endl;
//		return;
//	}
//	//draw
//	const int radius = 3;
//	for (unsigned int i = 0; i < from.size(); i++) {
//		const vec3f color = vec3f(RGBColor::randomColor());
//		vec2i p0 = from[i];
//		vec2i p1 = to[i] + vec2i(image1.getWidth(), 0);
//		ImageHelper::drawCircle(matchImage, p0, radius, color);
//		ImageHelper::drawCircle(matchImage, p1, radius, color);
//		ImageHelper::drawLine(matchImage, p0, p1, color);
//	}
//	FreeImageWrapper::saveImage(filename, matchImage);
//}

void SiftVisualization::printMatch(const std::string& filename, const SIFTImageManager* siftManager, const CUDACache* cudaCache, const vec2ui& imageIndices)
{
	const unsigned int widthSIFT = GlobalBundlingState::get().s_widthSIFT;
	const unsigned int heightSIFT = GlobalBundlingState::get().s_heightSIFT;

	//get images
	const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();

	ColorImageR32 xIntensity(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(xIntensity.getData(), cachedFrames[imageIndices.x].d_intensityDownsampled, sizeof(float) * xIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 xImage; convertIntensityToRGB(xIntensity, xImage);
	xImage.resize(widthSIFT, heightSIFT);
	ColorImageR32 yIntensity(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(yIntensity.getData(), cachedFrames[imageIndices.y].d_intensityDownsampled, sizeof(float) * yIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 yImage; convertIntensityToRGB(yIntensity, yImage);
	yImage.resize(widthSIFT, heightSIFT);

	//get matches for these images
	const unsigned int numImages = siftManager->getNumImages();
	std::vector<EntryJ> correspondences(siftManager->getNumGlobalCorrespondences());
	MLIB_ASSERT(!correspondences.empty());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(correspondences.data(), siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyDeviceToHost));

	mat4f colorIntrinsics = cudaCache->getIntrinsics();
	colorIntrinsics._m00 *= (float)widthSIFT / (float)cudaCache->getWidth();
	colorIntrinsics._m11 *= (float)heightSIFT / (float)cudaCache->getHeight();
	colorIntrinsics._m02 *= (float)(widthSIFT -1)/ (float)(cudaCache->getWidth()-1);
	colorIntrinsics._m12 *= (float)(heightSIFT-1) / (float)(cudaCache->getHeight()-1);

	printMatch(filename, xImage, yImage, correspondences, colorIntrinsics, imageIndices);
}

void SiftVisualization::printMatch(const std::string& filename, const CUDACache* cudaCache, const std::vector<SIFTKeyPoint>& keys, const std::vector<unsigned int>& numMatches, 
	const std::vector<uint2>& keyPointIndices, const vec2ui& imageIndices, unsigned int keyIndicesOffset, bool filtered)
{
	if (numMatches[imageIndices.x] == 0) {
		std::cout << "no matches to print for " << imageIndices << std::endl;
		return;
	}

	const unsigned int widthSIFT = GlobalBundlingState::get().s_widthSIFT;
	const unsigned int heightSIFT = GlobalBundlingState::get().s_heightSIFT;

	//get images
	const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();

	ColorImageR32 xIntensity(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(xIntensity.getData(), cachedFrames[imageIndices.x].d_intensityDownsampled, sizeof(float) * xIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 xImage; convertIntensityToRGB(xIntensity, xImage);
	xImage.resize(widthSIFT, heightSIFT);
	ColorImageR32 yIntensity(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(yIntensity.getData(), cachedFrames[imageIndices.y].d_intensityDownsampled, sizeof(float) * yIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 yImage; convertIntensityToRGB(yIntensity, yImage);
	yImage.resize(widthSIFT, heightSIFT);

	//collect matches for these images
	std::vector<std::pair<vec2f, vec2f>> imagePairMatches;
	for (unsigned int m = 0; m < numMatches[imageIndices.x]; m++) {
		const SIFTKeyPoint& k0 = keys[keyPointIndices[keyIndicesOffset + m].x];
		const SIFTKeyPoint& k1 = keys[keyPointIndices[keyIndicesOffset + m].y];
		imagePairMatches.push_back(std::make_pair(vec2f(k0.pos.x, k0.pos.y), vec2f(k1.pos.x, k1.pos.y)));
	}
	printMatch(filename, xImage, yImage, imagePairMatches);
}

void SiftVisualization::printMatch(const std::string& filename, const SensorData& sd, const std::vector<SIFTKeyPoint>& keys, 
	const std::vector<unsigned int>& numMatches, const std::vector<uint2>& keyPointIndices, const vec2ui& imageIndices, 
	unsigned int keyIndicesOffset, bool filtered, unsigned int singleMatchToPrint /*= (unsigned int)-1*/)
{
	if (numMatches[imageIndices.x] == 0) {
		std::cout << "no matches to print for " << imageIndices << std::endl;
		return;
	}

	//get images
	ColorImageR8G8B8 xImage = sd.computeColorImage(imageIndices.x);
	ColorImageR8G8B8 yImage = sd.computeColorImage(imageIndices.y);

	//collect matches for these images
	std::vector<std::pair<vec2f, vec2f>> imagePairMatches;
	if (singleMatchToPrint == (unsigned int)-1) {
		for (unsigned int m = 0; m < numMatches[imageIndices.x]; m++) {
			const SIFTKeyPoint& k0 = keys[keyPointIndices[keyIndicesOffset + m].x];
			const SIFTKeyPoint& k1 = keys[keyPointIndices[keyIndicesOffset + m].y];
			imagePairMatches.push_back(std::make_pair(vec2f(k0.pos.x, k0.pos.y), vec2f(k1.pos.x, k1.pos.y)));
		}
	}
	else {
		const SIFTKeyPoint& k0 = keys[keyPointIndices[keyIndicesOffset + singleMatchToPrint].x];
		const SIFTKeyPoint& k1 = keys[keyPointIndices[keyIndicesOffset + singleMatchToPrint].y];
		imagePairMatches.push_back(std::make_pair(vec2f(k0.pos.x, k0.pos.y), vec2f(k1.pos.x, k1.pos.y)));
	}
	printMatch(filename, xImage, yImage, imagePairMatches);
}

void SiftVisualization::printMatch(const std::string& filename, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2,
	const std::vector<std::pair<vec2f, vec2f>>& imagePairMatches)
{
	if (imagePairMatches.empty()) return; //no matches to print

	ColorImageR32G32B32 matchImage(image1.getWidth() * 2, image1.getHeight());
	{
		ColorImageR32G32B32 im1(image1);
		ColorImageR32G32B32 im2(image2);
		matchImage.copyIntoImage(im1, 0, 0);
		matchImage.copyIntoImage(im2, im1.getWidth(), 0);
	}
	for (const auto& m : imagePairMatches) {
		const vec3f color = vec3f(RGBColor::randomColor());
		const vec2i p0 = math::round(m.first);
		const vec2i p1 = math::round(m.second) + vec2i(image1.getWidth(), 0);
		const int radius = 3;
		ImageHelper::drawCircle(matchImage, p0, radius, color);
		ImageHelper::drawCircle(matchImage, p1, radius, color);
		ImageHelper::drawLine(matchImage, p0, p1, color);
	}
	FreeImageWrapper::saveImage(filename, matchImage);
}

void SiftVisualization::printMatch(const std::string& filename, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, const std::vector<EntryJ>& correspondences, const mat4f& colorIntrinsics, const vec2ui& imageIndices)
{
	std::vector<std::pair<vec2f, vec2f>> imagePairMatches;
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid() && corr.imgIdx_i == imageIndices.x && corr.imgIdx_j == imageIndices.y) {
			vec3f proj0 = colorIntrinsics * vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z);
			vec3f proj1 = colorIntrinsics * vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);
			vec2f p0 = vec2f(proj0.x / proj0.z, proj0.y / proj0.z);
			vec2f p1 = vec2f(proj1.x / proj1.z, proj1.y / proj1.z);
			imagePairMatches.push_back(std::make_pair(p0, p1));
		}
	}
	if (imagePairMatches.empty()) {
		std::cout << "no matches to print for " << imageIndices << std::endl;
		return;
	}
	printMatch(filename, image1, image2, imagePairMatches);
}

void SiftVisualization::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const CUDACache* cudaCache, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;
	unsigned int curFrame = siftManager->getCurrentFrame();
	const unsigned int startFrame = (curFrame + 1 == numFrames) ? 0 : curFrame + 1;

	const std::string dir = util::directoryFromPath(outPath);
	if (!util::directoryExists(dir)) util::makeDirectory(dir);

	const unsigned int widthSIFT = GlobalBundlingState::get().s_widthSIFT;
	const unsigned int heightSIFT = GlobalBundlingState::get().s_heightSIFT;

	// get images
	//TODO get color cpu for these functions
	const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
	ColorImageR32 curIntensity(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curIntensity.getData(), cachedFrames[curFrame].d_intensityDownsampled,
		sizeof(float) * curIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
	ColorImageR8G8B8 curImage; convertIntensityToRGB(curIntensity, curImage);
	curImage.resize(widthSIFT, heightSIFT);

	//print out images
	for (unsigned int prev = startFrame; prev < numFrames; prev++) {
		if (prev == curFrame) continue;

		ColorImageR32 prevIntensity(cudaCache->getWidth(), cudaCache->getHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevIntensity.getData(), cachedFrames[prev].d_intensityDownsampled,
			sizeof(float) * prevIntensity.getNumPixels(), cudaMemcpyDeviceToHost));
		ColorImageR8G8B8 prevImage; convertIntensityToRGB(prevIntensity, prevImage);
		prevImage.resize(widthSIFT, heightSIFT);

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}

void SiftVisualization::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;
	unsigned int curFrame = siftManager->getCurrentFrame();
	const unsigned int startFrame = (curFrame + 1 == numFrames) ? 0 : curFrame + 1;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	const ColorImageR8G8B8& curImage = colorImages[curFrame];

	//print out images
	for (unsigned int prev = startFrame; prev < numFrames; prev++) {
		if (prev == curFrame) continue;
		const ColorImageR8G8B8& prevImage = colorImages[prev];

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}

void SiftVisualization::printMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages, unsigned int frame, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	if (!util::directoryExists(dir)) util::makeDirectory(dir);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	const ColorImageR8G8B8& curImage = colorImages[frame];

	//print out images
	for (unsigned int prev = 0; prev < frame; prev++) {
		const ColorImageR8G8B8& prevImage = colorImages[prev];

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(frame) + ".png", ml::vec2ui(prev, frame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}

void SiftVisualization::saveImPairToPointCloud(const std::string& prefix, const std::vector<CUDACachedFrame>& cachedFrames,
	unsigned int cacheWidth, unsigned int cacheHeight, const vec2ui& imageIndices, const mat4f& transformPrvToCur)
{
	//transforms
	std::vector<mat4f> transforms = { transformPrvToCur, mat4f::identity() };

	//frames
	ColorImageR32G32B32A32 camPosition;
	ColorImageR32 intensity;
	camPosition.allocate(cacheWidth, cacheHeight);
	intensity.allocate(cacheWidth, cacheHeight);

	bool saveFrameByFrame = true;
	const std::string separator = (prefix.back() == '/' || prefix.back() == '\\') ? "" : "_";
	const std::string pre = prefix + separator + std::to_string(imageIndices.x) + "-" + std::to_string(imageIndices.y);

	PointCloudf pc;
	for (unsigned int i = 0; i < 2; i++) {
		mat4f transform = transforms[i];
		unsigned int f = imageIndices[i];
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPosition.getData(), cachedFrames[f].d_cameraposDownsampled, sizeof(float4)*camPosition.getNumPixels(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), cachedFrames[f].d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));

		PointCloudf framePc;

		for (unsigned int i = 0; i < camPosition.getNumPixels(); i++) {
			const vec4f& p = camPosition.getData()[i];
			if (p.x != -std::numeric_limits<float>::infinity()) {
				pc.m_points.push_back(transform * p.getVec3());
				const float c = intensity.getData()[i];
				pc.m_colors.push_back(vec4f(c));

				if (saveFrameByFrame) {
					framePc.m_points.push_back(pc.m_points.back());
					framePc.m_colors.push_back(pc.m_colors.back());
				}
			}
		}
		if (saveFrameByFrame) {
			PointCloudIOf::saveToFile(pre + "_" + std::to_string(f) + ".ply", framePc);
		}
	}
	PointCloudIOf::saveToFile(pre + ".ply", pc);
}

void SiftVisualization::saveImPairToPointCloud(const std::string& prefix, const DepthImage32& depthImage0, const ColorImageR8G8B8& colorImage0,
	const DepthImage32& depthImage1, const ColorImageR8G8B8& colorImage1,
	const mat4f& depthIntrinsicsInv, const vec2ui& imageIndices, const mat4f& transformPrvToCur)
{
	//transforms
	std::vector<mat4f> transforms = { transformPrvToCur, mat4f::identity() };

	bool saveFrameByFrame = true;
	const std::string separator = (prefix.back() == '/' || prefix.back() == '\\') ? "" : "_";
	const std::string pre = prefix + separator + std::to_string(imageIndices.x) + "-" + std::to_string(imageIndices.y);

	const float scaleWidthColor = (float)(colorImage0.getWidth() - 1) / (float)(depthImage0.getWidth() - 1);
	const float scaleHeightColor = (float)(colorImage0.getHeight() - 1) / (float)(depthImage0.getHeight() - 1);
	bool sameColorDepthRes = colorImage0.getDimensions() == depthImage0.getDimensions();

	PointCloudf pc;
	for (unsigned int i = 0; i < 2; i++) {
		mat4f transform = transforms[i];
		unsigned int frameIdx = imageIndices[i];
		PointCloudf framePc;

		const DepthImage32& depthImage = i == 0 ? depthImage0 : depthImage1;
		const ColorImageR8G8B8& colorImage = i == 0 ? colorImage0 : colorImage1;
		for (unsigned int y = 0; y < depthImage.getHeight(); y++) {
			for (unsigned int x = 0; x < depthImage.getWidth(); x++) {
				vec3f p = depthToCamera(depthIntrinsicsInv, depthImage.getData(), depthImage.getWidth(), depthImage.getHeight(), x, y);
				if (p.x != -std::numeric_limits<float>::infinity()) {
					pc.m_points.push_back(transform * p);
					vec4f c;
					if (!sameColorDepthRes) {
						unsigned int cx = (unsigned int)std::round(scaleWidthColor * x);
						unsigned int cy = (unsigned int)std::round(scaleHeightColor * y);
						c = vec4f(vec3f(colorImage(cx, cy)) / 255.0f);
					}
					else {
						c = vec4f(vec3f(colorImage(x, y)) / 255.0f);
					}
					pc.m_colors.push_back(c);

					if (saveFrameByFrame) {
						framePc.m_points.push_back(pc.m_points.back());
						framePc.m_colors.push_back(pc.m_colors.back());
					}
				}
			}// pixels
		}
		if (saveFrameByFrame) {
			PointCloudIOf::saveToFile(pre + "_" + std::to_string(frameIdx) + ".ply", framePc);
		}
	}
	PointCloudIOf::saveToFile(pre + ".ply", pc);
}

ml::vec3f SiftVisualization::depthToCamera(const mat4f& depthIntrinsincsinv, const float* depth, unsigned int width, unsigned int height, unsigned int x, unsigned int y)
{
	float d = depth[y*width + x];
	if (d == -std::numeric_limits<float>::infinity()) return vec3f(-std::numeric_limits<float>::infinity());
	else return depthIntrinsincsinv * (d * vec3f((float)x, (float)y, 1.0f));
}

ml::vec3f SiftVisualization::getNormal(const float* depth, unsigned int width, unsigned int height, const mat4f& depthIntrinsicsInv, unsigned int x, unsigned int y)
{
	vec3f ret(-std::numeric_limits<float>::infinity());
	if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
		vec3f cc = depthToCamera(depthIntrinsicsInv, depth, width, height, x, y);
		vec3f pc = depthToCamera(depthIntrinsicsInv, depth, width, height, x + 1, y + 0);
		vec3f cp = depthToCamera(depthIntrinsicsInv, depth, width, height, x + 0, y + 1);
		vec3f mc = depthToCamera(depthIntrinsicsInv, depth, width, height, x - 1, y + 0);
		vec3f cm = depthToCamera(depthIntrinsicsInv, depth, width, height, x + 0, y - 1);

		if (cc.x != -std::numeric_limits<float>::infinity() && pc.x != -std::numeric_limits<float>::infinity() && cp.x != -std::numeric_limits<float>::infinity() && mc.x != -std::numeric_limits<float>::infinity() && cm.x != -std::numeric_limits<float>::infinity())
		{
			vec3f n = (pc - mc) ^ (cp - cm);
			float l = n.length();
			if (l > 0.0f) {
				ret = n / -l;
			}
		}
	}
	return ret;
}

void SiftVisualization::computePointCloud(PointCloudf& pc, const float* depth, unsigned int depthWidth, unsigned int depthHeight, const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight, const mat4f& depthIntrinsicsInv, const mat4f& transform, float maxDepth)
{
	if (isnan(transform[0]) || transform[0] == -std::numeric_limits<float>::infinity()) {
		std::cerr << "[computePointCloud] ERROR bad transform! skipping..." << std::endl;
		return;
	}
	for (unsigned int y = 0; y < depthHeight; y++) {
		for (unsigned int x = 0; x < depthWidth; x++) {
			vec3f p = depthToCamera(depthIntrinsicsInv, depth, depthWidth, depthHeight, x, y);
			if (p.x != -std::numeric_limits<float>::infinity() && p.z < maxDepth) {

				vec3f n = getNormal(depth, depthWidth, depthHeight, depthIntrinsicsInv, x, y);
				if (n.x != -std::numeric_limits<float>::infinity()) {
					unsigned int cx = (unsigned int)math::round((float)x * (float)(colorWidth - 1) / (float)(depthWidth - 1));
					unsigned int cy = (unsigned int)math::round((float)y * (float)(colorHeight - 1) / (float)(depthHeight - 1));
					vec3f c = vec3f(color[cy * colorWidth + cx]) / 255.0f;
					if (!(c.x == 0 && c.y == 0 && c.z == 0)) {
						pc.m_points.push_back(p);
						pc.m_normals.push_back(n);
						pc.m_colors.push_back(vec4f(c.x, c.y, c.z, 1.0f));
					}
				} // valid normal
			} // valid depth
		} // x
	} // y

	for (auto& p : pc.m_points) {
		p = transform * p;
	}
	mat3f invTranspose = transform.getRotation();
	for (auto& n : pc.m_normals) {
		n = invTranspose * n;
		//n.normalize();
	}
}

void SiftVisualization::computePointCloud(PointCloudf& pc, const ColorImageR8G8B8& color,
	const ColorImageR32G32B32A32& camPos, const ColorImageR32G32B32A32& normal, const mat4f& transform, float maxDepth)
{
	if (isnan(transform[0]) || transform[0] == -std::numeric_limits<float>::infinity()) {
		std::cerr << "[computePointCloud] ERROR bad transform! skipping..." << std::endl;
		return;
	}
	const bool hasNormals = normal.getNumPixels() > 0;
	for (unsigned int y = 0; y < camPos.getHeight(); y++) {
		for (unsigned int x = 0; x < camPos.getWidth(); x++) {
			const vec4f& p = camPos(x, y);
			if (p.x != -std::numeric_limits<float>::infinity() && p.z < maxDepth) {

				vec4f n;
				if (hasNormals) n = normal(x, y);
				if (!hasNormals || n.x != -std::numeric_limits<float>::infinity()) {
					vec3f c;
					if (color.getWidth() != camPos.getWidth()) {
						unsigned int cx = (unsigned int)math::round((float)x * (float)(color.getWidth() - 1) / (float)(camPos.getWidth() - 1));
						unsigned int cy = (unsigned int)math::round((float)y * (float)(color.getHeight() - 1) / (float)(camPos.getHeight() - 1));
						c = vec3f(color(cx, cy)) / 255.0f;
					}
					else {
						c = vec3f(color(x, y)) / 255.0f;
					}
					pc.m_points.push_back(p.getVec3());
					if (hasNormals) pc.m_normals.push_back(n.getVec3());
					pc.m_colors.push_back(vec4f(c.x, c.y, c.z, 1.0f));
				} // valid normal
			} // valid depth
		} // x
	} // y

	for (auto& p : pc.m_points) {
		p = transform * p;
	}
	if (hasNormals) {
		mat3f invTranspose = transform.getRotation();
		for (auto& n : pc.m_normals) {
			n = invTranspose * n;
		}
	}
}

void SiftVisualization::saveToPointCloud(const std::string& filename, const std::vector<DepthImage32>& depthImages, const std::vector<ColorImageR8G8B8>& colorImages,
	const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv, unsigned int skip /*= 1*/, unsigned int numFrames /*= (unsigned int)-1*/, float maxDepth, bool saveFrameByFrame /*= false*/)
{
	std::cout << "#depth = " << depthImages.size() << ", #color = " << colorImages.size() << ", #traj = " << trajectory.size() << std::endl;
	if (numFrames == (unsigned int)-1) numFrames = (unsigned int)depthImages.size();
	MLIB_ASSERT(colorImages.size() >= numFrames && depthImages.size() >= numFrames && trajectory.size() >= numFrames);
	const unsigned int depthWidth = depthImages.front().getWidth();
	const unsigned int depthHeight = depthImages.front().getHeight();
	const unsigned int colorWidth = colorImages.front().getWidth();
	const unsigned int colorHeight = colorImages.front().getHeight();

	std::list<PointCloudf> pcs; std::vector<unsigned int> frameIdxs;
	for (unsigned int i = 0; i < numFrames; i+=skip) {
		if (trajectory[i][0] != -std::numeric_limits<float>::infinity()) {
			pcs.push_back(PointCloudf());
			computePointCloud(pcs.back(), depthImages[i].getData(), depthWidth, depthHeight, colorImages[i].getData(), colorWidth, colorHeight, depthIntrinsicsInv, trajectory[i], maxDepth);
			frameIdxs.push_back(i);
		}
	}

	const std::string prefix = util::directoryFromPath(filename) + "frames/"; unsigned int idx = 0;
	if (saveFrameByFrame) {
		if (!util::directoryExists(prefix)) util::makeDirectory(prefix);
		std::cout << "saving frames to " << prefix << std::endl;
	}
	PointCloudf pc;
	for (const auto& p : pcs) {
		if (saveFrameByFrame) {
			PointCloudIOf::saveToFile(prefix + std::to_string(frameIdxs[idx]) + ".ply", p);
			idx++;
		}
		pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
		pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
		pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
	}
	pc.sparsifyUniform(0.005f, true);
	PointCloudIOf::saveToFile(filename, pc);
}


void SiftVisualization::saveToPointCloud(const std::string& filename, const CUDACache* cache, const std::vector<mat4f>& trajectory, float maxDepth, bool saveFrameByFrame /*= false*/)
{
	const unsigned int numFrames = cache->getNumFrames();
	std::cout << "#cache frames = " << numFrames << "#traj = " << trajectory.size() << std::endl;
	MLIB_ASSERT(numFrames <= trajectory.size());
	const unsigned int width = cache->getWidth();
	const unsigned int height = cache->getHeight();

	const std::vector<CUDACachedFrame>& cachedFrames = cache->getCacheFrames();
	ColorImageR32G32B32A32 camPos(width, height), normals;
	ColorImageR8G8B8 color(width, height);
	ColorImageR32 intensity(width, height);
	std::list<PointCloudf> pcs; std::vector<unsigned int> frameIdxs;
	for (unsigned int i = 0; i < numFrames; i++) {
		if (trajectory[i][0] != -std::numeric_limits<float>::infinity()) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPos.getData(), cachedFrames[i].d_cameraposDownsampled, sizeof(float4)*camPos.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), cachedFrames[i].d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			convertIntensityToRGB(intensity, color);
#ifndef CUDACACHE_UCHAR_NORMALS
			normals.allocate(width, height);
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(normals.getData(), cachedFrames[i].d_normalsDownsampled, sizeof(float4)*normals.getNumPixels(), cudaMemcpyDeviceToHost));
#endif

			pcs.push_back(PointCloudf());
			computePointCloud(pcs.back(), color, camPos, normals, trajectory[i], maxDepth);
			frameIdxs.push_back(i);
		}
	}
	if (saveFrameByFrame) {
		const std::string prefix = util::removeExtensions(filename);
		unsigned int idx = 0;
		for (const auto& p : pcs) {
			PointCloudIOf::saveToFile(prefix + "_" + std::to_string(frameIdxs[idx]) + ".ply", p);
			idx++;
		}
	}
	else {
		//aggregate to single point cloud
		PointCloudf pc;
		for (const auto& p : pcs) {
			pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
			pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
			pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
		}
		PointCloudIOf::saveToFile(filename, pc);
	}
}

void SiftVisualization::saveCamerasToPLY(const std::string& filename, const std::vector<mat4f>& trajectory, bool printDir /*= true*/)
{
	const float radius = 0.05f;
	const vec3f up = vec3f::eY;
	const vec3f look = -vec3f::eZ;

	MeshDataf meshData;
	for (unsigned int i = 0; i < trajectory.size(); i++) {
		const mat4f& t = trajectory[i];
		if (t._m00 != -std::numeric_limits<float>::infinity()) {
			const vec3f color = BaseImageHelper::convertDepthToRGB((float)i, 0.0f, (float)trajectory.size());
			const vec3f eye = t.getTranslation();
			MeshDataf eyeMesh = Shapesf::sphere(radius, eye, 10, 10, vec4f(color, 1.0f)).computeMeshData();
			meshData.merge(eyeMesh);
			if (printDir) {
				MeshDataf upMesh = Shapesf::cylinder(eye, eye + t.getRotation() * up, radius, 10, 10, vec4f(0.0f, 0.0f, 1.0f, 1.0f)).computeMeshData(); // blue for up
				MeshDataf lookMesh = Shapesf::cylinder(eye, eye + t.getRotation() * look, radius, 10, 10, vec4f(1.0f, 0.0f, 1.0f, 0.0f)).computeMeshData(); // red for look
				meshData.merge(upMesh);
				meshData.merge(lookMesh);
			}
		}
	}
	MeshIOf::saveToFile(filename, meshData);
}

void SiftVisualization::saveKeyMatchToPointCloud(const std::string& filename, const EntryJ& corr, const mat4f& transformPrvToCur)
{
	MeshDataf meshData;
	if (util::fileExists(filename)) MeshIOf::loadFromFile(filename, meshData); //add to the current file

	vec3f prvPos = transformPrvToCur * vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z);
	vec3f curPos = vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z);

	vec4f prvColor(0.0f, 1.0f, 0.0f, 1.0f); // green
	vec4f curColor(0.0f, 0.0f, 1.0f, 1.0f); // blue

	const float radius = 0.02f;
	MeshDataf prv = Shapesf::sphere(radius, prvPos, 10, 10, prvColor).computeMeshData();
	MeshDataf cur = Shapesf::sphere(radius, curPos, 10, 10, curColor).computeMeshData();

	meshData.merge(prv);
	meshData.merge(cur);
	MeshIOf::saveToFile(filename, meshData);
}

void SiftVisualization::visualizeImageImageCorrespondences(const std::string& filename, SIFTImageManager* siftManager)
{
	std::vector<EntryJ> correspondences(siftManager->getNumGlobalCorrespondences());
	if (correspondences.empty()) {
		std::cout << "warning: no correspondences in siftmanager to visualize" << std::endl;
		return;
	}
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(correspondences.data(), siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyDeviceToHost));

	visualizeImageImageCorrespondences(filename, correspondences, siftManager->getValidImages(), siftManager->getNumImages());
}

void SiftVisualization::visualizeImageImageCorrespondences(const std::string& filename, const std::vector<EntryJ>& correspondences, const std::vector<int>& valid, unsigned int numImages)
{
	const vec3uc red(255, 0, 0);
	const vec3uc green(0, 255, 0);
	ColorImageR8G8B8 corrImage(numImages, numImages); corrImage.setPixels(vec3uc(0, 0, 0));

	//image-image connections
	unsigned int maxNumCorr = 0;
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid() && corr.imgIdx_i < numImages && corr.imgIdx_j < numImages) {
			corrImage(corr.imgIdx_i, corr.imgIdx_j).x += 1;
			corrImage(corr.imgIdx_j, corr.imgIdx_i).x += 1;

			unsigned int num = std::max((unsigned int)corrImage(corr.imgIdx_i, corr.imgIdx_j).x, (unsigned int)corrImage(corr.imgIdx_j, corr.imgIdx_i).x);
			if (num > maxNumCorr) maxNumCorr = num;
		}
	}
	//normalize
	for (unsigned int y = 0; y < numImages; y++) {
		for (unsigned int x = y + 1; x < numImages; x++) {
			if (corrImage(x, y).x > 0) {
				vec3uc color(0, (unsigned char)(255.0f * (float)corrImage(x, y).x / (float)maxNumCorr), 0); //green
				corrImage(x, y) = color;
				corrImage(y, x) = color;
			}
		}
	}

	//invalid images
	for (unsigned int i = 0; i < numImages; i++) {
		if (valid[i] == 0) {
			for (unsigned int k = 0; k < numImages; k++) {
				corrImage(i, k) = red;
				corrImage(k, i) = red;
			}
		}
	}

	FreeImageWrapper::saveImage(filename, corrImage);
}

template<>
struct std::hash<ml::vec2ui> : public std::unary_function < ml::vec2ui, size_t > {
	size_t operator()(const ml::vec2ui& v) const {
		//TODO larger prime number (64 bit) to match size_t
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		//const size_t p2 = 83492791;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);// ^ ((size_t)v.z * p2);
		return res;
	}
};

void SiftVisualization::getImageImageCorrespondences(const std::vector<EntryJ>& correspondences, unsigned int numImages, std::vector< std::vector<unsigned int> >& imageImageCorrs)
{
	imageImageCorrs.clear();
	imageImageCorrs.resize(numImages);

	std::unordered_set<vec2ui> imageImageCorrSet;
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid())
			imageImageCorrSet.insert(vec2ui(corr.imgIdx_i, corr.imgIdx_j));
	}
	for (const vec2ui& v : imageImageCorrSet) {
		imageImageCorrs[v.x].push_back(v.y);
		imageImageCorrs[v.y].push_back(v.x);
	}
}

void SiftVisualization::printAllMatches(const std::string& outDirectory, SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages, const mat4f& colorIntrinsics)
{
	const unsigned int numImages = siftManager->getNumImages();
	std::vector<EntryJ> correspondences(siftManager->getNumGlobalCorrespondences());
	if (correspondences.empty()) {
		std::cout << "warning: no correspondences in siftmanager to print" << std::endl;
		return;
	}
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(correspondences.data(), siftManager->getGlobalCorrespondencesGPU(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyDeviceToHost));

	printAllMatches(outDirectory, correspondences, numImages, colorImages, colorIntrinsics);
}

void SiftVisualization::printAllMatches(const std::string& outDirectory, const std::vector<EntryJ>& correspondences, unsigned int numImages,
	const std::vector<ColorImageR8G8B8>& colorImages, const mat4f& colorIntrinsics)
{
	std::cout << "printing all matches... ";
	if (!util::directoryExists(outDirectory)) util::makeDirectory(outDirectory);

	std::unordered_map<vec2ui, unsigned int> imageImageCorrSet; //indexes to below
	std::vector<std::vector<EntryJ>> matches;

	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid()) {
			vec2ui imageIndices(corr.imgIdx_i, corr.imgIdx_j);
			auto it = imageImageCorrSet.find(imageIndices);
			if (it == imageImageCorrSet.end()) {
				unsigned int idx = (unsigned int)matches.size();
				imageImageCorrSet[imageIndices] = idx;
				matches.push_back(std::vector<EntryJ>(1, corr));
			}
			else {
				matches[it->second].push_back(corr);
			}
		}
	}

	for (auto& a : imageImageCorrSet) {
		vec2ui imageIndices = a.first;
		const std::vector<EntryJ>& imagePairCorrs = matches[a.second];
		//print matches
		const std::string filename = outDirectory + std::to_string(imageIndices.x) + "-" + std::to_string(imageIndices.y) + ".png";
		printMatch(filename, colorImages[imageIndices.x], colorImages[imageIndices.y], imagePairCorrs, colorIntrinsics, imageIndices);
	}
	std::cout << "done!" << std::endl;
}

void SiftVisualization::saveKeyMatchToPointCloud(const std::string& prefix, const vec2ui& imageIndices, const std::vector<EntryJ>& correspondences,
	const DepthImage32& depthImage0, const ColorImageR8G8B8& colorImage0,
	const DepthImage32& depthImage1, const ColorImageR8G8B8& colorImage1, const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv)
{
	PointCloudf pc0, pc1;
	computePointCloud(pc0, depthImage0.getData(), depthImage0.getWidth(), depthImage0.getHeight(),
		colorImage0.getData(), colorImage0.getWidth(), colorImage0.getHeight(), depthIntrinsicsInv,
		trajectory[imageIndices.x], 3.5f);
	computePointCloud(pc1, depthImage1.getData(), depthImage1.getWidth(), depthImage1.getHeight(),
		colorImage1.getData(), colorImage1.getWidth(), colorImage1.getHeight(), depthIntrinsicsInv,
		trajectory[imageIndices.y], 3.5f);
	PointCloudIOf::saveToFile(prefix + "-" + std::to_string(imageIndices.x) + ".ply", pc0);
	PointCloudIOf::saveToFile(prefix + "-" + std::to_string(imageIndices.y) + ".ply", pc1);

	std::vector<vec3f> keys0, keys1; std::vector<vec3f> res;
	for (unsigned int i = 0; i < correspondences.size(); i++) {
		const EntryJ& corr = correspondences[i];
		if (corr.isValid() && corr.imgIdx_i == imageIndices.x && corr.imgIdx_j == imageIndices.y) {
			keys0.push_back(vec3f(corr.pos_i.x, corr.pos_i.y, corr.pos_i.z));
			keys1.push_back(vec3f(corr.pos_j.x, corr.pos_j.y, corr.pos_j.z));
			res.push_back(trajectory[imageIndices.x] * keys0.back() - trajectory[imageIndices.y] * keys1.back());
		}
	}
	if (keys0.empty()) { std::cout << "no matches to print for images " << imageIndices << std::endl; return; }
	//draw
	const float radius = 0.01f;
	MeshDataf keysMesh0, keysMesh1;
	for (unsigned int i = 0; i < keys0.size(); i++) {
		const vec3f color = BaseImageHelper::convertDepthToRGB(res[i].length(), 0.0f, 0.2f);//vec3f(RGBColor::randomColor());
		keysMesh0.merge(Shapesf::sphere(radius, trajectory[imageIndices.x] * keys0[i], 10, 10, vec4f(color)).computeMeshData());
		keysMesh1.merge(Shapesf::sphere(radius, trajectory[imageIndices.y] * keys1[i], 10, 10, vec4f(color)).computeMeshData());
	}
	std::ofstream s(prefix + "-residuals.txt");
	for (unsigned int i = 0; i < res.size(); i++) s << res[i] << std::endl;
	s.close();
	MeshIOf::saveToFile(prefix + "-keys_" + std::to_string(imageIndices.x) + ".ply", keysMesh0);
	MeshIOf::saveToFile(prefix + "-keys_" + std::to_string(imageIndices.y) + ".ply", keysMesh1);
}

void SiftVisualization::saveFrameToPointCloud(const std::string& filename, const DepthImage32& depth, const ColorImageR8G8B8& color, const mat4f& transform, const mat4f& depthIntrinsicsInverse, float maxDepth /*= 3.5f*/)
{
	PointCloudf pc;
	computePointCloud(pc, depth.getData(), depth.getWidth(), depth.getHeight(), color.getData(), color.getWidth(), color.getHeight(), depthIntrinsicsInverse, mat4f::identity(), maxDepth);
	PointCloudIOf::saveToFile(filename, pc);
}

