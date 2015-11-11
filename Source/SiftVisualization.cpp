#include "stdafx.h"
#include "SiftVisualization.h"
#include "ImageHelper.h"
#include "SIFTImageManager.h"
#include "CUDACache.h"
#include "GlobalBundlingState.h"
#include "CUDAImageManager.h"


void SiftVisualization::printKey(const std::string& filename, CUDAImageManager* cudaImageManager, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame)
{
	//TODO get color cpu for these functions
	CUDAImageManager::ManagedRGBDInputFrame& integrateFrame = cudaImageManager->getIntegrateFrame(allFrame);

	ColorImageR8G8B8A8 im(cudaImageManager->getIntegrationWidth(), cudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(im.getPointer(), integrateFrame.getColorFrameGPU(), sizeof(uchar4) * cudaImageManager->getIntegrationWidth() * cudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
	im.reSample(GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT);

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];
		RGBColor c = RGBColor::randomColor();
		vec4uc color(c.r, c.g, c.b, c.a);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}


void SiftVisualization::printKey(const std::string& filename, const ColorImageR8G8B8A8& image, const SIFTImageManager* siftManager, unsigned int frame)
{
	ColorImageR8G8B8A8 im = image;

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];
		RGBColor c = RGBColor::randomColor();
		vec4uc color(c.r, c.g, c.b, c.a);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}

void SiftVisualization::printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
	const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered, int maxNumMatches)
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

void SiftVisualization::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const CUDACache* cudaCache, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	const unsigned int widthSIFT = GlobalBundlingState::get().s_widthSIFT;
	const unsigned int heightSIFT = GlobalBundlingState::get().s_heightSIFT;

	// get images
	unsigned int curFrame = numFrames - 1; //TODO get color cpu for these functions
	const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
	ColorImageR8G8B8A8 curImage(cudaCache->getWidth(), cudaCache->getHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curImage.getPointer(), cachedFrames[curFrame].d_colorDownsampled,
		sizeof(uchar4) * curImage.getNumPixels(), cudaMemcpyDeviceToHost));
	curImage.reSample(widthSIFT, heightSIFT);

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		ColorImageR8G8B8A8 prevImage(cudaCache->getWidth(), cudaCache->getHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevImage.getPointer(), cachedFrames[prev].d_colorDownsampled,
			sizeof(uchar4) * prevImage.getNumPixels(), cudaMemcpyDeviceToHost));
		prevImage.reSample(widthSIFT, heightSIFT);

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}

void SiftVisualization::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8A8>& colorImages, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	unsigned int curFrame = numFrames - 1; 
	const ColorImageR8G8B8A8& curImage = colorImages[curFrame];

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		const ColorImageR8G8B8A8& prevImage = colorImages[prev];

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}

void SiftVisualization::printMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8A8>& colorImages, unsigned int frame, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	if (!util::directoryExists(dir)) util::makeDirectory(dir);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	const ColorImageR8G8B8A8& curImage = colorImages[frame];

	//print out images
	for (unsigned int prev = 0; prev < frame; prev++) {
		const ColorImageR8G8B8A8& prevImage = colorImages[prev];

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(frame) + ".png", ml::vec2ui(prev, frame),
			prevImage, curImage, 0.7f, filtered, maxNumMatches);
	}
}
