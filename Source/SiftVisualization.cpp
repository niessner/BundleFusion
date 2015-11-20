#include "stdafx.h"
#include "SiftVisualization.h"
#include "ImageHelper.h"
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

void SiftVisualization::printMatch(const std::string& filename, const EntryJ& correspondence, const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, const mat4f& colorIntrinsics)
{
	ColorImageR32G32B32 matchImage;
	if (!util::fileExists(filename)) {
		matchImage.allocate(image1.getWidth() * 2, image1.getHeight());
		ColorImageR32G32B32 im1(image1);
		ColorImageR32G32B32 im2(image2);
		matchImage.copyIntoImage(im1, 0, 0);
		matchImage.copyIntoImage(im2, image1.getWidth(), 0);
	}
	else {
		FreeImageWrapper::loadImage(filename, matchImage);
	}

	const vec3f color = vec3f(0.0f, 0.0f, 1.0f); // blue

	vec3f camPos0(correspondence.pos_i.x, correspondence.pos_i.y, correspondence.pos_i.z);
	vec3f camPos1(correspondence.pos_j.x, correspondence.pos_j.y, correspondence.pos_j.z);

	// project to image
	vec3f projPos0 = colorIntrinsics * camPos0;
	vec2i p0 = math::round(vec2f(projPos0.x / projPos0.z, projPos0.y / projPos0.z));
	vec3f projPos1 = colorIntrinsics * camPos1;
	vec2i p1 = math::round(vec2f(projPos1.x / projPos1.z, projPos1.y / projPos1.z));

	p1 += vec2i(image1.getWidth(), 0);

	const int radius = 3;
	ImageHelper::drawCircle(matchImage, p0, radius, color);
	ImageHelper::drawCircle(matchImage, p1, radius, color);
	ImageHelper::drawLine(matchImage, p0, p1, color);
	FreeImageWrapper::saveImage(filename, matchImage);
}

void SiftVisualization::printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const CUDACache* cudaCache, bool filtered, int maxNumMatches /*= -1*/)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	if (!util::directoryExists(dir)) util::makeDirectory(dir);

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

void SiftVisualization::saveImPairToPointCloud(const std::string& prefix, const std::vector<CUDACachedFrame>& cachedFrames,
	unsigned int cacheWidth, unsigned int cacheHeight, const vec2ui& imageIndices, const mat4f& transformPrvToCur)
{
	//transforms
	std::vector<mat4f> transforms = { transformPrvToCur, mat4f::identity() };

	//frames
	ColorImageR32G32B32A32 camPosition;
	ColorImageR8G8B8A8 color;
	camPosition.allocate(cacheWidth, cacheHeight);
	color.allocate(cacheWidth, cacheHeight);

	bool saveFrameByFrame = true;
	const std::string dir = util::directoryFromPath(prefix);
	if (saveFrameByFrame && !util::directoryExists(dir)) util::makeDirectory(dir);

	PointCloudf pc;
	for (unsigned int i = 0; i < 2; i++) {
		mat4f transform = transforms[i];
		unsigned int f = imageIndices[i];
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPosition.getPointer(), cachedFrames[f].d_cameraposDownsampled, sizeof(float4)*camPosition.getNumPixels(), cudaMemcpyDeviceToHost));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(color.getPointer(), cachedFrames[f].d_colorDownsampled, sizeof(uchar4)*color.getNumPixels(), cudaMemcpyDeviceToHost));

		//DepthImage32 dImage(cacheWidth, cacheHeight);
		//MLIB_CUDA_SAFE_CALL(cudaMemcpy(dImage.getPointer(), cachedFrames[f].d_depthDownsampled, sizeof(float)*dImage.getNumPixels(), cudaMemcpyDeviceToHost));
		//FreeImageWrapper::saveImage("debug/testDepth.png", ColorImageR32G32B32(dImage));
		//FreeImageWrapper::saveImage("debug/test.png", camPosition);
		//{
		//	PointCloudf wtf;
		//	for (unsigned int i = 0; i < dImage.getNumPixels(); i++) {
		//		float d = dImage.getPointer()[i];
		//		if (d != -std::numeric_limits<float>::infinity()) {
		//			//vec3f p = 
		//		}
		//	}
		//}

		PointCloudf framePc;

		for (unsigned int i = 0; i < camPosition.getNumPixels(); i++) {
			const vec4f& p = camPosition.getPointer()[i];
			if (p.x != -std::numeric_limits<float>::infinity()) {
				pc.m_points.push_back(transform * p.getVec3());
				const vec4uc& c = color.getPointer()[i];
				pc.m_colors.push_back(vec4f(c.x, c.y, c.z, c.w) / 255.f);

				if (saveFrameByFrame) {
					framePc.m_points.push_back(pc.m_points.back());
					framePc.m_colors.push_back(pc.m_colors.back());
				}
			}
		}
		if (saveFrameByFrame) {
			PointCloudIOf::saveToFile(dir + std::to_string(f) + ".ply", framePc);
		}
	}
	PointCloudIOf::saveToFile(prefix + "_" + std::to_string(imageIndices.x) + "-" + std::to_string(imageIndices.y) + ".ply", pc);
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

void SiftVisualization::computePointCloud(PointCloudf& pc, const float* depth, unsigned int depthWidth, unsigned int depthHeight, const vec4uc* color, unsigned int colorWidth, unsigned int colorHeight, const mat4f& depthIntrinsicsInv, const mat4f& transform)
{
	for (unsigned int y = 0; y < depthHeight; y++) {
		for (unsigned int x = 0; x < depthWidth; x++) {
			vec3f p = depthToCamera(depthIntrinsicsInv, depth, depthWidth, depthHeight, x, y);
			if (p.x != -std::numeric_limits<float>::infinity()) {

				vec3f n = getNormal(depth, depthWidth, depthHeight, depthIntrinsicsInv, x, y);
				if (n.x != -std::numeric_limits<float>::infinity()) {
					unsigned int cx = (unsigned int)math::round((float)x * (float)colorWidth / (float)depthWidth);
					unsigned int cy = (unsigned int)math::round((float)y * (float)colorHeight / (float)depthHeight);
					vec3f c = vec3f(color[cy * colorWidth + cx].getVec3()) / 255.0f;
					if (!(c.x == 0 && c.y == 0 && c.z == 0)) {
						pc.m_points.push_back(vec3f(p));
						pc.m_normals.push_back(vec3f(n));
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

void SiftVisualization::saveToPointCloud(const std::string& filename, const std::vector<DepthImage32>& depthImages, const std::vector<ColorImageR8G8B8A8>& colorImages, const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv)
{
	MLIB_ASSERT(depthImages.size() > 0 && depthImages.size() == colorImages.size() && depthImages.size() == trajectory.size());
	const unsigned int depthWidth = depthImages.front().getWidth();
	const unsigned int depthHeight = depthImages.front().getHeight();
	const unsigned int colorWidth = colorImages.front().getWidth();
	const unsigned int colorHeight = colorImages.front().getHeight();

	std::list<PointCloudf> pcs;
	for (unsigned int i = 0; i < depthImages.size(); i++) {
		pcs.push_back(PointCloudf());
		computePointCloud(pcs.back(), depthImages[i].getPointer(), depthWidth, depthHeight, colorImages[i].getPointer(), colorWidth, colorHeight, depthIntrinsicsInv, trajectory[i]);
	}
	PointCloudf pc;
	for (const auto& p : pcs) {
		pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
		pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
		pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
	}
	PointCloudIOf::saveToFile(filename, pc);
}
