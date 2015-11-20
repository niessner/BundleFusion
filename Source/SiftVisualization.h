#pragma once

#include "CUDACacheUtil.h"
#include "SIFTImageManager.h"

class CUDACache;
class CUDAImageManager;

class SiftVisualization {
public:
	static void printKey(const std::string& filename, const ColorImageR8G8B8A8& image, const SIFTImageManager* siftManager, unsigned int frame);
	static void printKey(const std::string& filename, CUDAImageManager* cudaImageManager, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame);

	static void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const CUDACache* cudaCache,
		bool filtered, int maxNumMatches = -1);
	static void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8A8>& colorImages,
		bool filtered, int maxNumMatches = -1);

	static void printMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8A8>& colorImages,
		unsigned int frame, bool filtered, int maxNumMatches = -1);


	static void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, float distMax, bool filtered, int maxNumMatches);

	static void printMatch(const std::string& filename, const EntryJ& correspondence,
		const ColorImageR8G8B8A8& image1, const ColorImageR8G8B8A8& image2, const mat4f& colorIntrinsics);


	static void saveImPairToPointCloud(const std::string& prefix, const std::vector<CUDACachedFrame>& cachedFrames, unsigned int cacheWidth, unsigned int cacheHeight,
		const vec2ui& imageIndices, const mat4f& transformCurToPrv);


	static vec3f depthToCamera(const mat4f& depthIntrinsincsinv, const float* depth, unsigned int width, unsigned int height, unsigned int x, unsigned int y);
	static vec3f getNormal(const float* depth, unsigned int width, unsigned int height, const mat4f& depthIntrinsicsInv,
		unsigned int x, unsigned int y);
	static void computePointCloud(PointCloudf& pc, const float* depth, unsigned int depthWidth, unsigned int depthHeight,
		const vec4uc* color, unsigned int colorWidth, unsigned int colorHeight,
		const mat4f& depthIntrinsicsInv, const mat4f& transform);

	static void saveToPointCloud(const std::string& filename, const std::vector<DepthImage32>& depthImages, const std::vector<ColorImageR8G8B8A8>& colorImages,
		const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv);
private:
};