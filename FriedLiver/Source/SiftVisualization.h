#pragma once

#include "CUDACacheUtil.h"
#include "SIFTImageManager.h"

class CUDACache;
class CUDAImageManager;

class SiftVisualization {
public:
	static void printKey(const std::string& filename, const CUDACache* cudaCache, const SIFTImageManager* siftManager, unsigned int frame);
	static void printKey(const std::string& filename, const ColorImageR8G8B8& image, const SIFTImageManager* siftManager, unsigned int frame);
	static void printKey(const std::string& filename, CUDAImageManager* cudaImageManager, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame);

	static void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const CUDACache* cudaCache,
		bool filtered, int maxNumMatches = -1);
	static void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages,
		bool filtered, int maxNumMatches = -1);

	static void printMatches(const std::string& outPath, const SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages,
		unsigned int frame, bool filtered, int maxNumMatches = -1);

	//print from global corrs
	static void printMatch(const std::string& filename, const SIFTImageManager* siftManager, const CUDACache* cudaCache, const vec2ui& imageIndices);
	//print from cur match set
	static void printMatch(const std::string& filename, const CUDACache* cudaCache, const std::vector<SIFTKeyPoint>& keys,
		const std::vector<unsigned int>& numMatches, const std::vector<uint2>& keyPointIndices, const vec2ui& imageIndices, 
		unsigned int keyIndicesOffset, bool filtered);
	static void printMatch(const std::string& filename, const SensorData& sd, const std::vector<SIFTKeyPoint>& keys,
		const std::vector<unsigned int>& numMatches, const std::vector<uint2>& keyPointIndices, const vec2ui& imageIndices,
		unsigned int keyIndicesOffset, bool filtered, unsigned int singleMatchToPrint = (unsigned int)-1);

	static void printMatch(const SIFTImageManager* siftManager, const std::string& filename, const vec2ui& imageIndices,
		const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, float distMax, bool filtered, int maxNumMatches);

	//static void printMatch(const std::string& filename, const EntryJ& correspondence,
	//	const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, const mat4f& colorIntrinsics);
	//static void printMatch(const std::string& filename, const vec2ui& imageIndices, const std::vector<EntryJ>& correspondences,
	//	const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2, const mat4f& colorIntrinsics);


	static void saveImPairToPointCloud(const std::string& prefix, const DepthImage32& depthImage0, const ColorImageR8G8B8& colorImage0,
		const DepthImage32& depthImage1, const ColorImageR8G8B8& colorImage1,
		const mat4f& depthIntrinsicsInv, const vec2ui& imageIndices, const mat4f& transformPrvToCur);
	static void saveImPairToPointCloud(const std::string& prefix, const std::vector<CUDACachedFrame>& cachedFrames, unsigned int cacheWidth, unsigned int cacheHeight,
		const vec2ui& imageIndices, const mat4f& transformPrvToCur);
	static void saveKeyMatchToPointCloud(const std::string& filename, const EntryJ& corr, const mat4f& transformPrvToCur);
	static void saveKeyMatchToPointCloud(const std::string& prefix, const vec2ui& imageIndices, const std::vector<EntryJ>& correspondences,
		const DepthImage32& depthImage0, const ColorImageR8G8B8& colorImage0,
		const DepthImage32& depthImage1, const ColorImageR8G8B8& colorImage1, const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv);


	static vec3f depthToCamera(const mat4f& depthIntrinsincsinv, const float* depth, unsigned int width, unsigned int height, unsigned int x, unsigned int y);
	static vec3f getNormal(const float* depth, unsigned int width, unsigned int height, const mat4f& depthIntrinsicsInv,
		unsigned int x, unsigned int y);
	static void computePointCloud(PointCloudf& pc, const float* depth, unsigned int depthWidth, unsigned int depthHeight,
		const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight,
		const mat4f& depthIntrinsicsInv, const mat4f& transform, float maxDepth);

	static void saveToPointCloud(const std::string& filename, const std::vector<DepthImage32>& depthImages, const std::vector<ColorImageR8G8B8>& colorImages,
		const std::vector<mat4f>& trajectory, const mat4f& depthIntrinsicsInv, unsigned int skip = 1, unsigned int numFrames = (unsigned int)-1, float maxDepth = 3.5f, bool saveFrameByFrame = false);

	static void saveToPointCloud(const std::string& filename, const CUDACache* cache, const std::vector<mat4f>& trajectory, float maxDepth = 3.5f, bool saveFrameByFrame = false);

	static void saveFrameToPointCloud(const std::string& filename, const DepthImage32& depth, const ColorImageR8G8B8& color, const mat4f& transform, const mat4f& depthIntrinsicsInverse, float maxDepth = 3.5f);

	static void computePointCloud(PointCloudf& pc, const ColorImageR8G8B8& color,
		const ColorImageR32G32B32A32& camPos, const ColorImageR32G32B32A32& normal,
		const mat4f& transform, float maxDepth);

	static void saveCamerasToPLY(const std::string& filename, const std::vector<mat4f>& trajectory, bool printDir = true);

	static void visualizeImageImageCorrespondences(const std::string& filename, SIFTImageManager* siftManager);
	static void visualizeImageImageCorrespondences(const std::string& filename, const std::vector<EntryJ>& correspondences, const std::vector<int>& valid, unsigned int numImages);

	static void getImageImageCorrespondences(const std::vector<EntryJ>& correspondences, unsigned int numImages, std::vector< std::vector<unsigned int> >& imageImageCorrs);

	static void printAllMatches(const std::string& outDirectory, SIFTImageManager* siftManager, const std::vector<ColorImageR8G8B8>& colorImages, const mat4f& colorIntrinsics);
	static void printAllMatches(const std::string& outDirectory, const std::vector<EntryJ>& correspondences, unsigned int numImages,
		const std::vector<ColorImageR8G8B8>& colorImages, const mat4f& colorIntrinsics);

	static void printMatch(const std::string& filename, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2,
		const std::vector<EntryJ>& correspondences, const mat4f& colorIntrinsics, const vec2ui& imageIndices);
	static void printMatch(const std::string& filename, const ColorImageR8G8B8& image1, const ColorImageR8G8B8& image2,
		const std::vector<std::pair<vec2f, vec2f>>& imagePairMatches);
private:

	static void convertIntensityToRGB(const ColorImageR32& intensity, ColorImageR8G8B8& image) {
		image.allocateSameSize(intensity);
		for (unsigned int i = 0; i < intensity.getNumPixels(); i++) {
			image.getData()[i] = vec3uc((unsigned char)std::round(255.0f * intensity.getData()[i]));
		}
	}
};