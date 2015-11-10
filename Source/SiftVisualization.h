#pragma once

class SIFTImageManager;
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

private:
};