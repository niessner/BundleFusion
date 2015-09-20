
#include "SiftGPU.h"
#include "CuTexImage.h"

#ifndef GPU_MATCH_H
#define GPU_MATCH_H


class CUDATimer;

///matcher export
//This is a gpu-based sift match implementation. 
class SiftMatchGPU
{
public:

	//Consructor, the argument specifies the maximum number of features to match
	SiftMatchGPU(int max_sift = 4096);
	//desctructor
	 ~SiftMatchGPU();

	void InitSiftMatch();

	//Specifiy descriptors to match, index = [0/1] for two features sets respectively
	//Option1, use float descriptors, and they be already normalized to 1.0
	 void SetDescriptorsFromCPU(int index, int num, const float* descriptors, int id  = -1);
	//Option 2 unsigned char descriptors. They must be already normalized to 512
	 void SetDescriptorsFromCPU(int index, int num, const unsigned char * descriptors, int id = -1);
	//unsigned char descriptors. They must be already normalized to 512
	 void SetDescriptors(int index, int num, unsigned char* d_descriptors, int id = -1);

	//match two sets of features, the function RETURNS the number of matches.
	//Given two normalized descriptor d1,d2, the distance here is acos(d1 *d2);
	// int  GetSiftMatch(
	//			int max_match,	// the length of the match_buffer.
	//			int match_buffer[][2], //buffer to receive the matched feature indices
	//			float* matchDistances, // buffer to receive match distances
	//			float distmax = 0.7,	//maximum distance of sift descriptor
	//			float ratiomax = 0.8,	//maximum distance ratio
	//			int mutual_best_match = 1); //mutual best match or one way
	 void  GetSiftMatch(
		int max_match,	// the length of the match_buffer.
		ImagePairMatch& imagePairMatch,
		uint2 keyPointOffset,
		float distmax = 0.7f,	//maximum distance of sift descriptor
		float ratiomax = 0.8f,	//maximum distance ratio
		int mutual_best_match = 1); //mutual best match or one way

	void EvaluateTimings();

	//two functions for guded matching, two constraints can be used 
	//one homography and one fundamental matrix, the use is as follows
	//1. for each image, first call SetDescriptor then call SetFeatureLocation
	//2. Call GetGuidedSiftMatch
	//input feature location is a vector of [float x, float y, float skip[gap]]
	 void SetFeautreLocation(int index, const float* locations, int gap = 0);
	inline void SetFeatureLocation(int index, const SiftGPU::SiftKeypoint * keys)
	{
		SetFeautreLocation(index, (const float*) keys, 2);
	}

	//static int  CheckCudaDevice(int device);

	//overload the new operator, the same reason as SiftGPU above
	//void* operator new (size_t size);
private:
	void  GetBestMatch(int max_match, ImagePairMatch& imagePairMatch, float distmax, float ratiomax, uint2 keyPointOffset);//, int mbm);

	//tex storage
	CuTexImage _texLoc[2];
	CuTexImage _texDes[2];
	CuTexImage _texDot;
	CuTexImage _texMatch[1];	//at some point, we had a col and a row result; but since the col kernel directly outputs the full feature list, we only need a row result as an intermediary
	CuTexImage _texCRT;

	// hack to store match distances
	float* d_rowMatchDistances;

	//programs
	//
	int _max_sift;
	int _num_sift[2];
	int _id_sift[2];
	int _have_loc[2];

	//gpu parameter
	int _initialized;
	std::vector<int> sift_buffer;

	CUDATimer* _timer;
};


#endif  //GPU_MATCH_H


