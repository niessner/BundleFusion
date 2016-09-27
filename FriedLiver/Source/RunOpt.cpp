
#include "stdafx.h"
#include "RunOpt.h"
#include "SIFTImageManager.h"
#include "SBA.h"
#include "SiftVisualization.h"

void RunOpt::run(const std::string& sensFile, const std::string& siftFile, const std::string& trajectoryFile)
{
	//load frames from sens file
	std::vector<DepthImage32> depthImages;
	std::vector<ColorImageR8G8B8> colorImages;
	mat4f depthIntrinsics, colorIntrinsics;
	loadFromSensFile(sensFile, GlobalBundlingState::get().s_submapSize, depthImages, colorImages, depthIntrinsics, colorIntrinsics);
	mat4f depthIntrinsicsInverse = depthIntrinsics.getInverse();

	//load sparse features
	std::cout << "loading sift manager from file... ";
	SIFTImageManager manager(GlobalBundlingState::get().s_submapSize, GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumKeysPerImage);
	manager.loadFromFile(siftFile);
	std::cout << "done!" << std::endl;
	const unsigned int numKeys = manager.getNumImages();
	if (numKeys < depthImages.size()) {
		depthImages.resize(numKeys);
		colorImages.resize(numKeys);
	}
	MLIB_ASSERT(numKeys == depthImages.size());

	//load initial trajectory
	std::cout << "loading trajectory from file... ";
	std::vector<mat4f> trajectory;
	BinaryDataStreamFile s(trajectoryFile, false);
	s >> trajectory; s.closeStream();
	std::cout << "done!" << std::endl;
	MLIB_ASSERT(trajectory.size() == numKeys);

	//save out point cloud of initial
	SiftVisualization::saveToPointCloud("init.ply", depthImages, colorImages, trajectory, depthIntrinsicsInverse);

	//optimization
	const unsigned int maxNumIters = 5; const unsigned int numPCGIts = 150;
	const bool useVerify = false;
	SBA sba;
	const unsigned int maxNumImages = GlobalBundlingState::get().s_maxNumImages;
	const unsigned int maxNumResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (maxNumImages*(maxNumImages - 1)) / 2;
	sba.init(numKeys, maxNumResiduals);

	//gpu transforms
	float4x4* d_transforms = NULL; MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_transforms, sizeof(float4x4)*numKeys));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_transforms, trajectory.data(), sizeof(float4x4)*numKeys, cudaMemcpyHostToDevice));
	
	//set weights to sparse only solve
	sba.setGlobalWeights(std::vector<float>(maxNumIters, 1.0f), std::vector<float>(maxNumIters, 0.0f), std::vector<float>(maxNumIters, 0.0f), false); 
	//optimize!
	sba.align(&manager, NULL, d_transforms, maxNumIters, numPCGIts, useVerify, false, false, true, true, false);
	
	//copy optimized poses to cpu
	std::vector<mat4f> optTrajectory(numKeys);
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(optTrajectory.data(), d_transforms, sizeof(float4x4)*numKeys, cudaMemcpyDeviceToHost));
	MLIB_CUDA_SAFE_FREE(d_transforms); //free gpu transforms

	//save out point cloud of optimized
	SiftVisualization::saveToPointCloud("opt.ply", depthImages, colorImages, optTrajectory, depthIntrinsicsInverse);
}
