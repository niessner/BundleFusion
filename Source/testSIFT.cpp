#pragma once

#include "stdafx.h"
#include "testSIFT.h"

#include "SIFTImageManager.h"
#include "SiftGPU/SiftCameraParams.h"
#include "SiftGPU/SiftMatch.h"
#include "SiftGPU/SiftMatchFilter.h"
#include "SiftGPU/MatrixConversion.h"


extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params);

void TestSIFT::test()
{
	MLIB_ASSERT(!m_intensityImages.empty() && !m_depthImages.empty());
	MLIB_ASSERT(m_targetNumKeypoints > 0 && m_minKeyScale >= 0);
	const unsigned int siftWidth = m_intensityImages.front().getWidth();
	const unsigned int siftHeight = m_intensityImages.front().getHeight();
	const unsigned int depthWidth = m_depthImages.front().getWidth();
	const unsigned int depthHeight = m_depthImages.front().getHeight();

	std::cout << "========================================================" << std::endl;
	std::cout << "sift size [ " << siftWidth << " x " << siftHeight << " ]" << std::endl;
	std::cout << "target num keys [ " << m_targetNumKeypoints << " ]" << std::endl;
	std::cout << "min key scale [ " << m_minKeyScale << " ]" << std::endl;

	//init sift
	SiftCameraParams siftCameraParams;
	siftCameraParams.m_depthWidth = depthWidth;
	siftCameraParams.m_depthHeight = depthHeight;
	siftCameraParams.m_intensityWidth = siftWidth;
	siftCameraParams.m_intensityHeight = siftHeight;
	siftCameraParams.m_siftIntrinsics = MatrixConversion::toCUDA(m_intensityIntrinsics);
	siftCameraParams.m_siftIntrinsicsInv = MatrixConversion::toCUDA(m_intensityIntrinsics.getInverse());
	siftCameraParams.m_minKeyScale = m_minKeyScale;
	updateConstantSiftCameraParams(siftCameraParams);

	const unsigned int maxKeysPerImage = 2048;
	SIFTImageManager siftManager(10, 2500, maxKeysPerImage);
	Timer t;
	double detectTime = 0.0, matchTime = 0.0;
	{ //detection
		SiftGPU* sift = new SiftGPU;
		sift->SetParams(siftWidth, siftHeight, false, m_targetNumKeypoints, 0.5f, 5.0f); //sensor min/max
		sift->InitSiftGPU();

		std::cout << "running sift detection... ";
		const unsigned int numFrames = (unsigned int)m_intensityImages.size();
		float *d_intensity = NULL, *d_depth = NULL;
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensity, sizeof(float)*siftWidth*siftHeight));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depth, sizeof(float)*depthWidth*depthHeight));
		for (unsigned int f = 0; f < numFrames; f++) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_intensity, m_intensityImages[f].getData(), sizeof(float)*m_intensityImages[f].getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_depth, m_depthImages[f].getData(), sizeof(float)*m_depthImages[f].getNumPixels(), cudaMemcpyHostToDevice));
			// run sift
			cudaDeviceSynchronize(); t.start();
			SIFTImageGPU& cur = siftManager.createSIFTImageGPU();
			int success = sift->RunSIFT(d_intensity, d_depth);
			if (!success) throw MLIB_EXCEPTION("Error running SIFT detection");
			unsigned int numKeypoints = sift->GetKeyPointsAndDescriptorsCUDA(cur, d_depth);
			if (numKeypoints > maxKeysPerImage) throw MLIB_EXCEPTION("too many keypoints (" + std::to_string(numKeypoints) + ")");
			siftManager.finalizeSIFTImageGPU(numKeypoints);
			cudaDeviceSynchronize(); t.stop();
			detectTime += t.getElapsedTimeMS();
		}
		std::cout << "done!" << std::endl;
		std::cout << "total time detect: " << detectTime << " ms (" << numFrames << " frames)" << std::endl;
		std::cout << "mean time detect:  " << (detectTime / numFrames) << " ms" << std::endl;
		float totNumKeys = 0.0f;
		for (unsigned int f = 0; f < siftManager.getNumImages(); f++)
			totNumKeys += siftManager.getNumKeyPointsPerImage(f);
		std::cout << "mean num keys:  " << (totNumKeys / numFrames) << std::endl;
		std::cout << std::endl;
		MLIB_CUDA_SAFE_FREE(d_intensity);
		MLIB_CUDA_SAFE_FREE(d_depth);
		SAFE_DELETE(sift);
	}
	{ //matching
		SiftMatchGPU* siftMatcher = new SiftMatchGPU(maxKeysPerImage);
		siftMatcher->InitSiftMatch();
		const float ratioMax = 0.8f;
		const float matchThresh = 0.7f;
		unsigned int numImages = siftManager.getNumImages();
		unsigned int cur = numImages - 1; //frame to match

		std::cout << "running sift matching... ";
		cudaDeviceSynchronize(); t.start();
		SIFTImageGPU& curImage = siftManager.getImageGPU(cur);
		int num2 = (int)siftManager.getNumKeyPointsPerImage(cur);
		if (num2 == 0) {
			std::cout << "ERROR: current image has no keys" << std::endl;
			SAFE_DELETE(siftMatcher);
			return;
		}
		for (unsigned int prev = 0; prev < cur; prev++) {
			SIFTImageGPU& prevImage = siftManager.getImageGPU(prev);
			int num1 = (int)siftManager.getNumKeyPointsPerImage(prev);
			if (num1 == 0) continue;

			uint2 keyPointOffset = make_uint2(0, 0);
			ImagePairMatch& imagePairMatch = siftManager.getImagePairMatch(prev, cur, keyPointOffset);
			siftMatcher->SetDescriptors(0, num1, (unsigned char*)prevImage.d_keyPointDescs);
			siftMatcher->SetDescriptors(1, num2, (unsigned char*)curImage.d_keyPointDescs);
			siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset, matchThresh, ratioMax);
		}
		cudaDeviceSynchronize(); t.stop(); matchTime = t.getElapsedTimeMS();
		std::cout << "done!" << std::endl;
		std::cout << "total time match: " << matchTime << " ms (" << cur << " frame pairs)" << std::endl;
		std::cout << "mean time match:  " << (matchTime / cur) << " ms" << std::endl;
		std::cout << std::endl;
		SAFE_DELETE(siftMatcher);
	}
}
