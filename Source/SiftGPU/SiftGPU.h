////////////////////////////////////////////////////////////////////////////
//	File:		SiftGPU.h
//	Author:		Changchang Wu
//	Description :	interface for the SIFTGPU class.
//					SiftGPU:	The SiftGPU Tool.  
//					SiftParam:	Sift Parameters
//					SiftMatchGPU: GPU SIFT Matcher;
//
//
//	Copyright (c) 2007 University of North Carolina at Chapel Hill
//	All Rights Reserved
//
//	Permission to use, copy, modify and distribute this software and its
//	documentation for educational, research and non-profit purposes, without
//	fee, and without a written agreement is hereby granted, provided that the
//	above copyright notice and the following paragraph appear in all copies.
//	
//	The University of North Carolina at Chapel Hill make no representations
//	about the suitability of this software for any purpose. It is provided
//	'as is' without express or implied warranty. 
//
//	Please send BUG REPORTS to ccwu@cs.unc.edu
//
////////////////////////////////////////////////////////////////////////////


#ifndef GPU_SIFT_H
#define GPU_SIFT_H

#include <vector>
#include "SIFTImageManager.h"

#if  defined(_WIN32) 
	#ifdef SIFTGPU_DLL
		#ifdef DLL_EXPORT
			#define SIFTGPU_EXPORT __declspec(dllexport)
		#else
			#define SIFTGPU_EXPORT __declspec(dllimport)
		#endif
	#else
		#define SIFTGPU_EXPORT
	#endif

    #define SIFTGPU_EXPORT_EXTERN SIFTGPU_EXPORT

	#if _MSC_VER > 1000
		#pragma once
	#endif
#else
	#define SIFTGPU_EXPORT
    #define SIFTGPU_EXPORT_EXTERN extern "C"
#endif

///////////////////////////////////////////////////////////////////
//clss SiftParam
//description: SIFT parameters
////////////////////////////////////////////////////////////////////
class GlobalUtil;
class SiftParam
{
public:
	std::vector<unsigned int> m_filterWidths;

	float*		_sigma;
	float		_sigma_skip0; // 
	float		_sigma_skip1; //
	
	//sigma of the first level
	float		_sigma0;
	float		_sigman;
	int			_sigma_num;

	//how many dog_level in an octave
	int			_dog_level_num;
	int			_level_num;

	//starting level in an octave
	int			_level_min;
	int			_level_max;
	int			_level_ds;
	//dog threshold
	float		_dog_threshold;
	//edge elimination
	float		_edge_threshold;
	void		 ParseSiftParam();
public:
	float GetLevelSigma(int lev);
	float GetInitialSmoothSigma(int octave_min);
	SIFTGPU_EXPORT SiftParam();
};

class SiftPyramid;
class ImageList;
////////////////////////////////////////////////////////////////
//class SIftGPU
//description: Interface of SiftGPU lib
////////////////////////////////////////////////////////////////
class SiftGPU:public SiftParam
{
public:
	typedef struct SiftKeypoint
	{
		float x, y, s, o; //x, y, scale, orientation.
	}SiftKeypoint;
public:
	//constructor, the parameter np is ignored..
	SIFTGPU_EXPORT SiftGPU();
	//destructor
	SIFTGPU_EXPORT ~SiftGPU();


	//Initialize OpenGL and SIFT paremeters, and create the shaders accordingly
	SIFTGPU_EXPORT void InitSiftGPU();
	//get the number of SIFT features in current image
	SIFTGPU_EXPORT  int	GetFeatureNum();


	//get sift keypoints & descriptors (compute into provided d_keypoints, d_descriptors)
	SIFTGPU_EXPORT  unsigned int GetKeyPointsAndDescriptorsCUDA(SIFTImageGPU& siftImage, const float* d_depthData);
	//get sift keypoints (compute into provided d_keypoints)
	SIFTGPU_EXPORT  void GetKeyPointsCUDA(SiftKeypoint* d_keypoints, float* d_depthData);
	//get sift descriptors (compute into provided d_descriptors)
	SIFTGPU_EXPORT  void GetDescriptorsCUDA(unsigned char* d_descriptors);

	//Copy the SIFT result to two vectors
	//SIFTGPU_EXPORT  void CopyFeatureVectorToCPU(SiftKeypoint * keys, float * descriptors);
	//parse SiftGPU parameters
	SIFTGPU_EXPORT  void SetParams(int cudaDeviceIndex, bool enableTiming, unsigned int featureCountThreshold);

	SIFTGPU_EXPORT int RunSIFT(float* d_colorData, const float* d_depthData);
	//set the active pyramid...dropped function
    SIFTGPU_EXPORT  void SetActivePyramid(int index) {}
	//allocate pyramid for a given size of image
	SIFTGPU_EXPORT  int AllocatePyramid(int width, int height);
	//none of the texture in processing can be larger
	//automatic down-sample is used if necessary. 
	SIFTGPU_EXPORT  void SetMaxDimension(int sz);

	SIFTGPU_EXPORT	void EvaluateTimings();
private:
	//when more than one images are specified
	//_current indicates the active one
	int		_current;
	//_initialized indicates if the shaders and OpenGL/SIFT parameters are initialized
	//they are initialized only once for one SiftGPU inistance
	//that is, SIFT parameters will not be changed
	int		_initialized;
	//_image_loaded indicates if the current images are loaded
	int		_image_loaded;
	//the SiftPyramid
	SiftPyramid *  _pyramid;
	//print out the command line options
	static void PrintUsage();
};



#endif 
