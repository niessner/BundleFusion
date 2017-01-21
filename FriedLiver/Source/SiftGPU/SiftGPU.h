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

///////////////////////////////////////////////////////////////////
//clss SiftParam
//description: SIFT parameters
////////////////////////////////////////////////////////////////////
class GlobalUtil;
class SiftParam
{
public:
	SiftParam();
	~SiftParam() {
		SAFE_DELETE_ARRAY(_sigma);
	}

	void		 ParseSiftParam();

	float GetLevelSigma(int lev);
	float GetInitialSmoothSigma(int octave_min);

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
	SiftGPU();
	//destructor
	~SiftGPU();


	//Initialize OpenGL and SIFT paremeters, and create the shaders accordingly
	void InitSiftGPU();
	//get the number of SIFT features in current image
	 int	GetFeatureNum();


	//get sift keypoints & descriptors (compute into provided d_keypoints, d_descriptors)
	 unsigned int GetKeyPointsAndDescriptorsCUDA(SIFTImageGPU& siftImage, const float* d_depthData, unsigned int maxNumKeyPoints = (unsigned int)-1);
	//get sift keypoints (compute into provided d_keypoints)
	 void GetKeyPointsCUDA(SiftKeypoint* d_keypoints, float* d_depthData, unsigned int maxNumKeyPoints = (unsigned int)-1);
	//get sift descriptors (compute into provided d_descriptors)
	 void GetDescriptorsCUDA(unsigned char* d_descriptors, unsigned int maxNumKeyPoints = (unsigned int)-1);

	//Copy the SIFT result to two vectors
	// void CopyFeatureVectorToCPU(SiftKeypoint * keys, float * descriptors);
	//parse SiftGPU parameters
	 void SetParams(unsigned int siftWidth, unsigned int siftHeight, bool enableTiming, unsigned int featureCountThreshold, float siftDepthMin, float siftDepthMax);

	int RunSIFT(float* d_colorData, const float* d_depthData);
	//set the active pyramid...dropped function
     void SetActivePyramid(int index) {}
	//allocate pyramid for a given size of image
	 int AllocatePyramid(int width, int height);
	//none of the texture in processing can be larger
	//automatic down-sample is used if necessary. 
	void SetMaxDimension(int sz);

	void EvaluateTimings();
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
