////////////////////////////////////////////////////////////////////////////
//	File:		SiftGPU.cpp
//	Author:		Changchang Wu
//	Description :	Implementation of the SIFTGPU classes.
//					SiftGPU:	The SiftGPU Tool.  
//					SiftParam:	Sift Parameters
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
#include "stdafx.h"

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <math.h>

#include <time.h>
using namespace std;


#include "GlobalUtil.h"
#include "SiftGPU.h"
#include "SiftPyramid.h"
#include "ProgramCU.h"


#include "direct.h"
#pragma warning (disable : 4786) 
#pragma warning (disable : 4996) 

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

SiftGPU::SiftGPU()
{
	_initialized = 0;
	_image_loaded = 0;
	_current = 0;
	_pyramid = NULL;
}

SiftGPU::~SiftGPU()
{
	if (_pyramid) delete _pyramid;
}


void SiftGPU::InitSiftGPU()
{
	if (_initialized) return;

	//Parse sift parameters
	ParseSiftParam();

	_pyramid = new SiftPyramid(*this);

	if ((GlobalUtil::_InitPyramidWidth & 0xfffffffc) != GlobalUtil::_InitPyramidWidth) {
		std::cout << "ERROR: image width must be a multiple of 4" << std::endl;
		return;
	}

	if (GlobalUtil::_InitPyramidWidth > 0 && GlobalUtil::_InitPyramidHeight > 0)
	{
		_pyramid->InitPyramid(GlobalUtil::_InitPyramidWidth, GlobalUtil::_InitPyramidHeight);
	}

	_initialized = 1;
}

int SiftGPU::RunSIFT(float* d_colorData, const float* d_depthData)
{

	if (!_initialized) return 0;


	if (d_colorData != NULL)
	{
		if (!_pyramid->isAllocated()) return 0;

		//process the image
		_pyramid->RunSIFT(d_colorData, d_depthData);

		return 1;
	}
	return 0;
}


SiftParam::SiftParam()
{

	_level_min = -1;
	_dog_level_num = 3;
	_level_max = 0;
	_sigma0 = 0;
	_sigman = 0;
	_edge_threshold = 0;
	_dog_threshold = 0;


}

float SiftParam::GetInitialSmoothSigma(int octave_min)
{
	float	sa = _sigma0 * powf(2.0f, float(_level_min) / float(_dog_level_num));
	float   sb = _sigman / powf(2.0f, float(octave_min));//
	float   sigma_skip0 = sa > sb + 0.001 ? sqrt(sa*sa - sb*sb) : 0.0f;
	return  sigma_skip0;
}

void SiftParam::ParseSiftParam()
{

	if (_dog_level_num == 0) _dog_level_num = 3;
	if (_level_max == 0) _level_max = _dog_level_num + 1;
	if (_sigma0 == 0.0f) _sigma0 = 1.6f * powf(2.0f, 1.0f / _dog_level_num);
	if (_sigman == 0.0f) _sigman = 0.5f;


	_level_num = _level_max - _level_min + 1;

	_level_ds = _level_min + _dog_level_num;
	if (_level_ds > _level_max) _level_ds = _level_max;

	///
	float _sigmak = powf(2.0f, 1.0f / _dog_level_num);
	float dsigma0 = _sigma0 * sqrt(1.0f - 1.0f / (_sigmak*_sigmak));
	float sa, sb;


	sa = _sigma0 * powf(_sigmak, (float)_level_min);
	sb = _sigman / powf(2.0f, (float)GlobalUtil::_octave_min_default);//

	_sigma_skip0 = sa > sb + 0.001 ? sqrt(sa*sa - sb*sb) : 0.0f;

	sa = _sigma0 * powf(_sigmak, float(_level_min));
	sb = _sigma0 * powf(_sigmak, float(_level_ds - _dog_level_num));

	_sigma_skip1 = sa > sb + 0.001 ? sqrt(sa*sa - sb*sb) : 0.0f;

	_sigma_num = _level_max - _level_min;
	_sigma = new float[_sigma_num];

	for (int i = _level_min + 1; i <= _level_max; i++)
	{
		_sigma[i - _level_min - 1] = dsigma0 * powf(_sigmak, float(i));
	}

	if (_dog_threshold == 0)	_dog_threshold = 0.02f / _dog_level_num;
	if (_edge_threshold == 0) _edge_threshold = 10.0f;

	std::vector<float> sigmas;
	sigmas.push_back(GetInitialSmoothSigma(GlobalUtil::_octave_min_default));
	for (int i = _level_min + 1; i <= _level_max; i++) {
		sigmas.push_back(_sigma[i]);
	}
	ProgramCU::InitFilterKernels(sigmas, m_filterWidths);
}

void SiftGPU::PrintUsage()
{
	std::cout
		<< "SiftGPU Usage:\n"
		<< "-h -help          : Parameter information\n"
		<< "-i <strings>      : Filename(s) of the input image(s)\n"
		<< "-il <string>      : Filename of an image list file\n"
		<< "-o <string>       : Where to save SIFT features\n"
		<< "-f <float>        : Filter width factor; Width will be 2*factor+1 (default : 4.0)\n"
		<< "-w  <float>       : Orientation sample window factor (default: 2.0)\n"
		<< "-dw <float>  *    : Descriptor grid size factor (default : 3.0)\n"
		<< "-fo <int>    *    : First octave to detect DOG keypoints(default : 0)\n"
		<< "-no <int>         : Maximum number of Octaves (default : no limit)\n"
		<< "-d <int>          : Number of DOG levels in an octave (default : 3)\n"
		<< "-t <float>        : DOG threshold (default : 0.02/3)\n"
		<< "-e <float>        : Edge Threshold (default : 10.0)\n"
		<< "-m  <int=2>       : Multi Feature Orientations (default : 1)\n"
		<< "-m2p              : 2 Orientations packed as one float\n"
		<< "-s  <int=1>       : Sub-Pixel, Sub-Scale Localization, Multi-Refinement(num)\n"
		<< "-lcpu -lc <int>   : CPU/GPU mixed Feature List Generation (defaut : 6)\n"
		<< "                    Use GPU first, and use CPU when reduction size <= pow(2,num)\n"
		<< "                    When <num> is missing or equals -1, no GPU will be used\n"
		<< "-noprep           : Upload raw data to GPU (default: RGB->LUM and down-sample on CPU)\n"
		<< "-sd               : Skip descriptor computation if specified\n"
		<< "-unn    *         : Write unnormalized descriptor if specified\n"
		<< "-b      *         : Write binary sift file if specified\n"
		<< "-fs <int>         : Block Size for freature storage <default : 4>\n"
		<< "-cuda <int=0>     : Use CUDA SiftGPU, and specifiy the device index\n"
		<< "-tight            : Automatically resize pyramid to fit new images tightly\n"
		<< "-p  <W>x<H>       : Inititialize the pyramids to contain image of WxH (eg -p 1024x768)\n"
		<< "-tc[1|2|3] <int> *: Threshold for limiting the overall number of features (3 methods)\n"
		<< "-v <int>          : Level of timing details. Same as calling Setverbose() function\n"
		<< "-loweo            : (0, 0) at center of top-left pixel (defaut: corner)\n"
		<< "-maxd <int> *     : Max working dimension (default : 2560 (unpacked) / 3200 (packed))\n"
		<< "-nomc             : Disabling auto-downsamping that try to fit GPU memory cap\n"
		<< "-exit             : Exit program after processing the input image\n"
		<< "-unpack           : Use the old unpacked implementation\n"
		<< "-di               : Use dynamic array indexing if available (defualt : no)\n"
		<< "                    It could make computation faster on cards like GTX 280\n"
		<< "-ofix     *       : use 0 as feature orientations.\n"
		<< "-ofix-not *       : disable -ofix.\n"
		<< "-winpos <X>x<Y> * : Screen coordinate used in Win32 to select monitor/GPU.\n"
		<< "-display <string>*: Display name used in Linux/Mac to select monitor/GPU.\n"
		<< "\n"
		<< "NOTE: parameters marked with * can be changed after initialization\n"
		<< "\n";
}

void SiftGPU::SetParams(unsigned int siftWidth, unsigned int siftHeight, bool enableTiming, unsigned int featureCountThreshold, float siftDepthMin, float siftDepthMax)
{
	GlobalUtil::_SiftDepthMin = siftDepthMin;
	GlobalUtil::_SiftDepthMax = siftDepthMax;

	//!!!TODO TRY THIS
	//GlobalUtil::_LoweOrigin = 1;

	// soft limit for num features detected
	GlobalUtil::_FeatureCountThreshold = (int)featureCountThreshold;

	// pyramid size to allocate
	GlobalUtil::_InitPyramidWidth = siftWidth;
	GlobalUtil::_InitPyramidHeight = siftHeight;

	// don't use subpixel localization
	GlobalUtil::_SubpixelLocalization = 0;

	// first octave to detect  = 0;
	GlobalUtil::_octave_min_default = 0;
	// use 4 octaves
	GlobalUtil::_octave_num_default = 4;

	if (GlobalUtil::_MaxOrientation < 2) {
		std::cout << "MAX ORIENTATION != 2 not supported" << std::endl;
		while (1);
	}

	if (enableTiming) GlobalUtil::_EnableDetailedTimings = true;
}


float SiftParam::GetLevelSigma(int lev)
{
	return _sigma0 * powf(2.0f, float(lev) / float(_dog_level_num)); //bug fix 9/12/2007
}

int SiftGPU::GetFeatureNum()
{
	return _pyramid->GetFeatureNum();
}


unsigned int SiftGPU::GetKeyPointsAndDescriptorsCUDA(SIFTImageGPU& siftImage, const float* d_depthData, unsigned int maxNumKeyPoints /*= (unsigned int)-1*/)
{
	_pyramid->GetKeyPointsCUDA((float4*)siftImage.d_keyPoints, d_depthData, maxNumKeyPoints);
	_pyramid->GetFeatureVectorCUDA((unsigned char*)siftImage.d_keyPointDescs, maxNumKeyPoints);
	return _pyramid->GetFeatureNum();
}

void SiftGPU::GetKeyPointsCUDA(SiftKeypoint* d_keypoints, float* d_depthData, unsigned int maxNumKeyPoints /*= (unsigned int)-1*/)
{
	_pyramid->GetKeyPointsCUDA((float4*)d_keypoints, d_depthData, maxNumKeyPoints);
}

void SiftGPU::GetDescriptorsCUDA(unsigned char* d_descriptors, unsigned int maxNumKeyPoints /*= (unsigned int)-1*/)
{
	_pyramid->GetFeatureVectorCUDA(d_descriptors, maxNumKeyPoints);
}

//void SiftGPU::CopyFeatureVectorToCPU(SiftKeypoint * keys, float * descriptors)
//{
//	//	keys.resize(_pyramid->GetFeatureNum());
//	if (GlobalUtil::_DescriptorPPT)
//	{
//		//	descriptors.resize(128*_pyramid->GetFeatureNum());
//		_pyramid->CopyFeaturesToCPU((float*)(keys), descriptors);
//	}
//	else
//	{
//		//descriptors.resize(0);
//		_pyramid->CopyFeaturesToCPU((float*)(&keys[0]), NULL);
//	}
//}

int SiftGPU::AllocatePyramid(int width, int height)
{
	_pyramid->setOctaveMin(GlobalUtil::_octave_min_default);
	if (GlobalUtil::_octave_min_default >= 0)
	{
		width >>= GlobalUtil::_octave_min_default;
		height >>= GlobalUtil::_octave_min_default;
	}
	else
	{
		width <<= (-GlobalUtil::_octave_min_default);
		height <<= (-GlobalUtil::_octave_min_default);
	}
	_pyramid->ResizePyramid(width, height);
	return _pyramid->getPyramidHeight() == height && width == _pyramid->getPyramidWidth();
}


void SiftGPU::EvaluateTimings()
{
	_pyramid->EvaluateTimings();
}

SiftGPU* CreateNewSiftGPU()
{
	return new SiftGPU();
}


