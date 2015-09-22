////////////////////////////////////////////////////////////////////////////
//	File:		GlobalUtil.cpp
//	Author:		Changchang Wu
//	Description : Global Utility class for SiftGPU
//
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
#include "stdafx.h"

#include <string.h>
#include <iostream>
using std::cout;

#include "GlobalUtil.h"


int GlobalUtil::_MaxFilterWidth = -1;	//maximum filter width, use when GPU is not good enough
int GlobalUtil::_SubpixelLocalization = 1; //sub-pixel and sub-scale localization 	
int	GlobalUtil::_MaxOrientation = 2;	//whether we find multiple orientations for each feature 
int	GlobalUtil::_OrientationPack2 = 0;  //use one float to store two orientations
float GlobalUtil::_MaxFeaturePercent = 0.005f;//at most 0.005 of all pixels
int	GlobalUtil::_MaxLevelFeatureNum = 4096; //maximum number of features of a level

//hardware parameter,   automatically retrieved
int GlobalUtil::_texMaxDim = 3200;	//Maximum working size for SiftGPU, 3200 for packed
int GlobalUtil::_texMinDim = 16; //
int	GlobalUtil::_MemCapGPU = 0;
int GlobalUtil::_FitMemoryCap = 0;

//when SiftGPUEX is not used, display VBO generation is skipped
int GlobalUtil::_InitPyramidWidth = 0;
int GlobalUtil::_InitPyramidHeight = 0;
int	GlobalUtil::_octave_min_default = 0;
int	GlobalUtil::_octave_num_default = -1;


//////////////////////////////////////////////////////////////////
int	GlobalUtil::_FixedOrientation = 0; //upright
int	GlobalUtil::_LoweOrigin = 0;       //(0, 0) to be at the top-left corner.
int	GlobalUtil::_NormalizedSIFT = 1;   //normalize descriptor
///
int GlobalUtil::_TruncateMethod = 0;


int GlobalUtil::_FeatureCountThreshold = -1;

bool GlobalUtil::_EnableDetailedTimings = false;
float GlobalUtil::_SiftDepthMin = 0.1f;
float GlobalUtil::_SiftDepthMax = 3.0f;








