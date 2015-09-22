////////////////////////////////////////////////////////////////////////////
//	File:		GlobalUtil.h
//	Author:		Changchang Wu
//	Description : 
//		GlobalParam:	Global parameters
//		ClockTimer:		Timer 
//		GlobalUtil:		Global Function wrapper
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


#ifndef _GLOBAL_UTILITY_H
#define _GLOBAL_UTILITY_H


class GlobalUtil
{
public:
	static int		_texMaxDim;
    static int      _texMinDim;
	static int		_MemCapGPU;
	static int		_FitMemoryCap;
	static int		_MaxFilterWidth;
	static int		_MaxOrientation;
	static int      _OrientationPack2;
	static float	_MaxFeaturePercent;
	static int		_MaxLevelFeatureNum;
	static int		_SubpixelLocalization;
    static int      _TruncateMethod;
	static int		_octave_min_default;
	static int		_octave_num_default;
	static int		_InitPyramidWidth;
	static int		_InitPyramidHeight;
	static int		_FixedOrientation;
	static int		_LoweOrigin;
	static int		_NormalizedSIFT;
	static int		_FeatureCountThreshold;
	static bool		_EnableDetailedTimings;
	static float	_SiftDepthMin;
	static float	_SiftDepthMax;
};



#endif

