////////////////////////////////////////////////////////////////////////////
//	File:		SiftPyramid.h
//	Author:		Changchang Wu
//	Description : interface for the SiftPyramid class.
//		SiftPyramid:			data storage for SIFT
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

#ifndef _SIFT_PYRAMID_H
#define _SIFT_PYRAMID_H

#include <cuda_runtime.h>
class CuTexImage;
class SiftParam;
class GlobalUtil;
class CUDATimer;

/////////////////////////////////////////////////////////////////////////////
//class SiftPyramid
//description: virutal class of SIFT data pyramid
//			   provides functions for SiftPU to run steps of GPU SIFT
/////////////////////////////////////////////////////////////////////////////

#define NO_DUPLICATE_DOWNLOAD

class SiftPyramid : public GlobalUtil
{
public:
	enum{
		DATA_GAUSSIAN = 0,
		DATA_DOG = 1,
		DATA_KEYPOINT = 2,
		DATA_GRAD = 3,
		DATA_ROT = 4,
		DATA_NUM = 5
	};
	enum{
		SIFT_SKIP_FILTERING = 0x01,
		SIFT_SKIP_DETECTION = 0x02,
		SIFT_SKIP_ORIENTATION = 0x04,
		SIFT_RECT_DESCRIPTION = 0x08
	};

	SiftPyramid(SiftParam&sp);
	~SiftPyramid();;

	//shared by all implementations
	void RunSIFT(float* d_colorData, const float* d_depthData);
	void BuildPyramid(float* d_data);

	void DestroyPerLevelData();
	void DestroyPyramidData();

	void GetKeyPointsCUDA(float4* d_keypoints, const float* d_depthData, unsigned int maxNumKeyPoints); //TODO limit in compute instead?
	void GetFeatureVectorCUDA(unsigned char* d_descriptor, unsigned int maxNumKeyPoints);
	//void CopyFeaturesToCPU(float*keys, float *descriptors);
	//implementation-dependent functions
	void GetFeatureDescriptors();
	void ReshapeFeatureList();
	void ResizePyramid(int w, int h);
	void InitPyramid(int w, int h);
	void DetectKeypoints(const float* d_depthData);
	void ComputeGradient();
	void GetFeatureOrientations();

	////////////////////////////////
	int  IsUsingRectDescription() { return 0; }
	static  int  GetRequiredOctaveNum(int inputsz);

	///inline functions, shared by all implementations
	inline int GetFeatureNum(){ return _featureNum; }
	inline const int* GetLevelFeatureNum(){ return _levelFeatureNum; }
	inline unsigned int getPyramidWidth() const { return _pyramid_width; }
	inline unsigned int getPyramidHeight() const { return _pyramid_height; }
	inline bool isAllocated() const { return _allocated; }
	inline void setOctaveMin(int d) { _octave_min = d; }

	//////////
	void CopyGradientTex();
	void FitPyramid(int w, int h);

	int ResizeFeatureStorage();
	//void SetLevelFeatureNum(int idx, int fcount);
	void SetLevelFinalFeatureNum(int idx, int fcount);
	CuTexImage* GetBaseLevel(int octave, int dataName = DATA_GAUSSIAN);
	//////////////////////////
	//static int CheckCudaDevice(int device);
	static int TruncateWidth(int w) { return w & 0xfffffffc; }

	void EvaluateTimings();
protected:
	inline  void PrepareBuffer();
	inline  void LimitFeatureCount(int have_keylist = 0);

	void CreateGlobalKeyPointList(float4* d_keypoints, const float* d_depthData, unsigned int maxNumKeyPoints);

	SiftParam&	param;
	int*		_levelFeatureNum;
	int			_featureNum;

	//image size related
	//first octave
	int			_octave_min;
	//how many octaves
	int			_octave_num;
	//pyramid storage
	int			_pyramid_octave_num;
	int			_pyramid_octave_first;
	int			_pyramid_width;
	int			_pyramid_height;
	bool		_allocated;

	// data
	CuTexImage* _inputTex;
	CuTexImage* _allPyramid;
	CuTexImage* _featureTexRaw;		//raw feature points; may contain zero to two orientations
	CuTexImage* _featureTexFinal;	//final feature points; only a single orientation per key point
	CuTexImage* _orientationTex;

	int* d_featureCount; // per dog per level
	float* d_outDescriptorList;

	CUDATimer* _timer;
};

#define SIFTGPU_ENABLE_REVERSE_ORDER
#ifdef SIFTGPU_ENABLE_REVERSE_ORDER
#define FIRST_OCTAVE(R)            (R? _octave_num - 1 : 0)
#define NOT_LAST_OCTAVE(i, R)      (R? (i >= 0) : (i < _octave_num))
#define GOTO_NEXT_OCTAVE(i, R)     (R? (--i) : (++i))
#define FIRST_LEVEL(R)             (R? param._dog_level_num - 1 : 0)   
#define GOTO_NEXT_LEVEL(j, R)      (R? (--j) : (++j))
#define NOT_LAST_LEVEL(j, R)       (R? (j >= 0) : (j < param._dog_level_num))
#define FOR_EACH_OCTAVE(i, R)      for(int i = FIRST_OCTAVE(R); NOT_LAST_OCTAVE(i, R); GOTO_NEXT_OCTAVE(i, R))
#define FOR_EACH_LEVEL(j, R)       for(int j = FIRST_LEVEL(R);  NOT_LAST_LEVEL(j, R);  GOTO_NEXT_LEVEL(j, R))
#else
#define FOR_EACH_OCTAVE(i, R) for(int i = 0; i < _octave_num; ++i)
#define FOR_EACH_LEVEL(j, R)  for(int j = 0; j < param._dog_level_num; ++j)
#endif

#endif 
