////////////////////////////////////////////////////////////////////////////
//	File:		CuTexImage.cpp
//	Author:		Changchang Wu
//	Description : implementation of the CuTexImage class.
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
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>
using namespace std;


#include <cuda.h>
#include <cuda_runtime_api.h>

#include "GlobalUtil.h"
#include "CuTexImage.h" 
#include "ProgramCU.h"

#if CUDA_VERSION <= 2010 && defined(SIFTGPU_ENABLE_LINEAR_TEX2D) 
#error "Require CUDA 2.2 or higher"
#endif


CuTexImage::CuTexImage()
{
	_cuData = NULL;
	_cuData2D = NULL;
	_numChannel = _numBytes = 0;
	_imgWidth = _imgHeight = _texWidth = _texHeight = 0;
	m_external = false;
}

CuTexImage::~CuTexImage()
{
	if (!m_external) {
		if (_cuData) cudaFree(_cuData);
	}
	if (_cuData2D)  cudaFreeArray(_cuData2D);
}

void CuTexImage::SetImageSize(int width, int height)
{
	_imgWidth = width;
	_imgHeight = height;
}

void CuTexImage::InitTexture(int width, int height, int nchannel)
{
	int size; 
	_imgWidth = width;
	_imgHeight = height;
	_numChannel = min(max(nchannel, 1), 4);

	size = width * height * _numChannel * sizeof(float);

	if(size <= _numBytes) return;
	
	if(_cuData) cudaFree(_cuData);
	
	//allocate the array data
	cudaMalloc(&_cuData, _numBytes = size);

#ifdef _DEBUG
	ProgramCU::CheckErrorCUDA("CuTexImage::InitTexture");
#endif
}

void CuTexImage::CopyFromHost(const void * buf)
{
	if(_cuData == NULL) return;
	cudaMemcpy( _cuData, buf, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyHostToDevice);
}


void CuTexImage::CopyToDevice(CuTexImage* other) const
{
	other->InitTexture(_imgWidth, _imgHeight, _numChannel);
	cudaMemcpy(other->_cuData, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyDeviceToDevice);
}

void CuTexImage::CopyToHost(void * buf)
{
	if(_cuData == NULL) return;
	cudaMemcpy(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyDeviceToHost);
}

void CuTexImage::CopyToHost(void * buf, int stream)
{
	if(_cuData == NULL) return;
	cudaMemcpyAsync(buf, _cuData, _imgWidth * _imgHeight * _numChannel * sizeof(float), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

void CuTexImage::InitTexture2D()
{
#if !defined(SIFTGPU_ENABLE_LINEAR_TEX2D) 
	if(_cuData2D && (_texWidth < _imgWidth || _texHeight < _imgHeight))
	{
		cudaFreeArray(_cuData2D); 
		_cuData2D = NULL;
	}

	if(_cuData2D == NULL)
	{
		_texWidth = max(_texWidth, _imgWidth);
		_texHeight = max(_texHeight, _imgHeight);
		cudaChannelFormatDesc desc;
		desc.f = cudaChannelFormatKindFloat;
		desc.x = sizeof(float) * 8;
		desc.y = _numChannel >=2 ? sizeof(float) * 8 : 0;
		desc.z = _numChannel >=3 ? sizeof(float) * 8 : 0;
		desc.w = _numChannel >=4 ? sizeof(float) * 8 : 0;
		cudaMallocArray(&_cuData2D, &desc, _texWidth, _texHeight); 
		ProgramCU::CheckErrorCUDA("cudaMallocArray");
	}
#endif
}

void CuTexImage::CopyToTexture2D()
{
#if !defined(SIFTGPU_ENABLE_LINEAR_TEX2D) 
	InitTexture2D();

	if(_cuData2D)
	{
		cudaMemcpy2DToArray(_cuData2D, 0, 0, _cuData, _imgWidth* _numChannel* sizeof(float) , 
		_imgWidth * _numChannel*sizeof(float), _imgHeight,	cudaMemcpyDeviceToDevice); 
		ProgramCU::CheckErrorCUDA("cudaMemcpy2DToArray");
	}
#endif

}

int CuTexImage::DebugCopyToTexture2D()
{

/*	CuTexImage tex;
	float data1[2][3] = {{1, 2, 5}, {3, 4, 5}}, data2[2][5];
	tex.InitTexture(3, 2, 1);
	cudaMemcpy(tex._cuData, data1[0], 6 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data1, tex._cuData, 4 * sizeof(float) , cudaMemcpyDeviceToHost);
	tex._texWidth =5;  tex._texHeight = 2;
	tex.CopyToTexture2D();
	cudaMemcpyFromArray(data2[0], tex._cuData2D, 0, 0, 10 * sizeof(float), cudaMemcpyDeviceToHost);*/
	
	return 1;
}

void CuTexImage::Memset( int value /*= 0*/)
{
	if (_cuData == NULL) return;
	cudaMemset(_cuData, value, _imgWidth * _imgHeight * _numChannel * sizeof(float));
}


