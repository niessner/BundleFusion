
#pragma once


#define MINF __int_as_float(0xff800000)

#define MAX_MATCHES_PER_IMAGE_PAIR_RAW 128
#define MAX_MATCHES_PER_IMAGE_PAIR_FILTERED 25

//#define MAX_NUM_INVALID


//DLL EXPORT


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

