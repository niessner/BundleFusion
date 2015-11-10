#ifndef _CAMERA_UTIL_
#define _CAMERA_UTIL_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "cuda_SimpleMatrixUtil.h"
#include "DepthCameraUtil.h"

#ifndef BYTE
#define BYTE unsigned char
#endif

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Copy Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFloatMapDevice(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = d_input[y*width+x];
}

extern "C" void copyFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	copyFloatMapDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

__global__ void copyDepthFloatMapDevice(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float depth = d_input[y*width+x];
	if (depth >= minDepth && depth <= maxDepth) 
		d_output[y*width+x] = depth;
	else
		d_output[y*width+x] = MINF;
}

extern "C" void copyDepthFloatMap(float* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	copyDepthFloatMapDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height,minDepth, maxDepth);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Copy Float Map Fill
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void initializeOptimizerMapsDevice(float* d_output, float* d_input, float* d_input2, float* d_mask, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float depth = d_input[y*width+x];
	if(d_mask[y*width+x] != MINF) { d_output[y*width+x] = depth; }
	else						  { d_output[y*width+x] = MINF; d_input[y*width+x] = MINF; d_input2[y*width+x] = MINF; }
}

extern "C" void initializeOptimizerMaps(float* d_output, float* d_input, float* d_input2, float* d_mask, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	initializeOptimizerMapsDevice<<<gridSize, blockSize>>>(d_output, d_input, d_input2, d_mask, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void copyFloat4MapDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = d_input[y*width+x];
}

extern "C" void copyFloat4Map(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	copyFloat4MapDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Raw Color to float
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorRawToFloatDevice(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;
	
	uchar4 c = make_uchar4(d_input[4*(y*width+x)+0], d_input[4*(y*width+x)+1], d_input[4*(y*width+x)+2], d_input[4*(y*width+x)+3]);
	//if (c.x == 0 && c.y == 0 && c.z == 0) { // NO INVALID COLORS!
	//	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF); 
	//} else {
		d_output[y*width+x] = make_float4(c.x/255.0f, c.y/255.0f, c.z/255.0f, c.w/255);	
	//}
}

extern "C" void convertColorRawToFloat4(float4* d_output, BYTE* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertColorRawToFloatDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Float4 Color to UCHAR4
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorFloat4ToUCHAR4Device(uchar4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	float4 color = d_input[y*width+x];
	d_output[y*width+x] = make_uchar4(color.x*255.0f, color.y*255.0f, color.z*255.0f, color.w*255.0f);	
}

extern "C" void convertColorFloat4ToUCHAR4(uchar4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);

	convertColorFloat4ToUCHAR4Device<<<blockSize, gridSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Mask Color Map using Depth
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void maskColorMapFloat4MapDevice(float4* d_inputColor, float4* d_inputDepth, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	float4 color = d_inputColor[y*width+x];
	
	if(d_inputDepth[y*width+x].x != MINF)	d_inputColor[y*width+x] = color;	
	else									d_inputColor[y*width+x] = make_float4(MINF, MINF, MINF, MINF);
}

extern "C" void maskColorMapFloat4Map(float4* d_inputColor, float4* d_inputDepth, unsigned int width, unsigned int height)
{
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);

	maskColorMapFloat4MapDevice<<<blockSize, gridSize>>>(d_inputColor, d_inputDepth, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Color to Intensity Float4
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorToIntensityFloat4Device(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float4 color = d_input[y*width+x];
	const float I = 0.299f*color.x + 0.587f*color.y + 0.114f*color.z;

	d_output[y*width+x] = make_float4(I, I, I, 1.0f);
}

extern "C" void convertColorToIntensityFloat4(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertColorToIntensityFloat4Device<<<gridSize, blockSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Color to Intensity
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertColorToIntensityFloatDevice(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float4 color = d_input[y*width+x];
	d_output[y*width+x] = 0.299f*color.x + 0.587f*color.y + 0.114f*color.z;
}

extern "C" void convertColorToIntensityFloat(float* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertColorToIntensityFloatDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert depth map to color map view
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthToColorSpaceDevice(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInv, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < depthWidth && y < depthHeight)
	{
		const float depth = d_input[y*depthWidth+x];

		if(depth != MINF && depth < 1.0f)
		{
			// Cam space depth
			float4 depthCamSpace = depthIntrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth);
			depthCamSpace = make_float4(depthCamSpace.x, depthCamSpace.y, depthCamSpace.w, 1.0f);

			// World Space
			const float4 worldSpace = depthExtrinsicsInv*depthCamSpace;

			// Cam space color
			float4 colorCamSpace = colorExtrinsics*worldSpace;
			//colorCamSpace = make_float4(colorCamSpace.x, colorCamSpace.y, 0.0f, colorCamSpace.z);
			colorCamSpace = make_float4(colorCamSpace.x, colorCamSpace.y, colorCamSpace.z, 1.0f);

			// Get coordinates in color image and set pixel to new depth
			const float4 screenSpaceColor  = colorIntrinsics*colorCamSpace;
			//const unsigned int cx = (unsigned int)(screenSpaceColor.x/screenSpaceColor.w + 0.5f);
			//const unsigned int cy = (unsigned int)(screenSpaceColor.y/screenSpaceColor.w + 0.5f);
			const unsigned int cx = (unsigned int)(screenSpaceColor.x/screenSpaceColor.z + 0.5f);
			const unsigned int cy = (unsigned int)(screenSpaceColor.y/screenSpaceColor.z + 0.5f);

			//if(cx < colorWidth && cy < colorHeight) d_output[cy*colorWidth+cx] = screenSpaceColor.w; // Check for minimum !!!
			if(cx < colorWidth && cy < colorHeight) d_output[cy*colorWidth+cx] = screenSpaceColor.z; // Check for minimum !!!
		}
	}
}

extern "C" void convertDepthToColorSpace(float* d_output, float* d_input, float4x4 depthIntrinsicsInv, float4x4 colorIntrinsics, float4x4 depthExtrinsicsInv, float4x4 colorExtrinsics, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
{
	const dim3 gridSize((depthWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (depthHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertDepthToColorSpaceDevice<<<gridSize, blockSize>>>(d_output, d_input, depthIntrinsicsInv, colorIntrinsics, depthExtrinsicsInv, colorExtrinsics, depthWidth, depthHeight, colorWidth, colorHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set invalid float map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setInvalidFloatMapDevice(float* d_output, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = MINF;
}

extern "C" void setInvalidFloatMap(float* d_output, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	setInvalidFloatMapDevice<<<gridSize, blockSize>>>(d_output, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Set invalid float4 map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void setInvalidFloat4MapDevice(float4* d_output, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF ,MINF);
}

extern "C" void setInvalidFloat4Map(float4* d_output, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	setInvalidFloat4MapDevice<<<gridSize, blockSize>>>(d_output, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Depth to Camera Space Positions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertDepthFloatToCameraSpaceFloat4Device(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

		float depth = d_input[y*width+x];

		if(depth != MINF)
		{
			//float4 cameraSpace(intrinsicsInv*make_float4((float)x*depth, (float)y*depth, depth, depth));
			//d_output[y*width+x] = make_float4(cameraSpace.x, cameraSpace.y, cameraSpace.w, 1.0f);
			d_output[y*width + x] = make_float4(DepthCameraData::kinectDepthToSkeleton(x, y, depth), 1.0f);
		}
	}
}

extern "C" void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertDepthFloatToCameraSpaceFloat4Device<<<gridSize, blockSize>>>(d_output, d_input, intrinsicsInv, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral Filter Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

inline __device__ float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f-(dist*dist)/(2.0*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x)
{
	return exp(-((x*x)/(2.0f*sigma*sigma)));
}

__global__ void bilateralFilterFloatMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	d_output[y*width+x] = MINF;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = d_input[y*width+x];
	if(depthCenter != MINF)
	{
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{		
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = d_input[n*width+m];

					if (currentDepth != MINF) {
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, currentDepth-depthCenter);

						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}

		if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
	}
}

extern "C" void bilateralFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	bilateralFilterFloatMapDevice<<<gridSize, blockSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bilateral Filter Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void bilateralFilterFloat4MapDevice(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	//d_output[y*width+x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float4 depthCenter = d_input[y*width+x];
	if (depthCenter.x != MINF) {
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{		
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float4 currentDepth = d_input[n*width+m];

					if (currentDepth.x != MINF) {
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, length(currentDepth-depthCenter));

						sum += weight*currentDepth;
						sumWeight += weight;
					}
				}
			}
		}
	}
	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
}

extern "C" void bilateralFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize(T_PER_BLOCK, T_PER_BLOCK);	
	const dim3 blockSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);	

	bilateralFilterFloat4MapDevice<<<gridSize, blockSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Gauss Filter Float Map
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//__global__ void gaussFilterFloatMapDevice(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
//{
//	const int x = blockIdx.x*blockDim.x + threadIdx.x;
//	const int y = blockIdx.y*blockDim.y + threadIdx.y;
//
//	if(x >= width || y >= height) return;
//
//	const int kernelRadius = (int)ceil(2.0*sigmaD);
//
//	d_output[y*width+x] = MINF;
//
//	float sum = 0.0f;
//	float sumWeight = 0.0f;
//
//	const float depthCenter = d_input[y*width+x];
//	if(depthCenter != MINF)
//	{
//		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
//		{
//			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
//			{		
//				if(m >= 0 && n >= 0 && m < width && n < height)
//				{
//					const float currentDepth = d_input[n*width+m];
//
//					if(currentDepth != MINF && fabs(depthCenter-currentDepth) < sigmaR)
//					{
//						const float weight = gaussD(sigmaD, m-x, n-y);
//
//						sumWeight += weight;
//						sum += weight*currentDepth;
//					}
//				}
//			}
//		}
//	}
//
//	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
//}
//
//extern "C" void gaussFilterFloatMap(float* d_output, float* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
//{
//	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
//	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
//
//	gaussFilterFloatMapDevice<<<gridSize, blockSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
//	#ifdef _DEBUG
//		cutilSafeCall(cudaDeviceSynchronize());
//		cutilCheckMsg(__FUNCTION__);
//	#endif
//}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gauss Filter Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussFilterFloat4MapDevice(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	//d_output[y*width+x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float sumWeight = 0.0f;

	const float4 depthCenter = d_input[y*width+x];
	if (depthCenter.x != MINF) {
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{		
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float4 currentDepth = d_input[n*width+m];

					if (currentDepth.x != MINF) {
						if(length(depthCenter-currentDepth) < sigmaR)
						{
							const float weight = gaussD(sigmaD, m-x, n-y);

							sumWeight += weight;
							sum += weight*currentDepth;
						}
					}
				}
			}
		}
	}

	if(sumWeight > 0.0f) d_output[y*width+x] = sum / sumWeight;
}

extern "C" void gaussFilterFloat4Map(float4* d_output, float4* d_input, float sigmaD, float sigmaR, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	gaussFilterFloat4MapDevice<<<gridSize, blockSize>>>(d_output, d_input, sigmaD, sigmaR, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeNormalsDevice(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float4 CC = d_input[(y+0)*width+(x+0)];
		const float4 PC = d_input[(y+1)*width+(x+0)];
		const float4 CP = d_input[(y+0)*width+(x+1)];
		const float4 MC = d_input[(y-1)*width+(x+0)];
		const float4 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(PC)-make_float3(MC), make_float3(CP)-make_float3(CM));
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = make_float4(n/-l, 1.0f);
			}
		}
	}
}

extern "C" void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormalsDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Normal Map 2
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeNormalsDevice2(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = make_float4(MINF, MINF, MINF, MINF);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float4 CC = d_input[(y+0)*width+(x+0)];
		const float4 MC = d_input[(y-1)*width+(x+0)];
		const float4 CM = d_input[(y+0)*width+(x-1)];

		if(CC.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(MC)-make_float3(CC), make_float3(CM)-make_float3(CC));
			const float  l = length(n);

			if(l > 0.0f)
			{
				d_output[y*width+x] = make_float4(n/-l, 1.0f);
			}
		}
	}
}

extern "C" void computeNormals2(float4* d_output, float4* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeNormalsDevice2<<<gridSize, blockSize>>>(d_output, d_input, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Shading Value
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void evaluateLightingModelTerms(float* d_out, float4 n)
{
	d_out[0] = 1.0;
	d_out[1] = n.y;
	d_out[2] = n.z;
	d_out[3] = n.x;
	d_out[4] = n.x*n.y;
	d_out[5] = n.y*n.z;
	d_out[6] = 3*n.z*n.z - 1;
	d_out[7] = n.z*n.x;
	d_out[8] = n.x*n.x-n.y*n.y;
}

inline __device__ float evaluateLightingModel(float* d_lit, float4 n)
{
	float tmp[9];
	evaluateLightingModelTerms(tmp, n);

	float sum = 0.0f;
	for(unsigned int i = 0; i<9; i++) sum += tmp[i]*d_lit[i];

	return sum;
}

__global__ void computeShadingValueDevice(float* d_outShading, float* d_indepth, float4* d_normals, float* d_clusterIDs, float* d_albedoEstimates, float4x4 Intrinsic, float* d_litcoeff, unsigned int width, unsigned int height)
{
	const unsigned int posx = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int posy = blockIdx.y*blockDim.y + threadIdx.y;

	if(posx >= width || posy >= height) return;

	d_outShading[posy*width+posx] = 0;

	if(posx > 0 && posx < width-1 && posy > 0 && posy < height-1)
	{
		float4 n = d_normals[posy*width+posx];

		if(n.x != MINF)
		{
			n.x = -n.x; // Change handedness
			n.z = -n.z;

			float albedo = d_albedoEstimates[(unsigned int)(d_clusterIDs[posy*width+posx]+0.5f)];
			float shadingval = albedo*evaluateLightingModel(d_litcoeff, n);

			if(shadingval<0.0f) shadingval = 0.0f;
			if(shadingval>1.0f) shadingval = 1.0f;

			d_outShading[posy*width+posx] = shadingval;
		}
	}
}

extern "C" void computeShadingValue(float* d_outShading, float* d_indepth, float4* d_normals, float* d_clusterIDs, float* d_albedoEstimates, float4x4 &Intrinsic, float* d_lighting, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeShadingValueDevice<<<gridSize, blockSize>>>(d_outShading, d_indepth, d_normals, d_clusterIDs, d_albedoEstimates, Intrinsic, d_lighting, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple Segmentation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeSimpleSegmentationDevice(float* d_output, float* d_input, float depthThres, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	const float inputDepth = d_input[y*width+x];
	if(inputDepth != MINF && inputDepth < depthThres) d_output[y*width+x] = inputDepth;
	else											  d_output[y*width+x] = MINF;
}

extern "C" void computeSimpleSegmentation(float* d_output, float* d_input, float depthThres, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeSimpleSegmentationDevice<<<gridSize, blockSize>>>(d_output, d_input, depthThres, width, height);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Edge Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeMaskEdgeMapFloat4Device(unsigned char* d_output, float4* d_input, float* d_indepth, float threshold, unsigned int width, unsigned int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = 1;
	d_output[width*height+y*width+x] = 1;

	const float thre = threshold *threshold *3.0f;
	if(x > 0 && y > 0 && x < width-1 && y < height-1)
	{	
		if(d_indepth[y*width+x] == MINF)
		{
			d_output[y*width+x] = 0;
			d_output[y*width+x-1] = 0;
			d_output[width*height+y*width+x] = 0;
			d_output[width*height+(y-1)*width+x] = 0;

			return;
		}

		const float4& p0 = d_input[(y+0)*width+(x+0)];
		const float4& p1 = d_input[(y+0)*width+(x+1)];
		const float4& p2 = d_input[(y+1)*width+(x+0)];

		float dU = sqrt(((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y) * (p1.y-p0.y) + (p1.z-p0.z)*(p1.z-p0.z))/3.0f);
		float dV = sqrt(((p2.x-p0.x)*(p2.x-p0.x) + (p2.y-p0.y) * (p2.y-p0.y) + (p2.z-p0.z)*(p2.z-p0.z))/3.0f);

		//float dgradx = abs(d_indepth[y*width+x-1] + d_indepth[y*width+x+1] - 2.0f * d_indepth[y*width+x]);
		//float dgrady = abs(d_indepth[y*width+x-width] + d_indepth[y*width+x+width] - 2.0f * d_indepth[y*width+x]);


		if(dU > thre ) d_output[y*width+x] = 0;
		if(dV > thre ) d_output[width*height+y*width+x] = 0;

		//remove depth discontinuities
		const int r = 1;
		const float thres = 0.01f;

		const float pCC = d_indepth[y*width+x];
		for(int i = -r; i<=r; i++)
		{
			for(int j = -r; j<=r; j++)
			{
				int currentX = x+j;
				int currentY = y+i;

				if(currentX >= 0 && currentX < width && currentY >= 0 && currentY < height)
				{
					float d = d_indepth[currentY*width+currentX];

					if(d != MINF && abs(pCC-d) > thres)
					{
						d_output[y*width+x] = 0;
						d_output[width*height+y*width+x] = 0;
						return;
					}
				}
			}
		}
	}
}

extern "C" void computeMaskEdgeMapFloat4(unsigned char* d_output, float4* d_input, float* d_indepth, float threshold, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeMaskEdgeMapFloat4Device<<<gridSize, blockSize>>>(d_output, d_input, d_indepth, threshold,width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Clear Decission Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void clearDecissionArrayPatchDepthMaskDevice(int* d_output, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight) d_output[y*inputWidth+x] = 0;
}

extern "C" void clearDecissionArrayPatchDepthMask(int* d_output, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	clearDecissionArrayPatchDepthMaskDevice<<<gridSize, blockSize>>>(d_output, inputWidth, inputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Decission Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeDecissionArrayPatchDepthMaskDevice(int* d_output, float* d_input, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight)
	{
		const int patchId_x = x/patchSize;
		const int patchId_y = y/patchSize;
		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;

		const float d = d_input[y*inputWidth+x];
		if(d != MINF) atomicMax(&d_output[patchId_y*nPatchesWidth+patchId_x], 1);
	}
}

extern "C" void computeDecissionArrayPatchDepthMask(int* d_output, float* d_input, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeDecissionArrayPatchDepthMaskDevice<<<gridSize, blockSize>>>(d_output, d_input, patchSize, inputWidth, inputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute Remapping Array for Patch Depth Mask
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void computeRemappingArrayPatchDepthMaskDevice(int* d_output, float* d_input, int* d_prefixSum, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < inputWidth && y >= 0 && y < inputHeight)
	{
		const int patchId_x = x/patchSize;
		const int patchId_y = y/patchSize;

		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;

		const float d = d_input[y*inputWidth+x];
		if(d != MINF) d_output[d_prefixSum[patchId_y*nPatchesWidth+patchId_x]-1] = patchId_y*nPatchesWidth+patchId_x;
	}
}

extern "C" void computeRemappingArrayPatchDepthMask(int* d_output, float* d_input, int* d_prefixSum, unsigned int patchSize, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((inputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (inputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeRemappingArrayPatchDepthMaskDevice<<<gridSize, blockSize>>>(d_output, d_input, d_prefixSum, patchSize, inputWidth, inputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Debug Patch Remap Array
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void DebugPatchRemapArrayDevice(float* d_mask, int* d_remapArray, unsigned int patchSize, unsigned int numElements, unsigned int inputWidth, unsigned int inputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x < numElements)
	{
		int patchID = d_remapArray[x];

		const int nPatchesWidth = (inputWidth+patchSize-1)/patchSize;
		const int patchId_x = patchID%nPatchesWidth;
		const int patchId_y = patchID/nPatchesWidth;

		for(unsigned int i = 0; i<patchSize; i++)
		{
			for(unsigned int j = 0; j<patchSize; j++)
			{
				const int pixel_x = patchId_x*patchSize;
				const int pixel_y = patchId_y*patchSize;

				d_mask[(pixel_y+i)*inputWidth+(pixel_x+j)] = 3.0f;
			}
		}
	}
}

extern "C" void DebugPatchRemapArray(float* d_mask, int* d_remapArray, unsigned int patchSize, unsigned int numElements, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((numElements + T_PER_BLOCK*T_PER_BLOCK - 1)/(T_PER_BLOCK*T_PER_BLOCK));
	const dim3 blockSize(T_PER_BLOCK*T_PER_BLOCK);

	DebugPatchRemapArrayDevice<<<gridSize, blockSize>>>(d_mask, d_remapArray, patchSize, numElements, inputWidth, inputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float bilinearInterpolationFloat(float x, float y, float* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float v00 = d_input[p00.y*imageWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float v10 = d_input[p10.y*imageWidth + p10.x]; if(v10 != MINF) { s0 +=		 alpha *v10; w0 +=		 alpha ; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float v01 = d_input[p01.y*imageWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float v11 = d_input[p11.y*imageWidth + p11.x]; if(v11 != MINF) { s1 +=		 alpha *v11; w1 +=		 alpha ;} }

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ss = 0.0f; float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return MINF;
}

__global__ void resampleFloatMapDevice(float* d_colorMapResampledFloat, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < outputWidth && y < outputHeight)
	{
		const float scaleWidth  = (float)(inputWidth-1) /(float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1)/(float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth +0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight+0.5f);

		if(xInput < inputWidth && yInput < inputHeight)
		{
			d_colorMapResampledFloat[y*outputWidth+x] = bilinearInterpolationFloat(x*scaleWidth, y*scaleHeight, d_colorMapFloat, inputWidth, inputHeight);
		}
	}
}

extern "C" void resampleFloatMap(float* d_colorMapResampledFloat, unsigned int outputWidth, unsigned int outputHeight, float* d_colorMapFloat, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloatMapDevice<<<gridSize, blockSize>>>(d_colorMapResampledFloat, d_colorMapFloat, inputWidth, inputHeight, outputWidth, outputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Float4 Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 bilinearInterpolationFloat4(float x, float y, float4* d_input, unsigned int imageWidth, unsigned int imageHeight)
{
	const int2 p00 = make_int2(floor(x), floor(y));
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	//const float INVALID = 0.0f;
	const float INVALID = MINF;

	float4 s0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w0 = 0.0f;
	if(p00.x < imageWidth && p00.y < imageHeight) { float4 v00 = d_input[p00.y*imageWidth + p00.x]; if(v00.x != INVALID && v00.y != INVALID && v00.z != INVALID) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < imageWidth && p10.y < imageHeight) { float4 v10 = d_input[p10.y*imageWidth + p10.x]; if(v10.x != INVALID && v10.y != INVALID && v10.z != INVALID) { s0 +=		alpha *v10; w0 +=		alpha ; } }

	float4 s1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float w1 = 0.0f;
	if(p01.x < imageWidth && p01.y < imageHeight) { float4 v01 = d_input[p01.y*imageWidth + p01.x]; if(v01.x != INVALID && v01.y != INVALID && v01.z != INVALID) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < imageWidth && p11.y < imageHeight) { float4 v11 = d_input[p11.y*imageWidth + p11.x]; if(v11.x != INVALID && v11.y != INVALID && v11.z != INVALID) { s1 +=		alpha *v11; w1 +=		alpha ;} }

	const float4 p0 = s0/w0;
	const float4 p1 = s1/w1;

	float4 ss = make_float4(0.0f, 0.0f, 0.0f, 0.0f); float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return make_float4(MINF, MINF, MINF, MINF);
}

__global__ void resampleFloat4MapDevice(float4* d_colorMapResampledFloat4, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x < outputWidth && y < outputHeight)
	{
		const float scaleWidth  = (float)(inputWidth-1) /(float)(outputWidth-1);
		const float scaleHeight = (float)(inputHeight-1)/(float)(outputHeight-1);

		const unsigned int xInput = (unsigned int)(x*scaleWidth +0.5f);
		const unsigned int yInput = (unsigned int)(y*scaleHeight+0.5f);

		if(xInput < inputWidth && yInput < inputHeight)
		{
			d_colorMapResampledFloat4[y*outputWidth+x] = bilinearInterpolationFloat4(x*scaleWidth, y*scaleHeight, d_colorMapFloat4, inputWidth, inputHeight);
		}
	}
}

extern "C" void resampleFloat4Map(float4* d_colorMapResampledFloat4, unsigned int outputWidth, unsigned int outputHeight, float4* d_colorMapFloat4, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	resampleFloat4MapDevice<<<gridSize, blockSize>>>(d_colorMapResampledFloat4, d_colorMapFloat4, inputWidth, inputHeight, outputWidth, outputHeight);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Resample Unsigned Char Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void downsampleUnsignedCharMapDevice(unsigned char* d_MapResampled, unsigned char* d_Map, unsigned int inputWidth, unsigned int inputHeight, unsigned int outputWidth, unsigned int outputHeight, unsigned int layerOffsetInput, unsigned int layerOffsetOutput)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= outputWidth || y >= outputHeight) return;

	unsigned char res = 0;

	const unsigned int inputX = 2*x;
	const unsigned int inputY = 2*y;

	if((inputY+0) < inputHeight && (inputX+0) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+0)*inputWidth + (inputX+0))];
	if((inputY+0) < inputHeight && (inputX+1) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+0)*inputWidth + (inputX+1))];
	if((inputY+1) < inputHeight && (inputX+0) < inputWidth)	res += d_Map[layerOffsetInput + ((inputY+1)*inputWidth + (inputX+0))];
	if((inputY+1) < inputHeight && (inputX+1) < inputWidth) res += d_Map[layerOffsetInput + ((inputY+1)*inputWidth + (inputX+1))];

	if(res == 4) d_MapResampled[layerOffsetOutput+(y*outputWidth+x)] = 1;
	else		 d_MapResampled[layerOffsetOutput+(y*outputWidth+x)] = 0;
}

extern "C" void downsampleUnsignedCharMap(unsigned char* d_MapResampled, unsigned int outputWidth, unsigned int outputHeight, unsigned char* d_Map, unsigned int inputWidth, unsigned int inputHeight)
{
	const dim3 gridSize((outputWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (outputHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	downsampleUnsignedCharMapDevice<<<gridSize, blockSize>>>(d_MapResampled, d_Map, inputWidth, inputHeight, outputWidth, outputHeight, 0, 0);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	downsampleUnsignedCharMapDevice<<<gridSize, blockSize>>>(d_MapResampled, d_Map, inputWidth, inputHeight, outputWidth, outputHeight, inputWidth*inputHeight, outputWidth*outputHeight);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Convert Edge Mask to Float Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void convertEdgeMaskToFloatDevice(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= width || y >= height) return;

	d_output[y*width+x] = min(d_input[y*width+x], d_input[width*height+y*width+x]);
}

extern "C" void convertEdgeMaskToFloat(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	convertEdgeMaskToFloatDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dilate Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void dilateDepthMapDevice(float* d_output, float* d_input, float* d_inputOrig, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float sum = 0.0f;
		float count = 0.0f;
		float oldDepth = d_inputOrig[y*width+x];
		if(oldDepth != MINF && oldDepth != 0)
		{
			for(int i = -structureSize; i<=structureSize; i++)
			{
				for(int j = -structureSize; j<=structureSize; j++)
				{
					if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
					{
						const float d = d_input[(y+i)*width+(x+j)];

						if(d != MINF && d != 0.0f && fabs(d-oldDepth) < 0.05f)
						{
							sum += d;
							count += 1.0f;
						}
					}
				}
			}
		}

		if(count > ((2*structureSize+1)*(2*structureSize+1))/36) d_output[y*width+x] = 1.0f;
		else			 d_output[y*width+x] = MINF;
	}
}

extern "C" void dilateDepthMapMask(float* d_output, float* d_input, float* d_inputOrig, int structureSize, int width, int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	dilateDepthMapDevice<<<gridSize, blockSize>>>(d_output, d_input, d_inputOrig, structureSize, width, height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Mean Filter Depth Map
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void removeDevMeanMapMaskDevice(float* d_output, float* d_input, int structureSize, int width, int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	d_output[y*width+x] = d_input[y*width+x];

	if(x >= 0 && x < width && y >= 0 && y < height)
	{
		float oldDepth = d_input[y*width+x];

		float mean = 0.0f;
		float meanSquared = 0.0f;
		float count = 0.0f;
		for(int i = -structureSize; i<=structureSize; i++)
		{
			for(int j = -structureSize; j<=structureSize; j++)
			{
				if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height)
				{
					float depth = d_input[(y+i)*width+(x+j)];
					if(depth == MINF)
					{
						depth = 8.0f;
					}

					if(depth > 0.0f)
					{
						mean		+= depth;
						meanSquared += depth*depth;
						count		+= 1.0f;
					}
				}
			}
		}

		mean/=count;
		meanSquared/=count;

		float stdDev = sqrt(meanSquared-mean*mean);

		if(fabs(oldDepth-mean) > 0.5f*stdDev)// || stdDev> 0.005f)
		{
			d_output[y*width+x] = MINF;
		}
	}
}

extern "C" void removeDevMeanMapMask(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height)
{
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	removeDevMeanMapMaskDevice<<<gridSize, blockSize>>>(d_output, d_input, structureSize, width, height);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}




// Nearest neighbour
inline __device__ bool getValueNearestNeighbourNoCheck(const float2& p, const float4* inputMap, unsigned int imageWidth, unsigned int imageHeight, float4* outValue)
{
	const int u = (int)(p.x + 0.5f);
	const int v = (int)(p.y + 0.5f);

	if(u < 0 || u > imageWidth || v < 0 || v > imageHeight) return false;

	*outValue = inputMap[v*imageWidth + u];

	return true;
}

inline __device__ bool getValueNearestNeighbour(const float2& p, const float4* inputMap, unsigned int imageWidth, unsigned int imageHeight, float4* outValue)
{
	bool valid = getValueNearestNeighbourNoCheck(p, inputMap, imageWidth, imageHeight, outValue);
	return valid && (outValue->x != MINF && outValue->y != MINF && outValue->z != MINF);
}

// Nearest neighbour
inline __device__ bool getValueNearestNeighbourFloatNoCheck(const float2& p, const float* inputMap, unsigned int imageWidth, unsigned int imageHeight, float* outValue)
{
	const int u = (int)(p.x + 0.5f);
	const int v = (int)(p.y + 0.5f);

	if(u < 0 || u > imageWidth || v < 0 || v > imageHeight) return false;

	*outValue = inputMap[v*imageWidth + u];

	return true;
}

inline __device__ bool getValueNearestNeighbourFloat(const float2& p, const float* inputMap, unsigned int imageWidth, unsigned int imageHeight, float* outValue)
{
	bool valid = getValueNearestNeighbourFloatNoCheck(p, inputMap, imageWidth, imageHeight, outValue);
	return valid && (*outValue != MINF);
}

/////////////////////////////////////////////
// Compute derivatives in camera space
/////////////////////////////////////////////

__global__ void computeDerivativesCameraSpaceDevice(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight, float4* d_positionsDU, float4* d_positionsDV)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int index = y*imageWidth+x;

	if(x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
	{
		d_positionsDU[index] = make_float4(MINF, MINF, MINF, MINF);
		d_positionsDV[index] = make_float4(MINF, MINF, MINF, MINF);
	
		if(x > 0 && x < imageWidth - 1 && y > 0 && y < imageHeight - 1)
		{
			float4 pos00; bool valid00 = getValueNearestNeighbour(make_float2(x-1, y-1), d_positions, imageWidth, imageHeight, &pos00); if(!valid00) return;
			float4 pos01; bool valid01 = getValueNearestNeighbour(make_float2(x-1, y-0), d_positions, imageWidth, imageHeight, &pos01); if(!valid01) return;
			float4 pos02; bool valid02 = getValueNearestNeighbour(make_float2(x-1, y+1), d_positions, imageWidth, imageHeight, &pos02); if(!valid02) return;
			
			float4 pos10; bool valid10 = getValueNearestNeighbour(make_float2(x-0, y-1), d_positions, imageWidth, imageHeight, &pos10); if(!valid10) return;
			float4 pos11; bool valid11 = getValueNearestNeighbour(make_float2(x-0, y-0), d_positions, imageWidth, imageHeight, &pos11); if(!valid11) return;
			float4 pos12; bool valid12 = getValueNearestNeighbour(make_float2(x-0, y+1), d_positions, imageWidth, imageHeight, &pos12); if(!valid12) return;

			float4 pos20; bool valid20 = getValueNearestNeighbour(make_float2(x+1, y-1), d_positions, imageWidth, imageHeight, &pos20); if(!valid20) return;
			float4 pos21; bool valid21 = getValueNearestNeighbour(make_float2(x+1, y-0), d_positions, imageWidth, imageHeight, &pos21); if(!valid21) return;
			float4 pos22; bool valid22 = getValueNearestNeighbour(make_float2(x+1, y+1), d_positions, imageWidth, imageHeight, &pos22); if(!valid22) return;

			float4 resU = (-1.0f)*pos00 + (1.0f)*pos20 +
						  (-2.0f)*pos01 + (2.0f)*pos21 +
						  (-1.0f)*pos02 + (1.0f)*pos22;
			resU /= 8.0f;
			
			float4 resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 + 
						  ( 1.0f)*pos02 + ( 2.0f)*pos12 + ( 1.0f)*pos22;
			resV /= 8.0f;

			//if(mat3x1(make_float3(resU)).norm1D() > 0.02f) return;
			//if(mat3x1(make_float3(resV)).norm1D() > 0.02f) return;

			d_positionsDU[index] = resU;
			d_positionsDV[index] = resV;
		}
	}
}

extern "C" void computeDerivativesCameraSpace(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight, float4* d_positionsDU, float4* d_positionsDV)
{

	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeDerivativesCameraSpaceDevice<<<gridSize, blockSize>>>(d_positions, imageWidth, imageHeight, d_positionsDU, d_positionsDV);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}


/////////////////////////////////////////////
// Compute Intensity and Derivatives
/////////////////////////////////////////////

__global__ void computeIntensityAndDerivativesDevice(float* d_intensity, unsigned int imageWidth, unsigned int imageHeight, float4* d_intensityAndDerivatives)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int index = y*imageWidth+x;

	if(x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
	{
		d_intensityAndDerivatives[index] = make_float4(MINF, MINF, MINF, MINF);
			
		if(x > 0 && x < imageWidth - 1 && y > 0 && y < imageHeight - 1)
		{
			float pos00; bool valid00 = getValueNearestNeighbourFloat(make_float2(x-1, y-1), d_intensity, imageWidth, imageHeight, &pos00); if(!valid00) return;
			float pos01; bool valid01 = getValueNearestNeighbourFloat(make_float2(x-1, y-0), d_intensity, imageWidth, imageHeight, &pos01); if(!valid01) return;
			float pos02; bool valid02 = getValueNearestNeighbourFloat(make_float2(x-1, y+1), d_intensity, imageWidth, imageHeight, &pos02); if(!valid02) return;
			
			float pos10; bool valid10 = getValueNearestNeighbourFloat(make_float2(x-0, y-1), d_intensity, imageWidth, imageHeight, &pos10); if(!valid10) return;
			float pos11; bool valid11 = getValueNearestNeighbourFloat(make_float2(x-0, y-0), d_intensity, imageWidth, imageHeight, &pos11); if(!valid11) return;
			float pos12; bool valid12 = getValueNearestNeighbourFloat(make_float2(x-0, y+1), d_intensity, imageWidth, imageHeight, &pos12); if(!valid12) return;

			float pos20; bool valid20 = getValueNearestNeighbourFloat(make_float2(x+1, y-1), d_intensity, imageWidth, imageHeight, &pos20); if(!valid20) return;
			float pos21; bool valid21 = getValueNearestNeighbourFloat(make_float2(x+1, y-0), d_intensity, imageWidth, imageHeight, &pos21); if(!valid21) return;
			float pos22; bool valid22 = getValueNearestNeighbourFloat(make_float2(x+1, y+1), d_intensity, imageWidth, imageHeight, &pos22); if(!valid22) return;

			float resU = (-1.0f)*pos00 + (1.0f)*pos20 +
						  (-2.0f)*pos01 + (2.0f)*pos21 +
						  (-1.0f)*pos02 + (1.0f)*pos22;
			resU /= 8.0f;
			
			float resV = (-1.0f)*pos00 + (-2.0f)*pos10 + (-1.0f)*pos20 + 
						  ( 1.0f)*pos02 + ( 2.0f)*pos12 + ( 1.0f)*pos22;
			resV /= 8.0f;

			d_intensityAndDerivatives[index] = make_float4(pos11, resU, resV, 1.0f);
		}
	}
}

extern "C" void computeIntensityAndDerivatives(float* d_intensity, unsigned int imageWidth, unsigned int imageHeight, float4* d_intensityAndDerivatives)
{
	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeIntensityAndDerivativesDevice<<<gridSize, blockSize>>>(d_intensity, imageWidth, imageHeight, d_intensityAndDerivatives);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}


/////////////////////////////////////////////
// Compute grdient intensity magnitude
/////////////////////////////////////////////

__global__ void computeGradientIntensityMagnitudeDevice(float4* d_inputDU, float4* d_inputDV, unsigned int imageWidth, unsigned int imageHeight, float4* d_ouput)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int index = y*imageWidth+x;

	d_ouput[index] = make_float4(MINF, MINF, MINF, MINF);

	float4 DU = d_inputDU[index];
	float4 DV = d_inputDV[index];

	if(DU.x != MINF && DV.x != MINF)
	{
		float m = sqrtf(DU.x*DU.x+DV.x*DV.x);

		if(m > 0.005f)
		{
			d_ouput[index] = make_float4(m, m, m, 1.0f);
		}
	}
}

extern "C" void computeGradientIntensityMagnitude(float4* d_inputDU, float4* d_inputDV, unsigned int imageWidth, unsigned int imageHeight, float4* d_ouput)
{
	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	computeGradientIntensityMagnitudeDevice<<<gridSize, blockSize>>>(d_inputDU, d_inputDV, imageWidth, imageHeight, d_ouput);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

/////////////////////////////////////////////
// Transform
/////////////////////////////////////////////

__global__ void transformCameraSpaceMapDevice(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight,  float4* d_output)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int index = y*imageWidth+x;

	if(x >= 0 && x < imageWidth && y >= 0 && y < imageHeight)
	{
		d_output[index] = d_positions[index];

		if(d_positions[index].x != MINF && d_positions[index].y != MINF && d_positions[index].z != MINF)
		{
			d_output[index] =  d_positions[index]+make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		}		
	}
}

extern "C" void transformCameraSpaceMap(float4* d_positions, unsigned int imageWidth, unsigned int imageHeight,  float4* d_output)
{
	const dim3 gridSize((imageWidth + T_PER_BLOCK - 1)/T_PER_BLOCK, (imageHeight + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	transformCameraSpaceMapDevice<<<gridSize, blockSize>>>(d_positions, imageWidth, imageHeight, d_output);
	#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
	#endif
}

















////////////////////////////////////
// Depth to HSV map conversion /////
////////////////////////////////////

__device__ float3 convertHSVToRGB(const float3& hsv) {
	float H = hsv.x;
	float S = hsv.y;
	float V = hsv.z;

	float hd = H/60.0f;
	unsigned int h = (unsigned int)hd;
	float f = hd-h;

	float p = V*(1.0f-S);
	float q = V*(1.0f-S*f);
	float t = V*(1.0f-S*(1.0f-f));

	if(h == 0 || h == 6)
	{
		return make_float3(V, t, p);
	}
	else if(h == 1)
	{
		return make_float3(q, V, p);
	}
	else if(h == 2)
	{
		return make_float3(p, V, t);
	}
	else if(h == 3)
	{
		return make_float3(p, q, V);
	}
	else if(h == 4)
	{
		return make_float3(t, p, V);
	}
	else
	{
		return make_float3(V, p, q);
	}
}


__device__ float3 convertDepthToRGB(float depth, float depthMin, float depthMax) {
	float depthZeroOne = (depth - depthMin)/(depthMax - depthMin);
	float x = 1.0f-depthZeroOne;
	if (x < 0.0f)	x = 0.0f;
	if (x > 1.0f)	x = 1.0f;
	//return convertHSVToRGB(make_float3(240.0f*x, 1.0f, 0.5f));
	x = 360.0f*x - 120.0f;
	if (x < 0.0f) x += 359.0f;
	return convertHSVToRGB(make_float3(x, 1.0f, 0.5f));
}

__global__ void depthToHSVDevice(float4* d_output, const float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if (x >= 0 && x < width && y >= 0 && y < height) {
		
		float depth = d_input[y*width + x];
		if (depth != MINF && depth != 0.0f && depth >= minDepth && depth <= maxDepth) {
			float3 c = convertDepthToRGB(depth, minDepth, maxDepth);
			d_output[y*width + x] = make_float4(c, 1.0f);
		} else {
			d_output[y*width + x] = make_float4(0.0f);
		}
	}
}

extern "C" void depthToHSV(float4* d_output, const float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth) {
	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	depthToHSVDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height, minDepth, maxDepth);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}





#endif // _CAMERA_UTIL_
