#pragma once

#include <cutil_inline.h>
#include <cutil_math.h>

#include "VoxelUtilHashSDF.h"
#include "DepthCameraUtil.h"
#include "CUDAScan.h"


extern "C" void computeHistogramCUDA(unsigned int* d_data, const HashDataStruct& hashData, const HashParams& hashParams);
extern "C" void resetHistrogramCUDA(unsigned int* d_data, unsigned int numValues);

class CUDAHistrogramHashSDF {
public:
	CUDAHistrogramHashSDF(const HashParams& hashParams) {
		create(hashParams);
	}
	~CUDAHistrogramHashSDF() {
		destroy();
	}

	void computeHistrogram(const HashDataStruct& hashData, const HashParams& hashParams) {
		resetHistrogramCUDA(d_historgram, hashParams.m_hashBucketSize + 1 + hashParams.m_hashMaxCollisionLinkedListSize + 1);
		computeHistogramCUDA(d_historgram, hashData, hashParams);
		printHistogram(hashParams);
	}
private:
	void create(const HashParams& hashParams) {
		cutilSafeCall(cudaMalloc(&d_historgram, sizeof(unsigned int)*(hashParams.m_hashBucketSize + 1 + hashParams.m_hashMaxCollisionLinkedListSize + 1))); 
	}

	void destroy() {
		cutilSafeCall(cudaFree(d_historgram));
	}

	void printHistogram(const HashParams& hashParams)
	{
		unsigned int* h_data = new unsigned int[hashParams.m_hashBucketSize + 1 + hashParams.m_hashMaxCollisionLinkedListSize + 1];
		cutilSafeCall(cudaMemcpy(h_data, d_historgram, sizeof(unsigned int)*(hashParams.m_hashBucketSize + 1 + hashParams.m_hashMaxCollisionLinkedListSize + 1), cudaMemcpyDeviceToHost));

		std::streamsize oldPrec = std::cout.precision(4);
		std::ios_base::fmtflags oldFlags = std::cout.setf( std::ios::fixed, std:: ios::floatfield );

		unsigned int nTotal = 0;
		unsigned int nElements = 0;
		for (unsigned int i = 0; i < hashParams.m_hashBucketSize+1; i++) {
			nTotal += h_data[i];
			nElements += h_data[i]*i;
		}

		std::cout << nTotal << std::endl;
		std::cout << "Histogram for hash with " << (unsigned int)nElements <<" of " << hashParams.m_hashNumBuckets*hashParams.m_hashBucketSize << " elements:" << std::endl;
		std::cout << "--------------------------------------------------------------" << std::endl;
		unsigned int checkBuckets = 0;
		for (unsigned int i = 0; i < hashParams.m_hashBucketSize+1; i++) {
			float percent = 100.0f*(float)h_data[i]/(float)nTotal;
			std::cout << i << ":\t" << (percent < 10.0f ? " " : "" ) << percent << "%\tabsolute: " << h_data[i] << std::endl;
			checkBuckets += h_data[i];
		}
		std::cout << std::endl;
		unsigned int checkLists = 0;
		for (unsigned int i = hashParams.m_hashBucketSize+1; i < hashParams.m_hashBucketSize+1+hashParams.m_hashMaxCollisionLinkedListSize; i++) {
			float percent = 100.0f*(float)h_data[i]/(float)hashParams.m_hashNumBuckets;
			std::cout << "listLen " << (i - (hashParams.m_hashBucketSize+1)) << ":\t" << (percent < 10.0f ? " " : "" ) << percent << "%\tabsolute: " << h_data[i] << std::endl;
			checkLists += h_data[i];
		}
		std::cout << "--------------------------------------------------------------" << std::endl;
		std::cout << "checkBuckets\t " << checkBuckets << "\t" << ((checkBuckets == hashParams.m_hashNumBuckets) ? "OK" : "FAIL") << std::endl; 
		std::cout << "checkLists\t " << checkLists << "\t" << ((checkBuckets == hashParams.m_hashNumBuckets) ? "OK" : "FAIL") << std::endl;
		std::cout << "--------------------------------------------------------------" << std::endl;

		std::cout.precision(oldPrec);
		std::cout.setf(oldFlags);

		SAFE_DELETE_ARRAY(h_data);
	}
	

	unsigned int* d_historgram;
};