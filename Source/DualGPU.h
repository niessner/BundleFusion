
#pragma once

#define DUAL_GPU

#include "mLibCuda.h"

class DualGPU {
public:
	class GPU {
	public:

		GPU(int deviceIdx) : m_deviceIndex(deviceIdx) {}

		void set() const {
			MLIB_CUDA_SAFE_CALL(cudaSetDevice(m_deviceIndex));
		}

		void printStats() const {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, m_deviceIndex);
			printf("Device Number: %d\n", m_deviceIndex);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		}

		size_t getFreeMemoryMB() const {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);	
			mem_free /= (1024 * 1024);	
			mem_total /= (1024 * 1024);
			return mem_free;

		}
		size_t getTotalMemoryMB() const {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			return mem_total;
		}
		size_t getUsedMemoryMB() const {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			return mem_total - mem_free;
		}
		std::string getName() const {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, m_deviceIndex);
			return std::string(prop.name);
		}

		unsigned int getDeviceIdx() const {
			return m_deviceIndex;
		}
	private:
		int m_deviceIndex;
	};




	void printList() {
		std::cout << "GPUs found:\n";
		for (auto& g : m_gpus) {
			std::cout << "\t" << g.getDeviceIdx() << ": " << g.getName() << std::endl;
		}
	}

	static DualGPU& get() {
		static DualGPU dualGPU;
		if (!dualGPU.m_bIsInit) dualGPU.init();
		return dualGPU;
	}

	void setDevice(unsigned int deviceIdx) const {
		m_gpus[deviceIdx].set();
	}

	const GPU& getDevice(unsigned int deviceIdx) const {
		return m_gpus[deviceIdx];
	}

	static unsigned int getActiveDevice() {
		int deviceIdx;
		MLIB_CUDA_SAFE_CALL(cudaGetDevice(&deviceIdx));
		return deviceIdx;
	}

	enum {
		DEVICE_RECONSTRUCTION = 0,
		DEVICE_BUNDLING = 1
	};

private:
	DualGPU() {
		m_bIsInit = false;
	}

	void init()
	{
		int numDevices;
		cudaGetDeviceCount(&numDevices);
		assert(numDevices > 0);
		for (int i = 0; i < numDevices; i++) {
			m_gpus.push_back(GPU(i));
		}
		printList();
		m_bIsInit = true;
	}

	bool m_bIsInit;
	std::vector<GPU> m_gpus;
};