
#pragma once

#define DUAL_GPU

#include "mLibCuda.h"

class DualGPU {
public:
	class GPU {
	public:

		GPU(int deviceIdx, bool isEmulated = false) : m_deviceIndex(deviceIdx), m_bIsEmulated(isEmulated) {}

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

		static size_t getFreeMemoryMB() {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			return mem_free;

		}
		static size_t getTotalMemoryMB() {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			return mem_total;
		}
		static size_t getUsedMemoryMB() {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			return mem_total - mem_free;
		}
		static void printMemoryStats() {
			size_t mem_free, mem_total;
			cudaMemGetInfo(&mem_free, &mem_total);
			mem_free /= (1024 * 1024);
			mem_total /= (1024 * 1024);
			printf("total memory: %d\n", mem_total);
			printf("free memory : %d\n", mem_free);
			printf("used memory : %d\n", mem_total - mem_free);
		}
		std::string getName() const {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, m_deviceIndex);
			std::string res(prop.name);
			if (m_bIsEmulated) {
				res = res + " (emulated) ";
			}
			return res;
		}

		unsigned int getDeviceIdx() const {
			return m_deviceIndex;
		}
	private:
		int m_deviceIndex;
		bool m_bIsEmulated;
	};




	void printList() {
		std::cout << "GPUs found:\n";
		for (auto& g : m_gpus) {
			std::cout << "\tdeviceId: " << g.getDeviceIdx() << ": " << g.getName() << std::endl;
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

	void init(unsigned int minDevices = 2, unsigned int maxPhysicalDevices = 2)
	{
		unsigned int numDevices;
		cudaGetDeviceCount((int*)&numDevices);
		assert(numDevices > 0);
		numDevices = std::min(maxPhysicalDevices, numDevices);	//if we want to artificially reduces the number of GPUs
		for (unsigned int i = 0; i < numDevices; i++) {
			m_gpus.push_back(GPU(i));
		}
		assert(m_gpus.size() > 0);
		const GPU& last = m_gpus.back();
		for (unsigned int i = numDevices; i < minDevices; i++) {	//fill up with emulated devices
			m_gpus.push_back(GPU(last.getDeviceIdx(), true));
		}
		printList();
		m_bIsInit = true;
	}

	bool m_bIsInit;
	std::vector<GPU> m_gpus;
};