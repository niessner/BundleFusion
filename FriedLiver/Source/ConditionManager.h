
#pragma once

#include <condition_variable>
#include "GlobalAppState.h"

class ConditionManager {
public:
	enum THREAD_NAME {
		Recon,
		Bundling,
		Num_Threads
	};
	ConditionManager() {}
	~ConditionManager() {}

	static void init() {
		s_lockImageManagerFrameReady.resize(Num_Threads);
		s_lockBundlerProcessedInput.resize(Num_Threads);
		for (unsigned int i = 0; i < Num_Threads; i++) {
			s_lockImageManagerFrameReady[i] = std::unique_lock<std::mutex>(s_mutexImageManagerHasFrameReady, std::defer_lock);
			s_lockBundlerProcessedInput[i] = std::unique_lock<std::mutex>(s_mutexBundlerProcessedInput, std::defer_lock);
		}
	}

	static void release(THREAD_NAME type) {
		if (s_lockImageManagerFrameReady[type].owns_lock()) {
			s_lockImageManagerFrameReady[type].unlock();
		}
		s_cvFrameReadyCheck.notify_one(); // possibly unlocked but not notified

		if (s_lockBundlerProcessedInput[type].owns_lock()) {
			s_lockBundlerProcessedInput[type].unlock();
		}
		s_cvBundlerProcessedCheck.notify_one(); // possibly unlocked but not notified
	}

	static void DEBUGRELEASE() {
		if (s_lockImageManagerFrameReady[0].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: RECON STILL OWNS LOCK" << std::endl;
		if (s_lockImageManagerFrameReady[1].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: BUNDLE STILL OWNS LOCK" << std::endl;
		if (s_lockBundlerProcessedInput[0].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: RECON STILL OWNS LOCK" << std::endl;
		if (s_lockBundlerProcessedInput[1].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: BUNDLE STILL OWNS LOCK" << std::endl;
	}

#ifdef RUN_MULTITHREADED
	static void lockImageManagerFrameReady(THREAD_NAME type) {
		s_lockImageManagerFrameReady[type].lock();
	}
	static void waitImageManagerFrameReady(THREAD_NAME type) {
		s_cvFrameReadyCheck.wait(s_lockImageManagerFrameReady[type]);
	}
	static void unlockAndNotifyImageManagerFrameReady(THREAD_NAME type) {
		s_lockImageManagerFrameReady[type].unlock();
		s_cvFrameReadyCheck.notify_one();
	}

	static void lockBundlerProcessedInput(THREAD_NAME type) {
		s_lockBundlerProcessedInput[type].lock();
	}
	static void waitBundlerProcessedInput(THREAD_NAME type) {
		s_cvBundlerProcessedCheck.wait(s_lockBundlerProcessedInput[type]);
	}
	static void unlockAndNotifyBundlerProcessedInput(THREAD_NAME type) {
		s_lockBundlerProcessedInput[type].unlock();
		s_cvBundlerProcessedCheck.notify_one();
	}
#else 
	static void lockImageManagerFrameReady(THREAD_NAME type) {}
	static void waitImageManagerFrameReady(THREAD_NAME type) {}
	static void unlockAndNotifyImageManagerFrameReady(THREAD_NAME type) {}

	static void lockBundlerProcessedInput(THREAD_NAME type) {}
	static void waitBundlerProcessedInput(THREAD_NAME type) {}
	static void unlockAndNotifyBundlerProcessedInput(THREAD_NAME type) {}
#endif

	static void setExit() {
		s_exit = true;
	}
	static bool shouldExit() {
		return s_exit;
	}
private:
	static std::mutex s_mutexImageManagerHasFrameReady;
	static std::condition_variable s_cvFrameReadyCheck;
	static std::vector<std::unique_lock<std::mutex>> s_lockImageManagerFrameReady;

	static std::mutex s_mutexBundlerProcessedInput;
	static std::condition_variable s_cvBundlerProcessedCheck;
	static std::vector<std::unique_lock<std::mutex>> s_lockBundlerProcessedInput;

	static bool s_exit;
};