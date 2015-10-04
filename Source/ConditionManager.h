
#pragma once

#include <condition_variable>

class ConditionManager {
public:
	enum THREAD_NAME {
		Recon,
		Bundling
	};
	ConditionManager() {}
	~ConditionManager() {}

	static void init() {
		s_lockImageManagerFrameReady.resize(2);
		s_lockImageManagerFrameReady[0] = std::unique_lock<std::mutex>(s_mutexImageManagerHasFrameReady, std::defer_lock);
		s_lockImageManagerFrameReady[1] = std::unique_lock<std::mutex>(s_mutexImageManagerHasFrameReady, std::defer_lock);
	}

	static void release(THREAD_NAME type) {
		if (s_lockImageManagerFrameReady[type].owns_lock()) {
			s_lockImageManagerFrameReady[type].unlock();
		}
		s_frameReadyCheck.notify_one(); // possibly unlocked but not notified
	}

	static void DEBUGRELEASE() {
		if (s_lockImageManagerFrameReady[0].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: RECON STILL OWNS LOCK" << std::endl;
		if (s_lockImageManagerFrameReady[1].owns_lock())
			std::cout << "ERROR: CONDITION MANAGER: BUNDLE STILL OWNS LOCK" << std::endl;
	}

	static void lockImageManagerFrameReady(THREAD_NAME type) {
		s_lockImageManagerFrameReady[type].lock();
	}
	static void waitImageManagerFrameReady(THREAD_NAME type) {
		s_frameReadyCheck.wait(s_lockImageManagerFrameReady[type]);
	}
	static void unlockAndNotifyImageManagerFrameReady(THREAD_NAME type) {
		s_lockImageManagerFrameReady[type].unlock();
		s_frameReadyCheck.notify_one();
	}

private:
	static std::mutex s_mutexImageManagerHasFrameReady;
	static std::condition_variable s_frameReadyCheck;

	static std::vector<std::unique_lock<std::mutex>> s_lockImageManagerFrameReady;
};