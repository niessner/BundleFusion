
#pragma once

#include <condition_variable>

class ConditionManager {
public:
	enum THREAD_NAME {
		Recon,
		Bundling
	};

	static void init() {
		s_lockImageManagerFrameReady[0] = std::unique_lock<std::mutex>(s_mutexImageManagerHasFrameReady, std::defer_lock);
		s_lockImageManagerFrameReady[1] = std::unique_lock<std::mutex>(s_mutexImageManagerHasFrameReady, std::defer_lock);
	}

	static void release() {
		for (unsigned int i = 0; i < 2; i++) {
			if (s_lockImageManagerFrameReady[i].owns_lock()) s_lockImageManagerFrameReady[i].unlock();
		}
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

	static std::unique_lock<std::mutex> s_lockImageManagerFrameReady[2];
};