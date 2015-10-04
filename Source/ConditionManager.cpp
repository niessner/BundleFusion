
#include "stdafx.h"
#include "ConditionManager.h"


std::mutex ConditionManager::s_mutexImageManagerHasFrameReady;
std::condition_variable ConditionManager::s_frameReadyCheck;

std::unique_lock<std::mutex> ConditionManager::s_lockImageManagerFrameReady[2];