
#include "stdafx.h"
#include "ConditionManager.h"


std::mutex ConditionManager::s_mutexImageManagerHasFrameReady;
std::condition_variable ConditionManager::s_cvFrameReadyCheck;
std::vector<std::unique_lock<std::mutex>> ConditionManager::s_lockImageManagerFrameReady;

std::mutex ConditionManager::s_mutexBundlerProcessedInput;
std::condition_variable ConditionManager::s_cvBundlerProcessedCheck;
std::vector<std::unique_lock<std::mutex>> ConditionManager::s_lockBundlerProcessedInput;

bool ConditionManager::s_exit = false;