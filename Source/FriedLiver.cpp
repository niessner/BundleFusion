

#include "stdafx.h"

#include "FriedLiver.h"


RGBDSensor* getRGBDSensor()
{
	if (GlobalAppState::get().s_sensorIdx == 0) {
#ifdef KINECT
		static KinectSensor s_kinect;
		return &s_kinect;
#else 
		throw MLIB_EXCEPTION("Requires KINECT V1 SDK and enable KINECT macro");
#endif
	}

	if (GlobalAppState::get().s_sensorIdx == 1)	{
#ifdef OPEN_NI
		static PrimeSenseSensor s_primeSense;
		return &s_primeSense;
#else 
		throw MLIB_EXCEPTION("Requires OpenNI 2 SDK and enable OPEN_NI macro");
#endif
	}
	else if (GlobalAppState::getInstance().s_sensorIdx == 2) {
#ifdef KINECT_ONE
		static KinectOneSensor s_kinectOne;
		return &s_kinectOne;
#else
		throw MLIB_EXCEPTION("Requires Kinect 2.0 SDK and enable KINECT_ONE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 3) {
#ifdef BINARY_DUMP_READER
		static BinaryDumpReader s_binaryDump;
		return &s_binaryDump;
#else 
		throw MLIB_EXCEPTION("Requires BINARY_DUMP_READER macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 4) {
#ifdef NETWORK_SENSOR
		static NetworkSensor s_networkSensor;
		return &s_networkSensor;
#else
		throw MLIB_EXCEPTION("Requires NETWORK_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 5) {
#ifdef INTEL_SENSOR
		static IntelSensor s_intelSensor;
		return &s_intelSensor;
#else 
		throw MLIB_EXCEPTION("Requires INTEL_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 6) {
#ifdef REAL_SENSE
		static RealSenseSensor s_realSenseSensor;
		return &s_realSenseSensor;
#else
		throw MLIB_EXCEPTION("Requires Real Sense SDK and REAL_SENSE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 7) {
#ifdef STRUCTURE_SENSOR
		static StructureSensor s_structureSensor;
		return &s_structureSensor;
#else
		throw MLIB_EXCEPTION("Requires STRUCTURE_SENSOR macro");
#endif
	}

	throw MLIB_EXCEPTION("unkown sensor id " + std::to_string(GlobalAppState::get().s_sensorIdx));

	return NULL;
}

RGBDSensor* g_RGBDSensor = NULL;
CUDAImageManager* g_imageManager = NULL;
Bundler* g_bundler = NULL;



void bundlingOptimization() {
	g_bundler->optimizeLocal(GlobalBundlingState::get().s_numLocalNonLinIterations, GlobalBundlingState::get().s_numLocalLinIterations);
	g_bundler->processGlobal();
	g_bundler->optimizeGlobal(GlobalBundlingState::get().s_numGlobalNonLinIterations, GlobalBundlingState::get().s_numGlobalLinIterations);
}

void bundlingOptimizationThreadFunc() {

	DualGPU::get().setDevice(DualGPU::DEVICE_BUNDLING);

	//for (unsigned int i = 0; i < 10; i++) {
		bundlingOptimization();
	//}
}

void bundlingThreadFunc() {
	assert(g_RGBDSensor && g_imageManager);
	DualGPU::get().setDevice(DualGPU::DEVICE_BUNDLING);
	g_bundler = new Bundler(g_RGBDSensor, g_imageManager);

	std::thread tOpt;

	while (1) {
		// opt
		if (g_RGBDSensor->isReceivingFrames()) {
			if (g_bundler->getCurrProcessedFrame() % 10 == 0) { // stop solve
				if (tOpt.joinable()) {
					tOpt.join();
				}
			}
			if (g_bundler->getCurrProcessedFrame() % 10 == 1) { // start solve
				MLIB_ASSERT(!tOpt.joinable());
				tOpt = std::thread(bundlingOptimizationThreadFunc);
			}
		}
		else {
			if (tOpt.joinable()) {
				tOpt.join();
			}
			tOpt = std::thread(bundlingOptimizationThreadFunc);
		}

		ConditionManager::lockImageManagerFrameReady(ConditionManager::Bundling);
		while (!g_imageManager->hasBundlingFrameRdy()) {
			ConditionManager::waitImageManagerFrameReady(ConditionManager::Bundling); //wait for a new input frame (LOCK IMAGE MANAGER)
		}
		//while (!g_imageManager->hasBundlingFrameRdy()) Sleep(0);	//wait for a new input frame (LOCK IMAGE MANAGER)
		{
			ConditionManager::lockBundlerProcessedInput(ConditionManager::Bundling);
			while (g_bundler->hasProcssedInputFrame()) ConditionManager::waitBundlerProcessedInput(ConditionManager::Bundling);	//wait until depth sensing has confirmed the last one (WAITING THAT DEPTH SENSING RELEASES ITS LOCK)
			//while (g_bundler->hasProcssedInputFrame()) Sleep(0);		//wait until depth sensing has confirmed the last one (WAITING THAT DEPTH SENSING RELEASES ITS LOCK)
			{
				if (g_bundler->getExitBundlingThread()) {
					if (tOpt.joinable()) {
						tOpt.join();
					}
					ConditionManager::release(ConditionManager::Bundling);
					break;
				}
				g_bundler->processInput();						//perform sift and whatever
			}
			g_bundler->setProcessedInputFrame();			//let depth sensing know we have a frame (UNLOCK BUNDLING)
			ConditionManager::unlockAndNotifyBundlerProcessedInput(ConditionManager::Bundling);
		}
		g_imageManager->confirmRdyBundlingFrame();		//here it's processing with a new input frame  (GIVE DEPTH SENSING THE POSSIBLITY TO LOCK IF IT WANTS)
		ConditionManager::unlockAndNotifyImageManagerFrameReady(ConditionManager::Bundling);

		if (g_bundler->getExitBundlingThread()) {
			ConditionManager::release(ConditionManager::Bundling);
			break;
		}
	}
}

int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//_CrtSetBreakAlloc(3727);
#endif 

	try {
		std::string fileNameDescGlobalApp;
		std::string fileNameDescGlobalBundling;
		if (argc == 3) {
			fileNameDescGlobalApp = std::string(argv[1]);
			fileNameDescGlobalBundling = std::string(argv[2]);
		}
		else {
			std::cout << "usage: DepthSensing [fileNameDescGlobalApp] [fileNameDescGlobalTracking]" << std::endl;
			fileNameDescGlobalApp = "zParametersDefault.txt";
			fileNameDescGlobalBundling = "zParametersBundlingDefault.txt";

			//fileNameDescGlobalApp = "zParametersMedium.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingHigh.txt";

			//fileNameDescGlobalApp = "zParametersHigh.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingHigh.txt";

			//fileNameDescGlobalApp = "zParametersKinect.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingKinect.txt";
		}

		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalBundling) << " = " << fileNameDescGlobalBundling << std::endl;
		std::cout << std::endl;

		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);

		//Read the global camera tracking state
		ParameterFile parameterFileGlobalBundling(fileNameDescGlobalBundling);
		GlobalBundlingState::getInstance().readMembers(parameterFileGlobalBundling);

		//!!!DEBUGGING
		{
			TestMatching test;
			//test.loadReferenceTrajectory("trajectory.bin", GlobalBundlingState::get().s_submapSize);
			//test.saveReferenceTrajectory("debug/refTraj.bin");
			test.loadReferenceTrajectory("debug/refTraj.bin");
			test.loadColorImages("debug/colorImages.bin");
			//test.load("debug/matchAll.bin", "debug/global.sift");
			//test.loadColorImagesFromSensor("../data/student/2296_eroded.sensor", GlobalBundlingState::get().s_submapSize);
			//test.loadColorImagesFromSensor("../data/testLost.sensor", GlobalBundlingState::get().s_submapSize);
			//test.saveColorImages("debug/colorImages.bin");
			test.loadIntrinsics("../data/debug45.sensor");
			test.load("", "debug/global.sift");
			test.test();
			test.save("debug/filtered.bin");

			std::cout << "done!" << std::endl;
			getchar();
			return 0;
		}
		//!!!DEBUGGING

		SIFTMatchFilter::init();

		DualGPU& dualGPU = DualGPU::get();	//needs to be called to initialize devices
		dualGPU.setDevice(DualGPU::DEVICE_RECONSTRUCTION);	//main gpu
		ConditionManager::init();

		g_RGBDSensor = getRGBDSensor();

		//init the input RGBD sensor
		if (g_RGBDSensor == NULL) throw MLIB_EXCEPTION("No RGBD sensor specified");
		g_RGBDSensor->createFirstConnected();

		
		g_imageManager = new CUDAImageManager(GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
			GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, g_RGBDSensor, false);

		//std::thread(bundlingThreadFunc).detach();
		std::thread bundlingThread(bundlingThreadFunc);
		

		//waiting until bundler is initialized
		while (!g_bundler)	Sleep(0);


		dualGPU.setDevice(DualGPU::DEVICE_RECONSTRUCTION);	//main gpu
	
		//start depthSensing render loop
		startDepthSensing(g_bundler, getRGBDSensor(), g_imageManager);

		TimingLog::printAllTimings();
		g_bundler->saveGlobalSiftManagerToFile("debug/global.sift");
		if (GlobalBundlingState::get().s_recordSolverConvergence) g_bundler->saveConvergence("convergence.txt");
		g_bundler->saveCompleteTrajectory("trajectory.bin");
		g_bundler->saveSiftTrajectory("siftTrajectory.bin");
		g_bundler->saveIntegrateTrajectory("intTrajectory.bin");
		//g_bundler->saveDEBUG();

		g_bundler->exitBundlingThread();
		
		g_imageManager->setBundlingFrameRdy();			//release all bundling locks
		g_bundler->confirmProcessedInputFrame();		//release all bundling locks
		ConditionManager::release(ConditionManager::Recon); // release bundling locks

		if (bundlingThread.joinable())	bundlingThread.join();	//wait for the bundling thread to return;
		SAFE_DELETE(g_bundler);
		SAFE_DELETE(g_imageManager);
		
		//ConditionManager::DEBUGRELEASE();

		std::cout << "DONE! <<press key to exit program>>" << std::endl;
		getchar();
	}
	catch (const std::exception& e)
	{
		MessageBoxA(NULL, e.what(), "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		MessageBoxA(NULL, "UNKNOWN EXCEPTION", "Exception caught", MB_ICONERROR);
		exit(EXIT_FAILURE);
	}


	return 0;
}


