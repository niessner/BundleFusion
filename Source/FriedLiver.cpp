

#include "stdafx.h"

#include "FriedLiver.h"

//!!!debugging
extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans);

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms);
//!!!debugging

RGBDSensor* getRGBDSensor()
{
	static RGBDSensor* g_sensor = NULL;
	if (g_sensor != NULL)	return g_sensor;

	if (GlobalAppState::get().s_sensorIdx == 0) {
#ifdef KINECT
		//static KinectSensor s_kinect;
		//return &s_kinect;
		g_sensor = new KinectSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires KINECT V1 SDK and enable KINECT macro");
#endif
	}

	if (GlobalAppState::get().s_sensorIdx == 1)	{
#ifdef OPEN_NI
		//static PrimeSenseSensor s_primeSense;
		//return &s_primeSense;
		g_sensor = new PrimeSenseSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires OpenNI 2 SDK and enable OPEN_NI macro");
#endif
	}
	else if (GlobalAppState::getInstance().s_sensorIdx == 2) {
#ifdef KINECT_ONE
		//static KinectOneSensor s_kinectOne;
		//return &s_kinectOne;
		g_sensor = new KinectOneSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires Kinect 2.0 SDK and enable KINECT_ONE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 3) {
#ifdef BINARY_DUMP_READER
		//static BinaryDumpReader s_binaryDump;
		//return &s_binaryDump;
		g_sensor = new BinaryDumpReader;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires BINARY_DUMP_READER macro");
#endif
	}
	//	if (GlobalAppState::get().s_sensorIdx == 4) {
	//		//static NetworkSensor s_networkSensor;
	//		//return &s_networkSensor;
	//		g_sensor = new NetworkSensor;
	//		return g_sensor;
	//}
	if (GlobalAppState::get().s_sensorIdx == 5) {
#ifdef INTEL_SENSOR
		//static IntelSensor s_intelSensor;
		//return &s_intelSensor;
		g_sensor = new IntelSensor;
		return g_sensor;
#else 
		throw MLIB_EXCEPTION("Requires INTEL_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 6) {
#ifdef REAL_SENSE
		//static RealSenseSensor s_realSenseSensor;
		//return &s_realSenseSensor;
		g_sensor = RealSenseSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires Real Sense SDK and REAL_SENSE macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 7) {
#ifdef STRUCTURE_SENSOR
		//static StructureSensor s_structureSensor;
		//return &s_structureSensor;
		g_sensor = new StructureSensor;
		return g_sensor;
#else
		throw MLIB_EXCEPTION("Requires STRUCTURE_SENSOR macro");
#endif
	}
	if (GlobalAppState::get().s_sensorIdx == 8) {
#ifdef SENSOR_DATA_READER
		//static SensorDataReader s_sensorDataReader;
		//return &s_sensorDataReader;
		g_sensor = new SensorDataReader;
		return g_sensor;
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

	// for no opt
	//g_bundler->resetDEBUG();
}

void bundlingOptimizationThreadFunc() {

	DualGPU::get().setDevice(DualGPU::DEVICE_BUNDLING);

	bundlingOptimization();
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
	//_CrtSetBreakAlloc(3333);
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
			//fileNameDescGlobalApp = "zParametersDefault.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingDefault.txt";

			//fileNameDescGlobalApp = "zParametersMedium.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingHigh.txt";

			//fileNameDescGlobalApp = "zParametersHigh.txt";
			//fileNameDescGlobalBundling = "zParametersBundlingHigh.txt";

			fileNameDescGlobalApp = "zParametersTUM.txt";
			fileNameDescGlobalBundling = "zParametersBundlingTUM.txt";

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
		if (true) {
			//debug test lie conversion
			std::vector<mat4f> transforms(10, mat4f::identity());
			RNG::global.init(0, 1, 2, 3); const float maxTrans = 0.1f; const float maxRot = 7.0f; 
			for (unsigned int i = 1; i < transforms.size(); i++) {
				float tx = RNG::global.uniform(0.0f, maxTrans); if (RNG::global.uniform(0, 1) == 0) tx = -tx;
				float ty = RNG::global.uniform(0.0f, maxTrans); if (RNG::global.uniform(0, 1) == 0) ty = -ty;
				float tz = RNG::global.uniform(0.0f, maxTrans); if (RNG::global.uniform(0, 1) == 0) ty = -ty;
				float rx = RNG::global.uniform(0.0f, maxRot); if (RNG::global.uniform(0, 1) == 0) rx = -rx;
				float ry = RNG::global.uniform(0.0f, maxRot); if (RNG::global.uniform(0, 1) == 0) ry = -ry;
				float rz = RNG::global.uniform(0.0f, maxRot); if (RNG::global.uniform(0, 1) == 0) ry = -ry;
				transforms[i] = mat4f::translation(tx, ty, tz) * mat4f::rotationZ(rz) * mat4f::rotationY(ry) * mat4f::rotationX(rx);
			}
			std::vector<Pose> poses = PoseHelper::convertToPoses(transforms);
			//gpu conversion
			const float eps = 0.00001f;
			float4x4* d_transforms = NULL; float3* d_xRot = NULL; float3* d_xTrans = NULL;
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_transforms, sizeof(float4x4)*transforms.size()));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xRot, sizeof(float3)*transforms.size()));
			MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xTrans, sizeof(float3)*transforms.size()));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_transforms, transforms.data(), sizeof(float4x4)*transforms.size(), cudaMemcpyHostToDevice));
			convertMatricesToPosesCU(d_transforms, (unsigned int)transforms.size(), d_xRot, d_xTrans);
			std::vector<vec3f> rot(transforms.size()), trans(transforms.size());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(rot.data(), d_xRot, sizeof(float3)*transforms.size(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(trans.data(), d_xTrans, sizeof(float3)*transforms.size(), cudaMemcpyDeviceToHost));

			//Pose p = PoseHelper::MatrixToPose(transforms[1]);

			for (unsigned int i = 0; i < poses.size(); i++) {
				vec3f rdiff = poses[i].getVec3() - rot[i];
				vec3f tdiff = vec3f(poses[i][3], poses[i][4], poses[i][5]) - trans[i];
				float rlen = rdiff.length(); float tlen = tdiff.length();
				if (rlen > eps || tlen > eps) {
					int a = 5;
				}
			}
			convertPosesToMatricesCU(d_xRot, d_xTrans, (unsigned int)transforms.size(), d_transforms);
			std::vector<mat4f> newTransforms(transforms.size());
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(newTransforms.data(), d_transforms, sizeof(float4x4)*transforms.size(), cudaMemcpyDeviceToHost));
			for (unsigned int i = 0; i < transforms.size(); i++) {
				for (unsigned int k = 0; k < 16; k++) {
					if (fabs(transforms[i][k] - newTransforms[i][k]) > eps)
						int a = 5;
				}
			}

			MLIB_CUDA_SAFE_FREE(d_transforms);
			MLIB_CUDA_SAFE_FREE(d_xRot);
			MLIB_CUDA_SAFE_FREE(d_xTrans);
			std::cout << "waiting..." << std::endl;
			getchar();

			TestMatching test;
			test.loadFromSensor("debug/66.sensor", "", 1);
			//test.printCacheFrames("debug/cache/");
			test.runOpt();
			//test.testGlobalDense();

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

#ifdef RUN_MULTITHREADED
		std::thread bundlingThread(bundlingThreadFunc);

		//waiting until bundler is initialized
		while (!g_bundler)	Sleep(0);
#else
		g_bundler = new Bundler(g_RGBDSensor, g_imageManager);
#endif

		dualGPU.setDevice(DualGPU::DEVICE_RECONSTRUCTION);	//main gpu

		//start depthSensing render loop
		startDepthSensing(g_bundler, getRGBDSensor(), g_imageManager);

		TimingLog::printAllTimings();
		g_bundler->saveGlobalSiftManagerAndCacheToFile("debug/global");
		if (GlobalBundlingState::get().s_recordSolverConvergence) g_bundler->saveConvergence("convergence.txt");
		g_bundler->saveCompleteTrajectory("trajectory.bin");
		g_bundler->saveSiftTrajectory("siftTrajectory.bin");
		g_bundler->saveIntegrateTrajectory("intTrajectory.bin");
		//g_bundler->saveDEBUG();

#ifdef RUN_MULTITHREADED
		g_bundler->exitBundlingThread();

		g_imageManager->setBundlingFrameRdy();			//release all bundling locks
		g_bundler->confirmProcessedInputFrame();		//release all bundling locks
		ConditionManager::release(ConditionManager::Recon); // release bundling locks

		if (bundlingThread.joinable())	bundlingThread.join();	//wait for the bundling thread to return;
#endif
		SAFE_DELETE(g_bundler);
		SAFE_DELETE(g_imageManager);

		//ConditionManager::DEBUGRELEASE();

		//this is a bit of a hack due to a bug in std::thread (a static object cannot join if the main thread exists)
		auto* s = getRGBDSensor();
		SAFE_DELETE(s);

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


