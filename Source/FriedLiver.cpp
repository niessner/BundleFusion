

#include "stdafx.h"

#include "FriedLiver.h"


CUDAImageManager		g_CudaImageManager;
SubmapManager			g_SubmapManager;
SBA						g_SparseBundler;

SiftGPU					*g_sift			= NULL;
SiftMatchGPU			*g_siftMatcher	= NULL;



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



void init() {
	
	if (getRGBDSensor() == NULL) throw MLIB_EXCEPTION("No RGBD sensor specified");

	//init the input RGBD sensor
	getRGBDSensor()->createFirstConnected();


	// init CUDA
	g_CudaImageManager.init(GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
		GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, getRGBDSensor());
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	g_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage,
		submapSize);
	//TODO fix
	if (GlobalAppState::get().s_sensorIdx == 3) {
		g_SubmapManager.setTotalNumFrames(((BinaryDumpReader*)getRGBDSensor())->getNumTotalFrames());
	}
	g_SparseBundler.init(GlobalBundlingState::get().s_maxNumImages, GlobalBundlingState::get().s_maxNumCorrPerImage);

	g_sift = new SiftGPU;
	g_siftMatcher = new SiftMatchGPU(GlobalBundlingState::get().s_maxNumKeysPerImage);
	g_sift->SetParams(0, GlobalBundlingState::get().s_enableDetailedTimings, 150);
	g_sift->InitSiftGPU();
	g_siftMatcher->InitSiftMatch();
}

void destroy() {
	SAFE_DELETE(g_sift);
	SAFE_DELETE(g_siftMatcher);
}

//TODO fix
void MatchAndFilter(SIFTImageManager* siftManager, const std::vector<CUDACache::CUDACachedFrame>& cachedFrames, const std::vector<int>& validImages);
void solve(std::vector<mat4f>& transforms, SIFTImageManager* siftManager);
void processCurrentFrame();

//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
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
		}

		std::cout << VAR_NAME(fileNameDescGlobalApp) << " = " << fileNameDescGlobalApp << std::endl;
		std::cout << VAR_NAME(fileNameDescGlobalBundling) << " = " << fileNameDescGlobalBundling << std::endl;
		std::cout << std::endl;

		//Read the global app state
		ParameterFile parameterFileGlobalApp(fileNameDescGlobalApp);
		GlobalAppState::getInstance().readMembers(parameterFileGlobalApp);
		//GlobalAppState::getInstance().print();

		//Read the global camera tracking state
		ParameterFile parameterFileGlobalBundling(fileNameDescGlobalBundling);
		GlobalBundlingState::getInstance().readMembers(parameterFileGlobalBundling);
		//GlobalCameraTrackingState::getInstance().print();


		init();
		
		//TODO debug stuff
		const std::string outputDirectory = GlobalBundlingState::get().s_outputDirectory;
		if (!util::directoryExists(outputDirectory)) util::makeDirectory(outputDirectory);
		const std::string outGlobalDir = outputDirectory + "keys/";
		if (!util::directoryExists(outGlobalDir)) util::makeDirectory(outGlobalDir);
		//TODO 
		while (g_CudaImageManager.process()) {
			processCurrentFrame();
		}

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

	//TODO stupid hack
	SIFTMatchFilter::free();

	getchar();
	return 0;
}


void processCurrentFrame()
{
	const unsigned int curFrame = g_CudaImageManager.getCurrFrameNumber();
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	std::cout << "[ frame " << curFrame << " ]" << std::endl;

	// run SIFT
	SIFTImageGPU& cur = g_SubmapManager.currentLocal->createSIFTImageGPU();
	Timer timer; //TODO 
	int success = g_sift->RunSIFT(g_CudaImageManager.getIntensityImage(), g_CudaImageManager.getDepthInput());
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection on frame " + std::to_string(curFrame));
	unsigned int numKeypoints = g_sift->GetKeyPointsAndDescriptorsCUDA(cur, g_CudaImageManager.getDepthInput());
	g_SubmapManager.currentLocal->finalizeSIFTImageGPU(numKeypoints);
	timer.stop();
	TimingLog::timeSiftDetection += timer.getElapsedTimeMS();
	TimingLog::countSiftDetection++;

	// process cuda cache
	const unsigned int curLocalIdx = g_SubmapManager.getCurrLocalIdx(curFrame);
	g_SubmapManager.currentLocalCache->storeFrame(g_CudaImageManager.getLastIntegrateDepth(), g_CudaImageManager.getLastIntegrateColor(), g_CudaImageManager.getIntegrationWidth(), g_CudaImageManager.getIntegrationHeight());

	// local submaps
	if (g_SubmapManager.isLastLocalFrame(curFrame)) {
		SIFTImageGPU& curNext = g_SubmapManager.nextLocal->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(curNext.d_keyPoints, cur.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(curNext.d_keyPointDescs, cur.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		g_SubmapManager.nextLocal->finalizeSIFTImageGPU(numKeypoints);

		g_SubmapManager.nextLocalCache->copyCacheFrameFrom(g_SubmapManager.currentLocalCache, curLocalIdx);
	}

	// match with every other local
	SIFTImageManager* currentLocal = g_SubmapManager.currentLocal;
	const unsigned int curLocalFrame = currentLocal->getNumImages() - 1;
	std::vector<int> validImagesLocal; currentLocal->getValidImagesDEBUG(validImagesLocal);
	if (curLocalFrame > 0) {
		const std::vector<CUDACache::CUDACachedFrame>& cachedFrames = g_SubmapManager.currentLocalCache->getCacheFrames();
		MatchAndFilter(currentLocal, cachedFrames, validImagesLocal);
	}

	// global frame
	if (g_SubmapManager.isLastFrame(curFrame) || g_SubmapManager.isLastLocalFrame(curFrame)) { // end frame or global frame

		// cache
		g_SubmapManager.globalCache->copyCacheFrameFrom(g_SubmapManager.currentLocalCache, curLocalIdx);
		
		// if valid local
		if (validImagesLocal[1]) {
			// solve local
			//const std::string curLocalOutDir = GlobalAppState::get().s_outputDirectory + std::to_string(curLocalIdx) + "/";
			//if (!util::directoryExists(curLocalOutDir)) util::makeDirectory(curLocalOutDir);
			const std::string curLocalOutDir = "";
			std::vector<mat4f> currentLocalTrajectory(currentLocal->getNumImages(), mat4f::identity());
			solve(currentLocalTrajectory, g_SubmapManager.currentLocal);
			g_SubmapManager.localTrajectories.push_back(currentLocalTrajectory);

			// fuse to global
			SIFTImageManager* global = g_SubmapManager.global;
			const mat4f lastTransform = g_SubmapManager.globalTrajectory.back();
			g_SubmapManager.currentLocal->fuseToGlobal(global, (float4x4*)currentLocalTrajectory.data(), currentLocal->getNumImages() - 1); // overlap frame

			//unsigned int gframe = (unsigned int)global->getNumImages() - 1;
			//printKey(GlobalAppState::get().s_outputDirectory + "keys/" + std::to_string(gframe) + ".png", binaryDumpReader, gframe*submapSize, global, gframe);

			// switch local submaps
			g_SubmapManager.switchLocal();

			// match with every other global
			std::vector<int> validImagesGlobal; global->getValidImagesDEBUG(validImagesGlobal);
			if (global->getNumImages() > 1) {
				const std::vector<CUDACache::CUDACachedFrame>& cachedFrames = g_SubmapManager.globalCache->getCacheFrames();

				MatchAndFilter(global, cachedFrames, validImagesGlobal);
				//printCurrentMatches("output/matches/", binaryDumpReader, global, true, 0, submapSize);

				if (validImagesGlobal.back()) {
					// solve global
					const std::string outGlobalDir = GlobalBundlingState::get().s_outputDirectory + "keys/";
					solve(g_SubmapManager.globalTrajectory, global);
				}
			}

			// complete trajectory
			g_SubmapManager.updateTrajectory();
			g_SubmapManager.globalTrajectory.push_back(lastTransform * currentLocalTrajectory.back()); //initialize next one
		}
		else { //!!!TODO check if need
			std::vector<mat4f> currentLocalTrajectory(submapSize + 1, mat4f::identity());
			g_SubmapManager.localTrajectories.push_back(currentLocalTrajectory);
			g_SubmapManager.updateTrajectory();
			g_SubmapManager.globalTrajectory.push_back(g_SubmapManager.globalTrajectory.back()); //initialize next one
		}
	} // global
}

void solve(std::vector<mat4f>& transforms, SIFTImageManager* siftManager)
{
	MLIB_ASSERT(transforms.size() == siftManager->getNumImages());
	bool useVerify = false; //TODO do we need verify?
	g_SparseBundler.align(siftManager, transforms, GlobalBundlingState::get().s_numNonLinIterations, GlobalBundlingState::get().s_numLinIterations, useVerify);
	//if (useVerify) bundle->verifyTrajectory();
}

void MatchAndFilter(SIFTImageManager* siftManager, const std::vector<CUDACache::CUDACachedFrame>& cachedFrames, const std::vector<int>& validImages)
{
	// match with every other
	const unsigned int curFrame = siftManager->getNumImages() - 1;
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		uint2 keyPointOffset = make_uint2(0, 0);
		ImagePairMatch& imagePairMatch = siftManager->getImagePairMatch(prev, keyPointOffset);

		SIFTImageGPU& image_i = siftManager->getImageGPU(prev);
		SIFTImageGPU& image_j = siftManager->getImageGPU(curFrame);
		int num1 = (int)siftManager->getNumKeyPointsPerImage(prev);
		int num2 = (int)siftManager->getNumKeyPointsPerImage(curFrame);

		if (validImages[prev] == 0 || num1 == 0 || num2 == 0) {
			unsigned int numMatch = 0;
			cutilSafeCall(cudaMemcpy(imagePairMatch.d_numMatches, &numMatch, sizeof(unsigned int), cudaMemcpyHostToDevice));
		}
		else {
			Timer timer;
			g_siftMatcher->SetDescriptors(0, num1, (unsigned char*)image_i.d_keyPointDescs);
			g_siftMatcher->SetDescriptors(1, num2, (unsigned char*)image_j.d_keyPointDescs);
			g_siftMatcher->GetSiftMatch(num1, imagePairMatch, keyPointOffset);
			timer.stop();
			TimingLog::timeSiftMatching += timer.getElapsedTimeMS();
			TimingLog::countSiftMatching++;
			//unsigned int numMatch; cutilSafeCall(cudaMemcpy(&numMatch, imagePairMatch.d_numMatches, sizeof(int), cudaMemcpyDeviceToHost));
			//std::cout << "images (" << prev << ", " << curGlobalFrame << "): " << numMatch << " matches" << std::endl;
			////printMatch(&siftManager, outDir + std::to_string(prev) + "-" + std::to_string(curGlobalFrame) + ".png", ml::vec2ui(prev, curGlobalFrame),
			////	intensityImages[prev], intensityImages[curGlobalFrame], distMax, false);
		}
	}

	if (curFrame > 0) { // can have a match to another frame
		//sort the current key point matches
		siftManager->SortKeyPointMatchesCU(curFrame);
		//printCurrentMatches("output/matches/", binaryDumpReader, siftManager, false);

		//filter matches
		SIFTMatchFilter::filterKeyPointMatches(siftManager);
		//global->FilterKeyPointMatchesCU(curFrame);
		SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);

		SIFTMatchFilter::filterFrames(siftManager);
		//printCurrentMatches("output/matchesFilt/", binaryDumpReader, siftManager, true);

		// add to global correspondences
		siftManager->AddCurrToResidualsCU(curFrame);
	}
}