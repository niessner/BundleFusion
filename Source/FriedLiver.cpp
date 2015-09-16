

#include "stdafx.h"

#include "FriedLiver.h"


CUDAImageManager*		g_CudaImageManager = NULL;
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
	g_CudaImageManager = new CUDAImageManager(GlobalAppState::get().s_integrationWidth, GlobalAppState::get().s_integrationHeight,
		GlobalBundlingState::get().s_widthSIFT, GlobalBundlingState::get().s_heightSIFT, getRGBDSensor());
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	g_SubmapManager.init(GlobalBundlingState::get().s_maxNumImages, submapSize + 1, GlobalBundlingState::get().s_maxNumKeysPerImage, submapSize, g_CudaImageManager);
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
void MatchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, const std::vector<int>& validImages,
	unsigned int frameStart, unsigned int frameSkip, bool print = false);
void solve(std::vector<mat4f>& transforms, SIFTImageManager* siftManager);
void processCurrentFrame();
void printKey(const std::string& filename, unsigned int allFrame, const SIFTImageManager* siftManager, unsigned int frame)
{
	ColorImageR8G8B8A8 im(g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(im.getPointer(), g_CudaImageManager->getIntegrateColor(allFrame), sizeof(uchar4) * g_CudaImageManager->getIntegrationWidth() * g_CudaImageManager->getIntegrationHeight(), cudaMemcpyDeviceToHost));
	im.reSample(g_CudaImageManager->getSIFTWidth(), g_CudaImageManager->getSIFTHeight());

	std::vector<SIFTKeyPoint> keys(siftManager->getNumKeyPointsPerImage(frame));
	const SIFTImageGPU& cur = siftManager->getImageGPU(frame);
	cutilSafeCall(cudaMemcpy(keys.data(), cur.d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < keys.size(); i++) {
		const SIFTKeyPoint& key = keys[i];

		RGBColor c = RGBColor::randomColor();
		vec4uc color(c.r, c.g, c.b, c.a);
		vec2i p0 = math::round(vec2f(key.pos.x, key.pos.y));
		ImageHelper::drawCircle(im, p0, math::round(key.scale), color);
	}
	FreeImageWrapper::saveImage(filename, im);
}
template<typename T>
void printMatch(const SIFTImageManager* siftManager, const std::string& filename,
	const vec2ui& imageIndices, const BaseImage<T>& image1, const BaseImage<T>& image2, float distMax, bool filtered)
{
	// get data
	std::vector<SIFTKeyPoint> keys;
	siftManager->getSIFTKeyPointsDEBUG(keys); // prev frame

	std::vector<uint2> keyPointIndices;
	std::vector<float> matchDistances;
	if (filtered) {
		siftManager->getFiltKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	else {
		siftManager->getRawKeyPointIndicesAndMatchDistancesDEBUG(imageIndices.x, keyPointIndices, matchDistances);
	}
	if (keyPointIndices.size() == 0) return;

	ColorImageR32G32B32 matchImage(image1.getWidth() * 2, image1.getHeight());
	ColorImageR32G32B32 im1(image1);
	ColorImageR32G32B32 im2(image2);
	matchImage.copyIntoImage(im1, 0, 0);
	matchImage.copyIntoImage(im2, image1.getWidth(), 0);

	RGBColor lowColor = ml::RGBColor::Blue;
	RGBColor highColor = ml::RGBColor::Red;
	for (unsigned int i = 0; i < keyPointIndices.size(); i++) {
		const SIFTKeyPoint& key1 = keys[keyPointIndices[i].x];
		const SIFTKeyPoint& key2 = keys[keyPointIndices[i].y];

		RGBColor c = RGBColor::interpolate(lowColor, highColor, matchDistances[i] / distMax);
		vec3f color(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f);
		vec2i p0 = ml::math::round(ml::vec2f(key1.pos.x, key1.pos.y));
		vec2i p1 = ml::math::round(ml::vec2f(key2.pos.x + image1.getWidth(), key2.pos.y));
		ImageHelper::drawCircle(matchImage, p0, ml::math::round(key1.scale), color);
		ImageHelper::drawCircle(matchImage, p1, ml::math::round(key2.scale), color);
		ImageHelper::drawLine(matchImage, p0, p1, color);
	}
	FreeImageWrapper::saveImage(filename, matchImage);
}

void printCurrentMatches(const std::string& outPath, const SIFTImageManager* siftManager, bool filtered,
	unsigned int frameStart, unsigned int frameSkip)
{
	const unsigned int numFrames = siftManager->getNumImages();
	if (numFrames <= 1) return;

	const std::string dir = util::directoryFromPath(outPath);
	MLIB_ASSERT(util::directoryExists(dir));

	// get images
	unsigned int curFrame = numFrames - 1;
	ColorImageR8G8B8A8 curImage(g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(curImage.getPointer(), g_CudaImageManager->getIntegrateColor(curFrame * frameSkip + frameStart), 
		sizeof(uchar4) * curImage.getNumPixels(), cudaMemcpyDeviceToHost));
	curImage.reSample(g_CudaImageManager->getSIFTWidth(), g_CudaImageManager->getSIFTHeight());

	//print out images
	for (unsigned int prev = 0; prev < curFrame; prev++) {
		ColorImageR8G8B8A8 prevImage(g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(prevImage.getPointer(), g_CudaImageManager->getIntegrateColor(prev * frameSkip + frameStart),
			sizeof(uchar4) * prevImage.getNumPixels(), cudaMemcpyDeviceToHost));
		prevImage.reSample(g_CudaImageManager->getSIFTWidth(), g_CudaImageManager->getSIFTHeight());

		printMatch(siftManager, outPath + std::to_string(prev) + "-" + std::to_string(curFrame) + ".png", ml::vec2ui(prev, curFrame),
			prevImage, curImage, 0.7f, filtered);
	}
}

//#include "SiftGPU/cuda_EigenValue.h"
//#include "SiftGPU/cuda_SVD.h"

//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif 

	//float2x2 m(1, 2, 3, 4);
	//m = m * m.getTranspose();
	//float2 evs = computeEigenValues(m);
	//float2 ev0 = computeEigenVector(m, evs.x);
	//float2 ev1 = computeEigenVector(m, evs.y);
	//auto res = ((mat2f*)&m)->eigenSystem();
	//int a = 5;
	//getchar();
	//exit(1);

	//float3x3 m(0.038489, -0.012003, -0.004768,
	//	-0.012003, 0.015097, 0.006327,
	//	-0.004768, 0.006327, 0.002659);
	//float3x3 m(1, 2, 3, 4, 5, 6, 7, 8, 9);
	//m = m * m.getTranspose();
	////float3 evs = computeEigenValues(m);
	////float3 ev0 = computeEigenVector(m, evs.x);
	////float3 ev1 = computeEigenVector(m, evs.y);
	////float3 ev2 = computeEigenVector(m, evs.z);
	//
	//float3 evs, ev0, ev1, ev2;
	//Timer t;
	//for (unsigned int i = 0; i < 100000; i++)
	//	MYEIGEN::eigenSystem(m, evs, ev0, ev1, ev2);
	//t.stop();
	//std::cout << "first " << t.getElapsedTimeMS() << std::endl;
	//t.start();
	//for (unsigned int i = 0; i < 100000; i++)
	//	MYEIGEN::eigenSystem3x3(m, evs, ev0, ev1, ev2);
	//std::cout << "second: " << t.getElapsedTimeMS() << std::endl;
	//float3x3 re(ev0, ev1, ev2);
	//re = re * float3x3::getDiagonalMatrix(evs.x, evs.y, evs.z) * re.getInverse();

	//auto res = ((mat3f*)&m)->eigenSystem();
	//mat3f res2 = (mat3f)res.eigenvectors * mat3f::diag(res.eigenvalues[0], res.eigenvalues[1], res.eigenvalues[2]) * ((mat3f)res.eigenvectors).getInverse();
	//int a = 5;
	//getchar();
	//exit(1);

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

		TimingLog::init();


		init();
		
		//TODO 
		while (g_CudaImageManager->process()) {
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

	//!!!DEBUG
	g_SubmapManager.evaluateTimings();
	TimingLog::printTimings("timingLog.txt");
	std::vector<int> validImagesGlobal; g_SubmapManager.global->getValidImagesDEBUG(validImagesGlobal);
	getRGBDSensor()->saveRecordedPointCloud("refined.ply", validImagesGlobal, g_SubmapManager.globalTrajectory);
	destroy();

	getchar();
	return 0;
}


void processCurrentFrame()
{
	const unsigned int curFrame = g_CudaImageManager->getCurrFrameNumber();
	const unsigned int submapSize = GlobalBundlingState::get().s_submapSize;
	std::cout << "[ frame " << curFrame << " ]" << std::endl;

	// run SIFT
	SIFTImageGPU& cur = g_SubmapManager.currentLocal->createSIFTImageGPU();
	Timer timer; //TODO 
	int success = g_sift->RunSIFT(g_CudaImageManager->getIntensityImage(), g_CudaImageManager->getDepthInput());
	if (!success) throw MLIB_EXCEPTION("Error running SIFT detection on frame " + std::to_string(curFrame));
	unsigned int numKeypoints = g_sift->GetKeyPointsAndDescriptorsCUDA(cur, g_CudaImageManager->getDepthInput());
	g_SubmapManager.currentLocal->finalizeSIFTImageGPU(numKeypoints);
	timer.stop();
	TimingLog::timeSiftDetection += timer.getElapsedTimeMS();
	TimingLog::countSiftDetection++;

	// process cuda cache
	const unsigned int curLocalFrame = g_SubmapManager.currentLocal->getNumImages() - 1;
	g_SubmapManager.currentLocalCache->storeFrame(g_CudaImageManager->getLastIntegrateDepth(), g_CudaImageManager->getLastIntegrateColor(), g_CudaImageManager->getIntegrationWidth(), g_CudaImageManager->getIntegrationHeight());
	if (curLocalFrame == 0 || g_SubmapManager.isLastLocalFrame(curFrame)) {
		//!!!DEBUG
		getRGBDSensor()->recordPointCloud();
		//!!!
	}
	//printKey("key" + std::to_string(curLocalFrame) + ".png", curFrame, g_SubmapManager.currentLocal, curLocalFrame);


	// local submaps
	if (g_SubmapManager.isLastLocalFrame(curFrame)) {
		SIFTImageGPU& curNext = g_SubmapManager.nextLocal->createSIFTImageGPU();
		cutilSafeCall(cudaMemcpy(curNext.d_keyPoints, cur.d_keyPoints, sizeof(SIFTKeyPoint) * numKeypoints, cudaMemcpyDeviceToDevice));
		cutilSafeCall(cudaMemcpy(curNext.d_keyPointDescs, cur.d_keyPointDescs, sizeof(SIFTKeyPointDesc) * numKeypoints, cudaMemcpyDeviceToDevice));
		g_SubmapManager.nextLocal->finalizeSIFTImageGPU(numKeypoints);

		g_SubmapManager.nextLocalCache->copyCacheFrameFrom(g_SubmapManager.currentLocalCache, curLocalFrame);
	}

	// match with every other local
	SIFTImageManager* currentLocal = g_SubmapManager.currentLocal;
	std::vector<int> validImagesLocal; currentLocal->getValidImagesDEBUG(validImagesLocal);
	if (curLocalFrame > 0) {
		MatchAndFilter(currentLocal, g_SubmapManager.currentLocalCache, validImagesLocal, curFrame - curLocalFrame, 1);
	}

	// global frame
	if (g_SubmapManager.isLastFrame(curFrame) || g_SubmapManager.isLastLocalFrame(curFrame)) { // end frame or global frame

		// cache
		g_SubmapManager.globalCache->copyCacheFrameFrom(g_SubmapManager.currentLocalCache, 0);
		
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
			//SIFTImageGPU& curGlobalImage = global->createSIFTImageGPU();
			//unsigned int numGlobalKeys = g_SubmapManager.currentLocal->FuseToGlobalKeyCU(curGlobalImage, (float4x4*)currentLocalTrajectory.data(),
			//	MatrixConversion::toCUDA(g_CudaImageManager->getSIFTIntrinsics()), MatrixConversion::toCUDA(g_CudaImageManager->getSIFTIntrinsicsInv()));
			//global->finalizeSIFTImageGPU(numGlobalKeys);

			//unsigned int gframe = (unsigned int)global->getNumImages() - 1;
			//printKey("debug/keys/" + std::to_string(gframe) + ".png", gframe*submapSize, global, gframe);
			////!!!
			//std::vector<SIFTKeyPoint> curKeys;
			//global->getSIFTKeyPointsDEBUG(curKeys);
			//std::sort(curKeys.begin(), curKeys.end(), [](const SIFTKeyPoint& left, const SIFTKeyPoint& right) {
			//	if (left.pos.x < right.pos.x) return true;
			//	else if (left.pos.x > right.pos.x) return false;
			//	if (left.pos.y < right.pos.y) return true;
			//	else if (left.pos.y > right.pos.y) return false;
			//	return (left.depth < right.depth);
			//});
			////!!!

			// switch local submaps
			g_SubmapManager.switchLocal();

			// match with every other global
			std::vector<int> validImagesGlobal; global->getValidImagesDEBUG(validImagesGlobal);
			if (global->getNumImages() > 1) {
				MatchAndFilter(global, g_SubmapManager.globalCache, validImagesGlobal, 0, submapSize);
				//printCurrentMatches("output/matches/", binaryDumpReader, global, true, 0, submapSize);

				if (validImagesGlobal.back()) {
					// solve global
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

void MatchAndFilter(SIFTImageManager* siftManager, const CUDACache* cudaCache, const std::vector<int>& validImages,
	unsigned int frameStart, unsigned int frameSkip, bool print /*= false*/) // frameStart/frameSkip for debugging (printing matches)
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
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(imagePairMatch.d_numMatches, &numMatch, sizeof(unsigned int), cudaMemcpyHostToDevice));
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
			//std::cout << "images (" << prev << ", " << curFrame << "): " << numMatch << " matches" << std::endl;
		}
	}

	if (curFrame > 0) { // can have a match to another frame
		//sort the current key point matches
		siftManager->SortKeyPointMatchesCU(curFrame);
		if (print) printCurrentMatches("debug/", siftManager, false, frameStart, frameSkip);

		//filter matches
		SIFTMatchFilter::filterKeyPointMatches(siftManager);
		//global->FilterKeyPointMatchesCU(curFrame);

		const std::vector<CUDACachedFrame>& cachedFrames = cudaCache->getCacheFrames();
		SIFTMatchFilter::filterBySurfaceArea(siftManager, cachedFrames);
		//siftManager->FilterMatchesBySurfaceAreaCU(curFrame, MatrixConversion::toCUDA(g_CudaImageManager->getSIFTIntrinsicsInv()), GlobalBundlingState::get().s_surfAreaPcaThresh);

		//SIFTMatchFilter::filterByDenseVerify(siftManager, cachedFrames);
		const CUDACachedFrame* cachedFramesCUDA = cudaCache->getCacheFramesGPU();
		siftManager->FilterMatchesByDenseVerifyCU(curFrame, cudaCache->getWidth(), cudaCache->getHeight(), MatrixConversion::toCUDA(cudaCache->getIntrinsics()),
			cachedFramesCUDA, GlobalBundlingState::get().s_projCorrDistThres, GlobalBundlingState::get().s_projCorrNormalThres,
			GlobalBundlingState::get().s_projCorrColorThresh, GlobalBundlingState::get().s_verifySiftErrThresh, GlobalBundlingState::get().s_verifySiftCorrThresh);


		SIFTMatchFilter::filterFrames(siftManager);
		if (print) printCurrentMatches("debug/filt", siftManager, true, frameStart, frameSkip);

		// add to global correspondences
		siftManager->AddCurrToResidualsCU(curFrame);
	}
}