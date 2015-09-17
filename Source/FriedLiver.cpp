

#include "stdafx.h"

#include "FriedLiver.h"


Bundler		g_bundler;



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

	g_bundler.init(getRGBDSensor());
}



//#include "SiftGPU/cuda_EigenValue.h"
//#include "SiftGPU/cuda_SVD.h"

//int WINAPI main(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
int main(int argc, char** argv)
{
	// Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//_CrtSetBreakAlloc(3727);
#endif 
	
	////RNG::global.init(10, 10, 10, 10);
	//srand(100);
	//unsigned int numTests = 100;
	//for (unsigned int t = 0; t < numTests; t++) {
	//	float4x4 m_test;
	//	for (unsigned int i = 0; i < 16; i++) {
	//		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	//		//m_test[i] = RNG::global.rand_closed01();
	//		m_test[i] = r;
	//	}
	//	mat4f m_gt = *(mat4f*)&m_test;
	//	for (unsigned int i = 0; i < 5; i++) {
	//		m_test = m_test * m_test;
	//		m_gt = m_gt * m_gt;

	//		std::cout << m_gt << std::endl;
	//		std::cout << *(mat4f*)&m_test << std::endl;
	//		std::cout << m_gt - *(mat4f*)&m_test << std::endl;

	//		std::cout << std::endl;
	//		getchar();
	//	}
	//}
	//std::cout << "done!" << std::endl;
	//getchar();
	//exit(1);

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
		while (g_bundler.process(getRGBDSensor()))
		{
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

	TimingLog::printTimings("timingLog.txt");
	if (GlobalBundlingState::get().s_recordKeysPointCloud) g_bundler.saveKeysToPointCloud(getRGBDSensor());

	getchar();
	return 0;
}


