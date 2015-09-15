#pragma once

#include <fstream>
#include <iostream>

class TimingLog
{	
	public:

		static void init()
		{
			resetTimings();
		}

		static void destroy()
		{
		}

		static void printTimings(const std::string& filename)
		{
			//if(GlobalAppState::get().s_timingsDetailledEnabled)
			{
				std::ofstream outFile;
				bool logToFile = false;
				if (!filename.empty()) {
					logToFile = true;
					outFile.open(filename, std::ios::out);
				}
				std::ostream &out = (logToFile ? outFile : std::cout);


				out << "Total Time SIFT Detection: " << timeSiftDetection << " ms" << std::endl;
				if (countSiftDetection > 0) out << "Time SIFT Detection Per Frames: " << timeSiftDetection / countSiftDetection << " ms (total " << countSiftDetection << ")" << std::endl;
				out << "Total Time SIFT Matching: " << timeSiftMatching << " ms" << std::endl;
				if (countSiftMatching > 0) out << "Time SIFT Matching: " << timeSiftMatching / countSiftMatching << " ms (total " << countSiftMatching << ")" << std::endl;
				out << std::endl;

				out << "Total Time Key Point Match Filter: " << timeKeyPointMatchFilter << " ms" << std::endl;
				if (countKeyPointMatchFilter > 0) out << "Time Key Point Match Filter Per 2Frames: " << timeKeyPointMatchFilter / countKeyPointMatchFilter << " ms (total " << countKeyPointMatchFilter << ")" << std::endl;
				out << "Total Time Surface Area Filter: " << timeSurfaceAreaFilter << " ms" << std::endl;
				if (countSurfaceAreaFilter > 0) out << "Time Surface Area Filter Per 2Frames: " << timeSurfaceAreaFilter / countSurfaceAreaFilter << " ms (total " << countSurfaceAreaFilter << ")" << std::endl;
				out << "Total Time Dense Verify Filter: " << timeDenseVerifyFilter << " ms" << std::endl;
				if (countDenseVerifyFilter > 0) out << "Time Dense Verify Filter Per 2Frames: " << timeDenseVerifyFilter / countDenseVerifyFilter << " ms (total " << countDenseVerifyFilter << ")" << std::endl;
				out << std::endl;

				out << "Total Time Fuse To Global Key: " << timeFuseToGlobalKey << " ms" << std::endl;
				if (countFuseToGlobalKey > 0) out << "Time Fuse To Global Key Per Frame: " << timeFuseToGlobalKey / countFuseToGlobalKey << " ms (total " << countFuseToGlobalKey << ")" << std::endl;
				out << std::endl;

				out << "Total Time Solve: " << timeSolveSparseBundling << " ms" << std::endl;
				if (countSolveSparseBundling > 0) out << "Time Solver Per NonLinear Iteration: " << timeSolveSparseBundling / countSolveSparseBundling << " ms (total " << countSolveSparseBundling << ")" << std::endl;
				out << std::endl;

				out << std::endl;
			}
		}

		static void resetTimings()
		{
			timeKeyPointMatchFilter = 0.0f;
			countKeyPointMatchFilter = 0;

			timeSurfaceAreaFilter = 0.0f;
			countSurfaceAreaFilter = 0;

			timeDenseVerifyFilter = 0.0f;
			countDenseVerifyFilter = 0;

			timeSiftDetection = 0.0f;
			countSiftDetection = 0;

			timeSiftMatching = 0.0f;
			countSiftMatching = 0;

			timeFuseToGlobalKey = 0.0f;
			countFuseToGlobalKey = 0;

			timeSolveSparseBundling = 0.0f;
			countSolveSparseBundling = 0;
		}

		static double timeSiftDetection;
		static unsigned int countSiftDetection;

		static double timeSiftMatching;
		static unsigned int countSiftMatching;

		static double timeKeyPointMatchFilter;
		static unsigned int countKeyPointMatchFilter;

		static double timeSurfaceAreaFilter;
		static unsigned int countSurfaceAreaFilter;

		static double timeDenseVerifyFilter;
		static unsigned int countDenseVerifyFilter;

		static double timeFuseToGlobalKey;
		static unsigned int countFuseToGlobalKey;

		static double timeSolveSparseBundling;
		static unsigned int countSolveSparseBundling;
};
