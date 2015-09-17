#pragma once

#include <fstream>
#include <iostream>

class TimingLog
{
public:
	struct FrameTiming {
		double timeSiftDetection; // fuse for global / sift detection for local
		double timeSiftMatching;
		double timeMatchFilterKeyPoint;
		double timeMatchFilterSurfaceArea;
		double timeMatchFilterDenseVerify;
		double timeFilterFrames;
		double timeAddCurrResiduals;
		double timeSolve;
		unsigned int numItersSolve;

		FrameTiming() {
			timeSiftDetection = 0;
			timeSiftMatching = 0;
			timeMatchFilterKeyPoint = 0;
			timeMatchFilterSurfaceArea = 0;
			timeMatchFilterDenseVerify = 0;
			timeFilterFrames = 0;
			timeAddCurrResiduals = 0;
			timeSolve = 0;
			numItersSolve = 0;
		}

		void print(std::ostream* out) {
			*out << "\tTime SIFT Detection: " << std::to_string(timeSiftDetection) << "ms" << std::endl;
			*out << "\tTime SIFT Matching: " << std::to_string(timeSiftMatching) << "ms" << std::endl;
			*out << "\tTime Match Filter Key Point: " << std::to_string(timeMatchFilterKeyPoint) << "ms" << std::endl;
			*out << "\tTime Match Filter Surface Area: " << std::to_string(timeMatchFilterSurfaceArea) << "ms" << std::endl;
			*out << "\tTime Match Filter Dense Verify: " << std::to_string(timeMatchFilterDenseVerify) << "ms" << std::endl;
			*out << "\tTime Filter Frames: " << std::to_string(timeFilterFrames) << "ms" << std::endl;
			*out << "\tTime Add Curr Residuals: " << std::to_string(timeAddCurrResiduals) << "ms" << std::endl;
			*out << "\tTime Solve: " << std::to_string(timeSolve) << "ms" << std::endl;
			*out << "\t#iters solve: " << std::to_string(numItersSolve) << std::endl;
		}
	};


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
			
			out << "Global Timings Per Frame:" << std::endl;
			for (unsigned int i = 0; i < m_globalFrameTimings.size(); i++) {
				out << "[ frame " << i << " ]" << std::endl;
				m_globalFrameTimings[i].print(&out);
			}
			out << std::endl << std::endl;

			out << "Local Timings Per Frame:" << std::endl;
			for (unsigned int i = 0; i < m_localFrameTimings.size(); i++) {
				out << "[ frame " << i << " ]" << std::endl;
				m_localFrameTimings[i].print(&out);
			}

			if (m_totalFrameTimings.size() > 0) out << "Total Timings Per Frame:" << std::endl;
			for (unsigned int i = 0; i < m_totalFrameTimings.size(); i++) {
				out << "[ frame " << i << " ] " << m_totalFrameTimings[i] << " ms" << std::endl;
			}

			out << std::endl;
			out << std::endl;
		}
	}

	static void printCurrentLocalFrame() 
	{
		if (m_localFrameTimings.empty()) return;

		std::ostream &out = std::cout;
		out << "[ frame " << m_localFrameTimings.size() - 1 << " ]" << std::endl;
		m_localFrameTimings.back().print(&out);
	}

	static void printCurrentGlobalFrame()
	{
		if (m_globalFrameTimings.empty()) return;

		std::ostream &out = std::cout;
		out << "[ frame " << m_globalFrameTimings.size() - 1 << " ]" << std::endl;
		m_globalFrameTimings.back().print(&out);
	}

	static void resetTimings()
	{
		m_localFrameTimings.clear();
		m_globalFrameTimings.clear();
		m_totalFrameTimings.clear();
	}

	static void addLocalFrameTiming()
	{
		m_localFrameTimings.push_back(FrameTiming());
	}
	static void addGlobalFrameTiming()
	{
		m_globalFrameTimings.push_back(FrameTiming());
	}

	static FrameTiming& getFrameTiming(bool local)
	{ 
		if (local) return m_localFrameTimings.back();
		else return m_globalFrameTimings.back();
	}

	static void addTotalFrameTime(double t)
	{
		m_totalFrameTimings.push_back(t);
	}

private:
	static std::vector<FrameTiming> m_localFrameTimings;
	static std::vector<FrameTiming> m_globalFrameTimings;
	static std::vector<double> m_totalFrameTimings;
};
