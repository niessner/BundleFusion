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
		double timeMisc;
		double timeSolve;
		unsigned int numItersSolve;

		double timeSensorProcess; // copy to gpu/resize/etc with input
		double timeReIntegrate;
		double timeReconstruct;
		double timeVisualize;

		FrameTiming() {
			timeSiftDetection = 0;
			timeSiftMatching = 0;
			timeMatchFilterKeyPoint = 0;
			timeMatchFilterSurfaceArea = 0;
			timeMatchFilterDenseVerify = 0;
			timeMisc = 0;
			timeSolve = 0;
			numItersSolve = 0;

			timeSensorProcess = 0;
			timeReIntegrate = 0;
			timeReconstruct = 0;
			timeVisualize = 0;
		}

		void print(std::ostream* out, bool printDepthSensing) {
			*out << "\tTime SIFT Detection: " << std::to_string(timeSiftDetection) << "ms" << std::endl;
			*out << "\tTime SIFT Matching: " << std::to_string(timeSiftMatching) << "ms" << std::endl;
			*out << "\tTime Match Filter Key Point: " << std::to_string(timeMatchFilterKeyPoint) << "ms" << std::endl;
			*out << "\tTime Match Filter Surface Area: " << std::to_string(timeMatchFilterSurfaceArea) << "ms" << std::endl;
			*out << "\tTime Match Filter Dense Verify: " << std::to_string(timeMatchFilterDenseVerify) << "ms" << std::endl;
			*out << "\tTime Misc: " << std::to_string(timeMisc) << "ms" << std::endl;
			*out << "\tTime Solve: " << std::to_string(timeSolve) << "ms" << std::endl;
			*out << "\t#iters solve: " << std::to_string(numItersSolve) << std::endl;
			if (printDepthSensing) {
				*out << "\tTime Process Input: " << std::to_string(timeSensorProcess) << "ms" << std::endl;
				*out << "\tTime Re-Integrate: " << std::to_string(timeReIntegrate) << "ms" << std::endl;
				*out << "\tTime Reconstruct: " << std::to_string(timeReconstruct) << std::endl;
				*out << "\tTime Visualize: " << std::to_string(timeVisualize) << std::endl;
			}
		}
	};


	static void init()
	{
		resetTimings();
	}

	static void destroy()
	{
	}

	static void printAllTimings(const std::string& dir = "./timings/")
	{
		if (!util::directoryExists(dir)) util::makeDirectory(dir);
		std::string read;
		if (m_totalFrameTimings.empty())
			read = dir + "timingLog.txt";
		else 
			read = dir + "timingLogPerFrame.txt";
		printTimings(read);
		printExcelTimings(dir + "excel");
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

			if (!m_globalFrameTimings.empty()) {
				out << "Global Timings Per Frame:" << std::endl;
				for (unsigned int i = 0; i < m_globalFrameTimings.size(); i++) {
					out << "[ frame " << i << " ]" << std::endl;
					m_globalFrameTimings[i].print(&out, false);
				}
				out << std::endl << std::endl;
			}

			if (!m_localFrameTimings.empty()) {
				out << "Local Timings Per Frame:" << std::endl;
				for (unsigned int i = 0; i < m_localFrameTimings.size(); i++) {
					out << "[ frame " << i << " ]" << std::endl;
					m_localFrameTimings[i].print(&out, true);
				}
				out << std::endl << std::endl;
			}

			if (!m_totalFrameTimings.empty()) {
				out << "Total Timings Per Frame:" << std::endl;
				for (unsigned int i = 0; i < m_totalFrameTimings.size(); i++) {
					out << "[ frame " << i << " ] " << m_totalFrameTimings[i] << " ms" << std::endl;
				}
				out << std::endl << std::endl;
			}
		}
	}

	static void printExcelTimings(std::ofstream* out, const std::string& separator, std::vector<FrameTiming>& frameTimings, bool printDepthSensing)
	{
		*out << "SIFT Detection";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeSiftDetection;
		*out << std::endl;
		*out << "SIFT Matching";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeSiftMatching;
		*out << std::endl;
		*out << "Match Filter Key Point";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeMatchFilterKeyPoint;
		*out << std::endl;
		*out << "Match Filter Surface Area";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeMatchFilterSurfaceArea;
		*out << std::endl;
		*out << "Match Filter Dense Verify";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeMatchFilterDenseVerify;
		*out << std::endl;
		*out << "Misc";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeMisc;
		*out << std::endl;
		*out << "Solve";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeSolve;
		*out << std::endl;
		*out << "Solve #Iters";
		for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].numItersSolve;
		*out << std::endl;

		if (printDepthSensing) {
			*out << "Process Input";
			for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeSensorProcess;
			*out << std::endl;
			*out << "Re-Integrate";
			for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeReIntegrate;
			*out << std::endl;
			*out << "Reconstruct";
			for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeReconstruct;
			*out << std::endl;
			*out << "Visualize";
			for (unsigned int i = 0; i < frameTimings.size(); i++) *out << separator << frameTimings[i].timeVisualize;
			*out << std::endl;
		}
	}

	static void printAverages(std::ofstream* out, const std::string& separator, std::vector<FrameTiming>& frameTimings, bool printDepthSensing)
	{
		double sum = 0.0f; unsigned int count = 0;
		*out << "Average times:" << std::endl;
		for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += frameTimings[i].timeSiftDetection; count++; }
		*out << "SIFT Detection" << separator << (sum / count) << separator << count << std::endl;
		sum = 0.0f; count = 0;
		for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += frameTimings[i].timeSiftMatching; count++; }
		*out << "SIFT Matching" << separator << (sum / count) << separator << count << std::endl;
		sum = 0.0f; count = 0;
		for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += (frameTimings[i].timeMatchFilterKeyPoint + frameTimings[i].timeMatchFilterSurfaceArea + frameTimings[i].timeMatchFilterDenseVerify); count++; }
		*out << "Corr Filter" << separator << (sum / count) << separator << count << std::endl;
		sum = 0.0f; count = 0;
		for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += frameTimings[i].timeMisc; count++; }
		*out << "Misc" << separator << (sum / count) << separator << count << std::endl;
		sum = 0.0f; count = 0;
		for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += frameTimings[i].timeSolve; count++; }
		*out << "Solve" << separator << (sum / count) << separator << count << std::endl;

		if (printDepthSensing) {
			sum = 0.0f; count = 0;
			for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += frameTimings[i].timeReIntegrate; count++; }
			*out << "Re-Integrate" << separator << (sum / count) << separator << count << std::endl;
			sum = 0.0f; count = 0;
			for (unsigned int i = 0; i < frameTimings.size(); i++) { sum += (frameTimings[i].timeSensorProcess + frameTimings[i].timeReconstruct + frameTimings[i].timeVisualize); count++; }
			*out << "Misc" << separator << (sum / count) << separator << count << std::endl;
		}
		*out << std::endl << std::endl;
	}

	static void printExcelTimings(const std::string& prefix)
	{
		const std::string separator = ",";

		if (!m_globalFrameTimings.empty()) {
			const std::string globalFile = prefix + "_global.txt";
			std::ofstream out(globalFile);
			printAverages(&out, separator, m_globalFrameTimings, false);
			printExcelTimings(&out, separator, m_globalFrameTimings, false);
			out.close();
		}
		if (!m_localFrameTimings.empty()) {
			const std::string localFile = prefix + "_local.txt";
			std::ofstream out(localFile);
			printAverages(&out, separator, m_localFrameTimings, true);
			printExcelTimings(&out, separator, m_localFrameTimings, true);
			out.close();
		}
		if (!m_totalFrameTimings.empty()) {
			const std::string totalFile = prefix + "_total.txt";
			std::ofstream out(totalFile);
			out << "Per Frame Timings";
			for (unsigned int i = 0; i < m_totalFrameTimings.size(); i++)
				out << separator << m_totalFrameTimings[i];
			out.close();
		}
	}

	static void printCurrentLocalFrame()
	{
		if (m_localFrameTimings.empty()) return;

		std::ostream &out = std::cout;
		out << "[ frame " << m_localFrameTimings.size() - 1 << " ]" << std::endl;
		m_localFrameTimings.back().print(&out, true);
	}

	static void printCurrentGlobalFrame()
	{
		if (m_globalFrameTimings.empty()) return;

		std::ostream &out = std::cout;
		out << "[ frame " << m_globalFrameTimings.size() - 1 << " ]" << std::endl;
		m_globalFrameTimings.back().print(&out, false);
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
