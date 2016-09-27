#ifndef CUDATimer_h
#define CUDATimer_h

#include <cuda_runtime.h>
#include <string>
#include <vector>

struct TimingInfo {
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    float duration;
    std::string eventName;
};

/** Copied wholesale from mLib, so nvcc doesn't choke. */
template<class T>
int findFirstIndex(const std::vector<T> &collection, const T &value)
{
    int index = 0;
    for (const auto &element : collection)
    {
        if (element == value)
            return index;
        index++;
    }
    return -1;
}

class CUDATimer {
public:
    std::vector<TimingInfo> timingEvents;
    int currentIteration;

    CUDATimer() : currentIteration(0) {}
    void nextIteration() {
        ++currentIteration;
    }

    void reset() {
        currentIteration = 0;
        timingEvents.clear();
    }

    void startEvent(const std::string& name) {
        TimingInfo timingInfo;
        cudaEventCreate(&timingInfo.startEvent);
        //cudaEventCreate(&timingInfo.endEvent);
		cudaEventCreateWithFlags(&timingInfo.endEvent, cudaEventBlockingSync);
        cudaEventRecord(timingInfo.startEvent);
        timingInfo.eventName = name;
        timingEvents.push_back(timingInfo);
    }

    void endEvent() {
        TimingInfo& timingInfo = timingEvents[timingEvents.size() - 1];
        cudaEventRecord(timingInfo.endEvent, 0);
    }

	void evaluate(bool showSum = false, bool showMax = false) {
		std::vector<std::string> aggregateTimingNames;
		std::vector<float> aggregateTimes;
		std::vector<int> aggregateCounts;
		std::vector<float> maxTimes;
		for (int i = 0; i < timingEvents.size(); ++i) {
			TimingInfo& eventInfo = timingEvents[i];
			cudaEventSynchronize(eventInfo.endEvent);
			cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
			int index = findFirstIndex(aggregateTimingNames, eventInfo.eventName);
			if (index < 0) {
				aggregateTimingNames.push_back(eventInfo.eventName);
				aggregateTimes.push_back(eventInfo.duration);
				aggregateCounts.push_back(1);
				maxTimes.push_back(eventInfo.duration);
			}
			else {
				aggregateTimes[index] = aggregateTimes[index] + eventInfo.duration;
				aggregateCounts[index] = aggregateCounts[index] + 1;
				if (maxTimes[index] < eventInfo.duration)
					maxTimes[index] = eventInfo.duration;
			}

			//if (eventInfo.eventName == "MultiplyDescriptor") {
			//	printf("time %f\n", eventInfo.duration);
			//}
		}
		printf("------------------------------------------------------------\n");
		printf("          Kernel          |   Count  |   Total   | Average \n");
		printf("--------------------------+----------+-----------+----------\n");
		for (int i = 0; i < aggregateTimingNames.size(); ++i) {
			printf("--------------------------+----------+-----------+----------\n");
			printf(" %-24s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames[i].c_str(), aggregateCounts[i], aggregateTimes[i], aggregateTimes[i] / aggregateCounts[i]);
		}
		printf("------------------------------------------------------------\n\n");

		if (showMax) {
			printf("------------------------------------------------------------\n");
			printf("          Kernel          |   Count  |   Total   | Max \n");
			printf("--------------------------+----------+-----------+----------\n");
			for (int i = 0; i < aggregateTimingNames.size(); ++i) {
				printf("--------------------------+----------+-----------+----------\n");
				printf(" %-24s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames[i].c_str(), aggregateCounts[i], aggregateTimes[i], maxTimes[i]);
			}
			printf("------------------------------------------------------------\n\n");
		}
		if (showSum) {
			int sumCount = 0;
			float sumAvg = 0.0f;
			for (unsigned int i = 0; i < aggregateTimingNames.size(); i++) {
				sumCount += aggregateCounts[i];
				sumAvg += aggregateTimes[i] / aggregateCounts[i];
			}
			printf("\n");
			printf("-----------------------------------------------------\n");
			printf("          TOTAL           |   Count  |   Total/Run   \n");
			printf("--------------------------+----------+---------------\n");
			printf(" %-24s |   %4d   | %8.3fms\n", "TOTAL Avg", sumCount, sumAvg);
			printf("-----------------------------------------------------\n");
		}
    }
};

#endif