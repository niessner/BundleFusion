# pragma once

# include "./types.h"
# include <vector>

namespace uplink {

#define UPLINK_PROFILER_TASKS() \
    UPLINK_PROFILER_TASK(TCPConnection) \
    UPLINK_PROFILER_TASK(ServicePublishing) \
    UPLINK_PROFILER_TASK(ReadMessage) \
    UPLINK_PROFILER_TASK(CompressImage) \
    UPLINK_PROFILER_TASK(DecompressImage) \
    UPLINK_PROFILER_TASK(SocketReadLoop) \
    UPLINK_PROFILER_TASK(WriteMessage) \
    UPLINK_PROFILER_TASK(SocketWriteLoop) \
    UPLINK_PROFILER_TASK(EncodeImage) \
    UPLINK_PROFILER_TASK(DecodeImage)

enum ProfilerTask {
    ProfilerTask_Invalid = -1,

# define UPLINK_PROFILER_TASK(Name) \
    ProfilerTask_##Name,
     UPLINK_PROFILER_TASKS()
# undef  UPLINK_PROFILER_TASK

    ProfilerTask_HowMany
};

void profiler_task_started (ProfilerTask task);
void profiler_task_stopped (ProfilerTask task);

struct ScopedProfiledTask
{
    ScopedProfiledTask (ProfilerTask task) : task(task)
    {
# if UPLINK_PROFILE
        profiler_task_started(task);
#endif
    }
    ~ScopedProfiledTask ()
    {
# if UPLINK_PROFILE
        profiler_task_stopped(task);
# endif
    }

    ProfilerTask task;
};

struct Profiler
{
    Profiler ()
    {
# if UPLINK_PROFILE
        taskIds.reserve(ProfilerTask_HowMany);
# endif
    }

    virtual ~Profiler () {}

    virtual void registerTasks ()
    {
        if (!UPLINK_PROFILE)
            return;

#   define UPLINK_PROFILER_TASK(Name) \
        taskIds.push_back(registerTask(#Name, #Name));
         UPLINK_PROFILER_TASKS()
#   undef  UPLINK_PROFILER_TASK
    }

    virtual int registerTask (CString name, CString label) = 0;

    virtual void taskStarted (int identifier) = 0;
    virtual void taskStopped (int identifier) = 0;

    std::vector<int> taskIds;
};

} // uplink namespace
