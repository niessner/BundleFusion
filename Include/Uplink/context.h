//
//  context.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/platform.h"
# include "./core/threads.h"
# include <vector>

namespace uplink {

//------------------------------------------------------------------------------

class Context
{
public:
    Context ()
    : _profiler(0)
    {
        platform_startup();
    }

public:
    ~Context ()
    {
        zero_delete(_profiler);

        platform_shutdown();
    }

public:
    Profiler* profiler () { return _profiler; }

    void setProfiler (Profiler* profiler_)
    {
        assert(0 == _profiler); // Call this once only.

        assert(0 != profiler_);

        _profiler = profiler_;

        _profiler->registerTasks();
    }

public:
    void log_line (Verbosity verbosity, CString message)
    {
        const MutexLocker _(_logging);

        // FIXME: Timestamps.
        // FIXME: Remote log.
        // FIXME: Verbosity filter.

        console_log_line(verbosity, message);
    }

private:
    Profiler*     _profiler;
    Mutex         _logging;
};

//------------------------------------------------------------------------------

// The context singleton instance is defined uniquely in:
//
//     uplink.cpp (C++),
//     uplink.mm (Objective C++).

extern Context context;

//------------------------------------------------------------------------------

// Callbacks

inline void
profiler_task_started (ProfilerTask task)
{
    Profiler* const profiler = context.profiler();

    if (0 == profiler)
        return;

    profiler->taskStarted(profiler->taskIds[task]);
}

inline void
profiler_task_stopped (ProfilerTask task)
{
    Profiler* const profiler = context.profiler();

    if (0 == profiler)
        return;

    profiler->taskStopped(profiler->taskIds[task]);
}

inline void
log_line (Verbosity verbosity, CString message)
{
    context.log_line(verbosity, message);
}

}

# include "./context.hpp"
