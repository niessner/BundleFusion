//
//  system/threads.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./threads.h"
# include <cassert>

namespace uplink {

//------------------------------------------------------------------------------

inline void
Thread::start ()
{
    const MutexLocker lock (mutex);

    if (state != Stopped)
        return;

    state = Starting;

    platform::ThreadStart(&handle, main, this);
}

inline void
Thread::stop ()
{
    const MutexLocker lock (mutex);

    if (state != Running)
        return;

    state = Stopping;

    notify();
}

inline void
Thread::join ()
{
    assert(!isCurrentThread()); // A thread cannot join itself.

    while (!isStopped())
    {
        stop();
        sleep(.001f);
    }
}

inline void
Thread::wait ()
{
    condition.wait(&mutex);
}

inline void
Thread::notify ()
{
    condition.broadcast();
}

inline void
Thread::setPriority (Priority priority)
{
    switch (priority)
    {
    case PriorityLow   : return platform::ThreadSetPriority(&handle, ThreadPriorityLow);
    case PriorityNormal: return platform::ThreadSetPriority(&handle, ThreadPriorityNormal);
    case PriorityHigh  : return platform::ThreadSetPriority(&handle, ThreadPriorityHigh);
    }
}

inline void*
Thread::main (void* data)
{
    Thread* that = reinterpret_cast<Thread*>(data);

# if UPLINK_HAS_APPLE
    if (0 != that->name)
        pthread_setname_np(that->name);
# endif

    {
        const MutexLocker lock (that->mutex);

        that->state = Running;
    }

    that->run();

    {
        const MutexLocker lock (that->mutex);

        that->state = Stopped;
    }

    return 0;
}

//------------------------------------------------------------------------------

}
