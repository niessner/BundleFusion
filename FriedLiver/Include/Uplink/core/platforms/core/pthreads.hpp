//
//  system/backends/pthreads.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.

# include <stdlib.h>
# include <unistd.h>

namespace uplink { namespace platform {

//------------------------------------------------------------------------------

inline bool IsCurrentThread (const Thread& thread)
{
    return pthread_equal(pthread_self(), thread);
}

//------------------------------------------------------------------------------

inline Condition* ConditionCreate()
{
    Condition* condition = (Condition*)malloc(sizeof(Condition));
    pthread_cond_init(condition, NULL);
    return condition;
}

inline void ConditionWait(Condition* condition, Mutex* mutex)
{
    pthread_cond_wait(condition, mutex);
}

inline void ConditionSignal(Condition* condition)
{
    pthread_cond_signal(condition);
}

inline void ConditionBroadcast (Condition* condition)
{
    pthread_cond_broadcast(condition);
}

inline void ConditionDestroy(Condition* condition)
{
    pthread_cond_destroy(condition);
}

//------------------------------------------------------------------------------

inline void ThreadSetPriority(Thread * thread, ThreadPriority priority)
{
    sched_param schedParam;
    int kPolicy = -1;
    pthread_getschedparam(*thread, &kPolicy, &schedParam);

    int priority_to_set;

    switch (priority) {
    case ThreadPriorityLow:
        priority_to_set = sched_get_priority_min( kPolicy );
        break;
    case ThreadPriorityNormal:
        return;
    case ThreadPriorityHigh:
        priority_to_set = sched_get_priority_max( kPolicy );
        break;
    default:
        return;
    }

    schedParam.sched_priority = priority_to_set;
    pthread_setschedparam(*thread, kPolicy, &schedParam);
}

inline void ThreadStart(Thread * thread, void *(*func)(void *), void * state)
{
    // Create the thread using POSIX routines.
    pthread_attr_t  attr;
    int             returnVal;

    returnVal = pthread_attr_init(&attr);

    //assert(!returnVal);

    returnVal = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    /*
    //Priority Setting.
    const int kPolicy = SCHED_RR;
    int minPriority = sched_get_priority_min( kPolicy );

    sched_param schedParam;
    schedParam.sched_priority = minPriority;
    pthread_attr_setschedparam( &attr, &schedParam );
    pthread_attr_setschedpolicy( &attr, kPolicy );
    */

    //assert(!returnVal);

    int threadError = pthread_create(thread, &attr, func, state);

    returnVal = pthread_attr_destroy(&attr);

    if (threadError != 0)  {
        // Report an error.
    }
}

inline void ThreadSleep(float seconds) {
    usleep( (int)(1e6 * seconds) );
}

//------------------------------------------------------------------------------

inline Mutex* CreateMutexObject()  {
    Mutex* mutex = (Mutex*)malloc(sizeof(Mutex));
    pthread_mutex_init(mutex, NULL);
    return mutex;
}

inline void DestroyMutexObject(Mutex* mutex)  {
    pthread_mutex_destroy(mutex);
}

inline bool MutexTryLock(Mutex* mutex) { return (pthread_mutex_trylock(mutex) == 0); }
inline void MutexLock(Mutex* mutex) { pthread_mutex_lock(mutex); }
inline void MutexUnlock(Mutex* mutex) { pthread_mutex_unlock(mutex); }

//------------------------------------------------------------------------------

} }
