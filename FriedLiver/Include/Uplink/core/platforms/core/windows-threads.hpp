#include "windows-threads.h"

namespace uplink { namespace platform {

//------------------------------------------------------------------------------

inline bool
IsCurrentThread (const Thread& thread)
{
    return GetCurrentThread() == thread;
}

//------------------------------------------------------------------------------

inline Condition*
ConditionCreate()
{
    Condition* condition = new Condition;

    InitializeConditionVariable(condition);

    return condition;
}

inline void
ConditionWait (Condition* condition, Mutex* mutex)
{
    SleepConditionVariableCS(condition, mutex, INFINITE);
}

inline void
ConditionSignal (Condition* condition)
{
    WakeConditionVariable(condition);
}

inline void
ConditionBroadcast (Condition* condition)
{
    WakeAllConditionVariable(condition);
}

inline void
ConditionDestroy(Condition* condition)
{
    delete condition;
}

//------------------------------------------------------------------------------

typedef void* (*ThreadFunction)(void*);

struct ThreadState
{
    ThreadFunction func;
    void*          state;
};

inline DWORD WINAPI
threadMain (LPVOID lpParam)
{
    ThreadState* ts = reinterpret_cast<ThreadState*>(lpParam);

    void* ret = ts->func(ts->state);

    delete ts;

    return reinterpret_cast<DWORD>(ret);
}

//------------------------------------------------------------------------------

inline void
ThreadSetPriority (Thread * thread, ThreadPriority priority)
{
    assert(false); // FIXME: Implement.
}

inline void
ThreadStart (Thread* thread, void* (*func) (void*), void* state)
{
    ThreadState * ts = new ThreadState;
    ts->func = func;
    ts->state = state;

    DWORD tid;

    *thread = CreateThread (
        0, // Security attributes
        0, // Stack size
        threadMain,
        ts,
        0, // Could use CREATE_SUSPENDED here
        &tid
    );
}

inline void
ThreadSleep (float seconds)
{
    Sleep((int)(seconds * 1000 + 0.5));
}

//------------------------------------------------------------------------------

inline Mutex*
CreateMutexObject()
{
    Mutex* mutex = new Mutex;

    InitializeCriticalSection(mutex);

    return mutex;
}

inline void
DestroyMutexObject(Mutex* mutex)
{
    DeleteCriticalSection(mutex);

    delete mutex;
}

inline bool
MutexTryLock(Mutex* mutex)
{
    return FALSE != TryEnterCriticalSection(mutex);
}

inline void
MutexLock (Mutex* mutex)
{
    EnterCriticalSection(mutex);
}

inline void
MutexUnlock (Mutex* mutex)
{
    LeaveCriticalSection(mutex);
}

//------------------------------------------------------------------------------

} }
