//
//  system/threads.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

namespace uplink {

//------------------------------------------------------------------------------

struct Mutex
{
     Mutex () : handle(platform::CreateMutexObject()) {}
    ~Mutex () { platform::DestroyMutexObject(handle); }
    
    bool tryLock () { return platform::MutexTryLock(handle); }

    void    lock () { platform::MutexLock(   handle); }
    void  unlock () { platform::MutexUnlock( handle); }
     
    platform::Mutex* handle;
};

//------------------------------------------------------------------------------

class MutexLocker
{
public:
     MutexLocker (Mutex& mutex) : mutex(mutex) { mutex.lock(); }
    ~MutexLocker ()                            { mutex.unlock(); }

private:
    Mutex& mutex;
};

typedef MutexLocker Locked;

//------------------------------------------------------------------------------

struct Condition
{
     Condition () : handle(platform::ConditionCreate()) {}
    ~Condition () { platform::ConditionDestroy(handle); }

    void wait (Mutex* mutex)
    {
        const MutexLocker lock(*mutex);

        platform::ConditionWait(handle, mutex->handle);
    }

    void signal ()
    {
        platform::ConditionSignal(handle);
    }

    void broadcast ()
    {
        platform::ConditionBroadcast(handle);
    }

    platform::Condition* handle;
};

//------------------------------------------------------------------------------

struct Thread
{
public:    
    enum Priority
    {
        PriorityLow,
        PriorityNormal,
        PriorityHigh
    };

public:
    explicit Thread (CString name = 0)
    : state(Stopped)
    , name(name)
    {

    }

    virtual ~Thread ()
    {
        if (!isCurrentThread())
            join();
    }

public:
     virtual void run () = 0;

public:
    void start  ();
    void stop   ();
    void join   ();
    void wait   ();
    void notify ();

    void setPriority (Priority priority);
    
    bool isCurrentThread () const { return platform::IsCurrentThread(handle); }

    enum State { Stopped, Starting, Running, Stopping };

    State getState () const { const MutexLocker lock (mutex); return state; }

    bool isStarting  () const { return getState() == Starting; }
    bool isRunning   () const { return getState() == Running ; }
    bool isStopping  () const { return getState() == Stopping; }
    bool isStopped   () const { return getState() == Stopped ; }

public:
    static void  sleep (float seconds) { platform::ThreadSleep(seconds); }

private:
    static void* main (void* data);

private:
    platform::Thread  handle;
    mutable Mutex     mutex;
    mutable Condition condition;
    State             state;
    const CString     name;
};

//------------------------------------------------------------------------------

}

# include "./threads.hpp"
