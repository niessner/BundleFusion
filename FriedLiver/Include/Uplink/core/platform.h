//
//  core/platform.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

//------------------------------------------------------------------------------

// Platform-independent definitions

# if __cplusplus > 199711L
#   define UPLINK_HAS_CXX11 1
# endif

//------------------------------------------------------------------------------

// Platform-specific workarounds

# if _MSC_VER
#   define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#ifndef _CRT_SECURE_NO_WARNINGS
#   define _CRT_SECURE_NO_WARNINGS 1
#endif
#ifndef _SCL_SECURE_NO_WARNINGS
#   define _SCL_SECURE_NO_WARNINGS 1
#endif
# endif

//------------------------------------------------------------------------------

// Platform-independent API

# include "./config.h"
# include "./macros.h"
# include "./types.h"
# include "./streams.h"
# include "./logging.h"
# include "./profiling.h"
# include "./memory.h"

namespace uplink {

void platform_startup ();
void platform_shutdown ();

// Console output.
void console_log_line (Verbosity verbosity, CString message);

struct NetworkAddress
{
    String toString () const;

    void operator = (uint32_t ipv4_addr);

    uint8_t ipv4 [4];
};

inline std::ostream& operator << (std::ostream& o, const NetworkAddress& addr);

String getLocalHostName ();

class TCPConnection : public DuplexStream
{
public:
             TCPConnection();
    virtual ~TCPConnection();

public:
    static TCPConnection* connect (CString host, uint16 port);
    static TCPConnection* create  (int descriptor);

public:
    bool disconnecting;
};

TCPConnection* TCPConnection_connect (CString host, uint16 port);
TCPConnection* TCPConnection_create  (int descriptor);

struct UDPBroadcaster
{
    virtual ~UDPBroadcaster () {}

    static UDPBroadcaster* create (int port);

    virtual bool broadcast (const Byte* bytes, Size length) = 0;
};

UDPBroadcaster* UDPBroadcaster_create (int port);

struct UDPListener
{
    virtual ~UDPListener () {}

    static UDPListener* create (int port);

    virtual bool receive (Byte* bytes, Size size, NetworkAddress& sender) = 0;
};

UDPListener* UDPListener_create (int port);

int64  getTickCount ();
double getTickFrequency ();

enum ThreadPriority {
    ThreadPriorityLow,
    ThreadPriorityNormal,
    ThreadPriorityHigh
};

//------------------------------------------------------------------------------

// NOTE: JPEG quality values are remapped across platform-specific implementations in order to consistently mimic IJG's libjpeg quality scale.
// Some typical values used at Occipital are:
//     .8162f for Structure app's Uplink color camera frames.
//     .8f for Skanect's Uplink feedback images.
//     .95f for Skanect's on-disk color camera frames.

static const float defaultQuality = .95f;
static const float occQuality = 0.98f;
static const float UseDefaultQuality = -12345.0;

enum graphics_PixelFormat // FIXME: Solve the tension with Image and use a shorter name.
{
    graphics_PixelFormat_Gray,
    graphics_PixelFormat_RGB,
    graphics_PixelFormat_RGBA,
    graphics_PixelFormat_YCbCr,
};

enum graphics_ImageCodec // FIXME: Solve the tension with Image and use a shorter name.
{
    graphics_ImageCodec_JPEG,
    graphics_ImageCodec_PNG,
    graphics_ImageCodec_BMP,
    graphics_ImageCodec_Hardware_JPEG
};

bool
encode_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t*        inputBuffer,
    size_t                inputSize,
    graphics_PixelFormat    inputFormat,
    size_t                inputWidth,
    size_t                inputHeight,
    MemoryBlock&          outputMemoryBlock,
    float                 outputQuality = defaultQuality // ] 0.f .. 1.f [
);

bool
decode_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t*        inputBuffer,
    size_t                inputSize,
    graphics_PixelFormat    outputFormat,
    size_t&               outputWidth,
    size_t&               outputHeight,
    MemoryBlock&          outputMemoryBlock
);

//------------------------------------------------------------------------------

} // uplink namespace

//------------------------------------------------------------------------------

// Platform-specific headers

# if defined(_WIN32)
#   include "./platforms/windows.h"
# elif defined(__APPLE__) && defined(__MACH__)
#   include <TargetConditionals.h>
#   if TARGET_OS_IPHONE
#       include "./platforms/ios.h"
#   elif TARGET_OS_MAC
#       include "./platforms/osx.h"
#   endif
# else
#   error "Unknown system."
# endif

//------------------------------------------------------------------------------

// Platform-specific API

namespace uplink {

namespace platform {

# if !defined(UPLINK_THREAD) || !defined(UPLINK_MUTEX) || !defined(UPLINK_CONDITION)
#   error "Threads backend missing."
# endif

typedef UPLINK_THREAD    Thread;
typedef UPLINK_MUTEX     Mutex;
typedef UPLINK_CONDITION Condition;

bool IsCurrentThread (const Thread& thread);

// Conditions
Condition* ConditionCreate();
void ConditionWait(Condition* condition, Mutex* mutex);
void ConditionSignal(Condition* condition);
void ConditionBroadcast (Condition* condition);
void ConditionDestroy(Condition* condition);

// Create/Destroy thread.
void ThreadStart(Thread*, void* (*) (void*), void*);
void ThreadSleep(float seconds);
void ThreadSetPriority(Thread* thread, ThreadPriority priority);

// Create/Destroy mutex.
Mutex* CreateMutexObject();
void DestroyMutexObject(Mutex*);

// Locking/unlocking.
bool MutexTryLock(Mutex*);
void MutexLock(Mutex*);
void MutexUnlock(Mutex*);

} // platform namespace

} // uplink namespace

//------------------------------------------------------------------------------

// Platform-specific implementations

# if UPLINK_HAS_WINDOWS
#   include "./platforms/windows.hpp"
# elif UPLINK_HAS_IOS
#   include "./platforms/ios.hpp"
# elif UPLINK_HAS_OSX
#   include "./platforms/osx.hpp"
# endif

//------------------------------------------------------------------------------

// Platform-dependent macros

#if UPLINK_HAS_OBJC
#   define objc_weak __weak
#else
#   define objc_weak
#endif

//------------------------------------------------------------------------------

#include "./platform.hpp"
