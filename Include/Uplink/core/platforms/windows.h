//
//  core/platforms/windows.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/win32-api.h"
# include "./core/windows-sockets.h"
# include "./core/windows-threads.h"
# include "./core/windows-image-codecs.h"
# include <cstdio>
# include <cstdarg>
# include <cstdlib>

# define UPLINK_HAS_WINDOWS 1

namespace uplink {

//------------------------------------------------------------------------------

inline int
vasprintf (char** output, const char* format, va_list args)
{
    *output = 0;
    const int length = _vscprintf(format, args);

    if (length < 0)
        return 0;

    *output = reinterpret_cast<char*>(malloc(length + 1));

    if (0 == *output)
        return 0;

    vsnprintf_s(*output, length + 1, length, format, args);

    // Output is null-terminated.

    return length;
}

inline void
console_log_line (Verbosity verbosity, CString message)
{
     CStringPtr line = formatted("%s%s\n", log_prefix(verbosity), message);

     OutputDebugStringA(line.get());
}

inline String
getLocalHostName ()
{
    char name [256];

    if (0 != gethostname(name, 255))
        return "localhost";

    name[255] = '\0';

    return name;
}

inline int64
getTickCount ()
{
    LARGE_INTEGER counter;

    QueryPerformanceCounter(&counter);

    return int64(counter.QuadPart);
}

inline double
getTickFrequency ()
{
    LARGE_INTEGER frequency;

    QueryPerformanceFrequency(&frequency);

    return double(frequency.QuadPart);
}


}
