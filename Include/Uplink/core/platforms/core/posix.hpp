//
//  core/platforms/posix.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./posix.h"
# include "../../macros.h"
# include <cassert>
# include <memory>
# include <fcntl.h>
# include <errno.h>
# include <cstring>
# include <unistd.h>

namespace uplink {

//------------------------------------------------------------------------------

inline int
vasprintf (char** output, const char* format, va_list args)
{
    return ::vasprintf(output, format, args);
}

# if !UPLINK_HAS_OBJC
inline void
console_log_line (Verbosity verbosity, CString message)
{
    printf("%s%s\n", log_prefix(verbosity), message);
}
# endif

inline String
getLocalHostName ()
{
    char name [256];

    if (0 != gethostname(name, 255))
        return "localhost";

    name[255] = '\0';

    return name;
}

# if !UPLINK_HAS_APPLE
inline int64
getTickCount ()
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);

    return int64(t.tv_sec * 1000000000 + t.tv_nsec);
}

inline double
getTickFrequency ()
{
    return 1e9;
}
#endif

//------------------------------------------------------------------------------

}

# include "./bsd-sockets.hpp"
# include "./pthreads.hpp"
