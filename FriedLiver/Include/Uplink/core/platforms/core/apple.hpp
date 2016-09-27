//
//  core/platforms/apple.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./apple.h"
# include <mach/mach_time.h>

namespace uplink {

//------------------------------------------------------------------------------

inline int64
getTickCount ()
{
    return int64(mach_absolute_time());
}

inline double
getTickFrequency ()
{
    static double frequency = 0;

    if (frequency != 0)
        return frequency;

    mach_timebase_info_data_t info;
    mach_timebase_info(&info);

    frequency = info.denom * 1e9 / info.numer;

    return frequency;
}

inline void
platform_startup ()
{
    // Nothing to do.
}

inline void
platform_shutdown ()
{
    // Nothing to do.
}

//------------------------------------------------------------------------------

}

# include "./posix.hpp"

# if UPLINK_HAS_OBJC
#   include "./objc.hpp"
# endif

# include "./apple-image-codecs.hpp"
