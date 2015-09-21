//
//  core/platforms/posix.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./pthreads.h"
# include "./bsd-sockets.h"

# define UPLINK_HAS_POSIX 1

namespace uplink {

//------------------------------------------------------------------------------

int vasprintf (char** output, const char* format, va_list args);

String getLocalHostName ();

int64 getTickCount ();

double getTickFrequency ();

//------------------------------------------------------------------------------

}
