//
//  core/platforms/apple.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./posix.h"
# include "./apple-image-codecs.h"

# define UPLINK_HAS_APPLE 1

# if defined(__OBJC__)
#   include "./objc.h"
# endif

namespace uplink {

//------------------------------------------------------------------------------

int64  getTickCount ();
double getTickFrequency ();

//------------------------------------------------------------------------------

}
