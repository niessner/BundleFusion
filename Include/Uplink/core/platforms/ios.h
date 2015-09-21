//
//  core/platforms/ios.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/apple.h"

# define UPLINK_HAS_IOS 1

# if TARGET_IPHONE_SIMULATOR
#   define UPLINK_HAS_IOS_SIMULATOR 1
# else
#   define UPLINK_HAS_IOS_DEVICE 1
# endif

namespace uplink {

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

}
