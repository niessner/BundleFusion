//
//  core/config.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

//------------------------------------------------------------------------------

// Set this to 1 if you feel like slowing things down a lot and bloating the console with a lot of debugging info.

# ifndef   UPLINK_DEBUG
#   define UPLINK_DEBUG 0
# endif

//------------------------------------------------------------------------------

// Set this to 1 you are willing to add extra profiling code and pay its performance price tag.

# ifndef   UPLINK_PROFILE
#   define UPLINK_PROFILE 0
# endif

//------------------------------------------------------------------------------

// Set this to 1 if you are on windows. Stay away from it otherwise, for now.

//# ifndef   UPLINK_HAS_DESKTOP_UI
//#   define UPLINK_HAS_DESKTOP_UI 1
//# endif

//------------------------------------------------------------------------------

// Set this to 1 if you're building Uplink in a context that provides the Eigen library headers.

# ifndef   UPLINK_HAS_EIGEN
#   define UPLINK_HAS_EIGEN 0
# endif

//------------------------------------------------------------------------------

// Uplink can serialize primitive types using (big endian) network byte order.
// Since the current set of supported architectures is completely little endian, we use native ordering instead.
// Set this to 0 if you feel adventurous about running Uplink on big endian processors.

# ifndef   UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
#   define UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER 1
# endif
