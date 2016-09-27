//
//  core/objc.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

// Objective-C helpers.

# pragma once

# include "./objc.h"

namespace uplink {

//------------------------------------------------------------------------------

inline NSString*
toNSString (const String& str)
{
    return [ NSString stringWithUTF8String: str.c_str() ];
}

inline String
fromNSString (NSString* str)
{
    return [ str UTF8String ];
}

inline void
console_log_line (Verbosity verbosity, CString message)
{
    NSLog(@"%s%s\n", log_prefix(verbosity), message);
}

//------------------------------------------------------------------------------

}
