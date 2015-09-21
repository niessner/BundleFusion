//
//  core/platforms/windows.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./windows.h"

namespace uplink {

//------------------------------------------------------------------------------

inline void
platform_startup ()
{
    WindowsSocket::DLL::require();
}

inline void
platform_shutdown ()
{

}

//------------------------------------------------------------------------------

}

# include "./core/windows-sockets.hpp"
# include "./core/windows-threads.hpp"
# include "./core/windows-image-codecs.hpp"
