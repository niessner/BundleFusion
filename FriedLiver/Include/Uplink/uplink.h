//
//  uplink.h
//  Uplink
//
//  Copyright (c) 2014 Occipital. All rights reserved.
//

# pragma once

//------------------------------------------------------------------------------

// Uplink 1.2

# define UPLINK_SERVER_MINIMUM_VERSION_MAJOR 1
# define UPLINK_SERVER_MINIMUM_VERSION_MINOR 2
# define UPLINK_SERVER_MAXIMUM_VERSION_MAJOR 1
# define UPLINK_SERVER_MAXIMUM_VERSION_MINOR 2

# define UPLINK_CLIENT_VERSION_MAJOR 1
# define UPLINK_CLIENT_VERSION_MINOR 2

//------------------------------------------------------------------------------

# include "./core/platform.h"

//------------------------------------------------------------------------------

# include "./context.h"
# include "./clients.h"
# include "./image.h"
# include "./image-codecs.h"
# include "./camera-calibration.h"
# include "./camera-fixedparams.h"
# include "./camera-pose.h"
# include "./camera-frame.h"
# include "./endpoints.h"
# include "./discovery.h"
# include "./messages.h"
# include "./motion.h"
# include "./servers.h"
# include "./desktop-server.h"
# include "./services.h"
# include "./sessions-settings.h"
# include "./sessions-setup.h"
# include "./sessions-setup-presets.h"

//------------------------------------------------------------------------------

// OSX's AssertMacros.h header defines the 'check' macro, which other headers (ie: Qt) may not like.
# ifdef check
#   undef check
# endif
