# pragma once

#ifdef UPLINK_HAS_DESKTOP_UI

# include "./core/types.h"

namespace uplink {

struct DesktopServerUI
{
    DesktopServerUI();
    ~DesktopServerUI();

    void run ();

    void setColorImage(const uint8*  buffer, int width, int height);
    void setDepthImage(const uint16* buffer, int width, int height);

    struct Impl;
    Impl*  impl;
};

} // uplink namespace

# include "./desktop-ui.hpp"
#endif
