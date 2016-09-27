#pragma once

// FIXME: Dynamically compute the table with the camera info data, instead of using this fixed one.

# include "./types.h"

namespace uplink {

uint16 shift2depth (uint16  shift);
void   shift2depth (uint16* buffer, size_t size);

//------------------------------------------------------------------------------

}

#include "shift2depth.hpp"
