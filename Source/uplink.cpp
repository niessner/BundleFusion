#include "stdafx.h"
#include "GlobalAppState.h"

#ifdef STRUCTURE_SENSOR
#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include "../Include/Uplink/uplink.h"

namespace uplink {

Context context;

}

#endif
