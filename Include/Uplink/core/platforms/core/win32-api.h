#pragma once

// This monstrosity exists because the sequence of inclusion of windows main and socket headers actually matters.

# ifndef _WINDOWS_ // Then no other header already included Windows.h, yet.
#   define NOMINMAX
#   define VC_EXTRALEAN
#   define WIN32_LEAN_AND_MEAN
#   include <WinSock2.h>
#   include <Windows.h>
#   include <Ws2tcpip.h>
#   pragma comment(lib, "ws2_32.lib")
# else // We're in trouble, because WinSock2.h cannot be included from here.
#   ifndef _WINSOCK2API_
#       error "The Windows.h header has already been included without WinSock2 support. You must include WinSock2.h before it. See: http://msdn.microsoft.com/en-us/library/windows/desktop/ms738562.aspx."
#   endif
# endif
