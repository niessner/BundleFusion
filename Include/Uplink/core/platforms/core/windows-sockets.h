//
//  network/backends/windows-sockets.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./win32-api.h"

# define UPLINK_HAS_WINDOWS_SOCKETS 1

namespace uplink {

//------------------------------------------------------------------------------

struct WindowsSocket
{
    WindowsSocket ()
    {

    }

    struct DLL
    {
        DLL ()
        {
            version = MAKEWORD(2, 2);

            report_unless("Windows socket DLL failed to load.", 0 == WSAStartup(version, &data));
        }

        ~DLL ()
        {
            WSACleanup();
        }

        static void require ()
        {
            static const DLL dll;
        }

        WORD    version;
        WSADATA data;
    };

    bool  enableOption (int option);
    bool setTimeOption (int option, int msec);

    SOCKET        fd;
    sockaddr_in addr;
};

//------------------------------------------------------------------------------

class WindowsTCPConnection : public TCPConnection
{
public:
    WindowsTCPConnection (int descriptor);

public:
    virtual ~WindowsTCPConnection ();

public:
    virtual bool  read (      Byte* bytes, Size size);
    virtual bool write (const Byte* bytes, Size size);

private:
    WindowsSocket socket;
};

//------------------------------------------------------------------------------

struct WindowsUDPBroadcaster : UDPBroadcaster
{
    virtual bool broadcast (const Byte* bytes, Size length);

    WindowsSocket socket;
};

//------------------------------------------------------------------------------

struct WindowsUDPListener : UDPListener
{
    static UDPListener* create (int port);

    virtual bool receive (Byte* bytes, Size size, NetworkAddress& sender);

    WindowsSocket socket;
};

//------------------------------------------------------------------------------

}
