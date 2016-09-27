//
//  network/backends/bsd-sockets.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include <sys/socket.h>
# include <sys/types.h>
# include <netinet/in.h>

# define UPLINK_HAS_BSD_SOCKETS 1

namespace uplink {

//------------------------------------------------------------------------------

struct BSDSocket
{
    bool  enableOption (int option);
    bool setTimeOption (int option, int sec, int usec);

    int         fd;
    sockaddr_in addr;
};

//------------------------------------------------------------------------------

class BSDTCPConnection : public TCPConnection
{
public:
    BSDTCPConnection (int descriptor);

public:
    virtual ~BSDTCPConnection ();

public:
    virtual bool  read (      Byte* bytes, Size size);
    virtual bool write (const Byte* bytes, Size size);

private:
    BSDSocket socket;
};

//------------------------------------------------------------------------------

struct BSDUDPBroadcaster : UDPBroadcaster
{
    static UDPBroadcaster* create (int port);

    bool broadcast (const Byte* bytes, Size length);

    BSDSocket socket;
};

//------------------------------------------------------------------------------

struct BSDUDPListener : UDPListener
{
    static UDPListener* create (int port);

    bool receive (Byte* bytes, Size size, NetworkAddress& sender);

    BSDSocket socket;
};

//------------------------------------------------------------------------------

}
