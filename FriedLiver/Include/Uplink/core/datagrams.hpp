//
//  network/datagrams.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./datagrams.h"
# include "./macros.h"
# include "./platform.h"
# include <cassert>

namespace uplink {

//------------------------------------------------------------------------------

inline String
NetworkAddress::toString () const
{
    return formatted_copy("%d.%d.%d.%d", ipv4[3], ipv4[2], ipv4[1], ipv4[0]);
}

inline void
NetworkAddress::operator = (uint32_t ipv4_addr)
{
    ipv4[0] = (ipv4_addr & 0xFF000000) >> 24;
    ipv4[1] = (ipv4_addr & 0x00FF0000) >> 16;
    ipv4[2] = (ipv4_addr & 0x0000FF00) >> 8 ;
    ipv4[3] = (ipv4_addr & 0x000000FF)      ;
}

inline std::ostream&
operator << (std::ostream& o, const NetworkAddress& addr)
{
    return o << addr.toString();
}

//------------------------------------------------------------------------------

inline
DatagramBroadcaster::DatagramBroadcaster ()
: broadcaster(0)
{

}

inline
DatagramBroadcaster::~DatagramBroadcaster ()
{
    close();
}

inline bool
DatagramBroadcaster::open (int port)
{
    if (ready())
        close();

    broadcaster = UDPBroadcaster_create(port);

    return 0 != broadcaster;
}

inline void
DatagramBroadcaster::close ()
{
    if (!ready())
        return;

    zero_delete(broadcaster);
}

inline bool
DatagramBroadcaster::ready () const
{
    return 0 != broadcaster;
}

inline bool
DatagramBroadcaster::send (const Byte* bytes, Size length)
{
    assert(ready());

    return broadcaster->broadcast(bytes, length);
}

inline bool
DatagramBroadcaster::send (CString message)
{
    assert(0 != message);

    const size_t length = strlen(message);

    return send(reinterpret_cast<const Byte*>(message), length);
}

//------------------------------------------------------------------------------

inline
DatagramListener::DatagramListener ()
: listener(0)
{

}

inline
DatagramListener::~DatagramListener ()
{
    close();
}

inline bool
DatagramListener::open (int port)
{
    if (ready())
        close();

    listener = UDPListener_create(port);

    return 0 != listener;
}

inline void
DatagramListener::close ()
{
    if (!ready())
        return;

    zero_delete(listener);
}

inline bool
DatagramListener::ready () const
{
    return 0 != listener;
}

inline bool
DatagramListener::receive (Byte* bytes, Size size, NetworkAddress& sender)
{
    assert(0 != listener);

    return listener->receive(bytes, size, sender);
}

//------------------------------------------------------------------------------

}
