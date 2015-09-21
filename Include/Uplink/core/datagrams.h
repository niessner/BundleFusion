//
//  network/datagrams.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./platform.h"

//------------------------------------------------------------------------------

namespace uplink {

class DatagramBroadcaster
{
public:
    DatagramBroadcaster ();
   ~DatagramBroadcaster ();

public:
    bool open (int port);
    void close ();
    bool ready () const;

public:
    bool send (const Byte* bytes, Size length);
    bool send (CString);

private:
    UDPBroadcaster* broadcaster;
};

//------------------------------------------------------------------------------

class DatagramListener
{
public:
    DatagramListener ();
   ~DatagramListener ();

public:
    bool open (int port);
    void close ();
    bool ready () const;

public:
    bool receive (Byte* bytes, Size size, NetworkAddress& sender);

private:
    UDPListener* listener;
};

}

# include "./datagrams.hpp"
