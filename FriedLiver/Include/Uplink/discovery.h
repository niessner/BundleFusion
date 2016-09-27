//
//  network/discovery.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./services.h"
# include "./core/datagrams.h"
# include "./core/threads.h"
# include <cstring>
# include <vector>

namespace uplink {

//------------------------------------------------------------------------------

# define UPLINK_SERVICE_DISCOVERY_DATAGRAM_MAGIC  "^l1nk"
# define UPLINK_SERVICE_DISCOVERY_DATAGRAM_MAGIC_LENGTH sizeof_array(UPLINK_SERVICE_DISCOVERY_DATAGRAM_MAGIC)
# define UPLINK_SERVICE_DISCOVERY_DATAGRAM_LENGTH 512

//------------------------------------------------------------------------------

struct ServicePublisher : public Thread
{
    ServicePublisher (uint16 udpport, bool autoStart = false);

    ~ServicePublisher ();

    void setBroadcastingEnabled (bool enabled);

    virtual void run ();

    uint16                      udpport;
    std::vector<ServiceEntry>   serviceList;
    bool                        broadcastingEnabled;
};

//------------------------------------------------------------------------------

struct ServiceFinder : public Thread
{
    ServiceFinder (uint16 port, bool autostart = true);

    virtual ~ServiceFinder ();

    virtual void run ();

    virtual void onServiceDiscovered (const Service& service) = 0;

    uint16            port;
};

//------------------------------------------------------------------------------

}

# include "./discovery.hpp"
