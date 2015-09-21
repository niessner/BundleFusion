//
//  network/discovery.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./discovery.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
ServicePublisher::ServicePublisher (uint16 udpport, bool autoStart)
: Thread("uplink::ServicePublisher")
, udpport(udpport)
, broadcastingEnabled (true)
{
    if (autoStart)
        start();
}

inline
ServicePublisher::~ServicePublisher ()
{
    join();
}

inline void
ServicePublisher::setBroadcastingEnabled (bool enabled)
{
    broadcastingEnabled = enabled;
}

inline void
ServicePublisher::run ()
{
    DatagramBroadcaster broadcaster;

    report_unless("Cannot open datagram broadcaster.", broadcaster.open(udpport));

    int serviceCounts = (int)serviceList.size();

    std::vector<Buffer> bufferList(serviceCounts);

    // init serviceList

    for(int i = 0; i < serviceCounts; i++)
    {
        Buffer *buffer = &bufferList[i];
        buffer->reserve(UPLINK_SERVICE_DISCOVERY_DATAGRAM_LENGTH);
        BufferOutputStream output(*buffer);
        OutputStreamWriter writer(output);

        writer.writeMagic(UPLINK_SERVICE_DISCOVERY_DATAGRAM_MAGIC);

        writer.write(serviceList[i].port);

        const size_t serviceNameSize_ = strlen(serviceList[i].serviceName.c_str());
        assert(0 < serviceNameSize_ && serviceNameSize_ < 256);
        uint8 serviceNameSize = uint8(serviceNameSize_);

        writer.writeBytes(&serviceNameSize, sizeof(uint8));

        writer.writeBytes((unsigned char*)serviceList[i].serviceName.c_str(), serviceNameSize);

        writer.write(serviceList[i].versionRange.minimum.major);

        writer.write(serviceList[i].versionRange.minimum.minor);

        writer.write(serviceList[i].versionRange.maximum.major);

        writer.write(serviceList[i].versionRange.maximum.minor);

        //TODO: If this happens, we need to remove this service from the bufferList
        report_continue_if("Error! the service protocol string is too long",
                            buffer->size() > UPLINK_SERVICE_DISCOVERY_DATAGRAM_LENGTH);

    }

    while (isRunning())
    {
        if (broadcastingEnabled)
        {
            for(int i = 0; i < serviceCounts; i++)
            {
                ScopedProfiledTask _(ProfilerTask_ServicePublishing);
            
                broadcaster.send(bufferBytes(bufferList[i]), bufferList[i].size());
            }
        }

        sleep(.5);
    }

}

//------------------------------------------------------------------------------

inline
ServiceFinder::ServiceFinder (uint16 port, bool autostart)
: Thread("uplink::ServiceFinder")
, port(port)
{
    // Otherwise call start manually.
    if (autostart)
        start();
}

inline
ServiceFinder::~ServiceFinder ()
{
    join();
}


inline void
ServiceFinder::run ()
{
    DatagramListener listener;

    return_unless(listener.open(port)); // FIXME: Report.

    uplink_log_info("Listening to services.");

    Buffer buffer(UPLINK_SERVICE_DISCOVERY_DATAGRAM_LENGTH, 0);
    while (isRunning())
    {
        NetworkAddress sender;
        BufferInputStream input(buffer);
        InputStreamReader reader(input);

        // This receive call is blocking, but has a 2 second timeout (see datagrams.hpp)
        // This will allow the listener to process as many services as it needs to, while still
        // allowing this thread to gracefully exit if isRunning() changes to false.
        if (listener.receive(&buffer[0], buffer.size(), sender))
        {      
            continue_unless(reader.readMagic(UPLINK_SERVICE_DISCOVERY_DATAGRAM_MAGIC));

            uint16 servicePort;
            continue_unless(reader.read(servicePort));

            uint8 serviceNameSize;
            continue_unless(reader.read(serviceNameSize));

            continue_unless(0 < serviceNameSize);

            String serviceName(serviceNameSize, 0);
            continue_unless(reader.readBytes((unsigned char*)&serviceName[0], serviceNameSize));

            uint8 minimumVersionMajor;
            continue_unless(reader.read(minimumVersionMajor));

            uint8 minimumVersionMinor;
            continue_unless(reader.read(minimumVersionMinor));

            uint8 maximumVersionMajor;
            continue_unless(reader.read(maximumVersionMajor));

            uint8 maximumVersionMinor;
            continue_unless(reader.read(maximumVersionMinor));

            onServiceDiscovered(
                Service(
                    ServiceEntry(
                        serviceName,
                        servicePort,
                        VersionRange(
                            minimumVersionMajor,
                            minimumVersionMinor,
                            maximumVersionMajor,
                            maximumVersionMinor
                        )
                    ),
                    sender
                )
            );
        }

        // EAGAIN will be set when the receive call times out, but we're ok with that
        // check out the man page from "recvfrom"
        else if(errno != 0 && errno != EAGAIN)
        {
            // In case of power button down, we need to recreate the listener
            listener.open(port);
            errno = 0;
        }

    }
}

//------------------------------------------------------------------------------

}
