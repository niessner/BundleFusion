//
//  network/wires.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./wires.h"
# include "../endpoints.h"
# include "./threads.h"
# include "./macros.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
Wire::Wire (DuplexStream* stream, Endpoint* endpoint)
    : stream(stream)
    , endpoint(endpoint)
    , sender(this)
    , receiver(this)
{
    assert(0 != stream);
    assert(0 != endpoint);

    endpoint->wire = this;
}

inline
Wire::~Wire ()
{
    stop();
}

inline void
Wire::start ()
{
    endpoint->registerMessages(serializer);

    receiver.start();
    sender.start();
}

inline void
Wire::stop ()
{
    TCPConnection* connection = dynamic_cast<TCPConnection*>(stream);
    if (0 != connection)
        connection->disconnecting = true;

    receiver.join();
    sender.join();
    
    delete connection;
    stream = 0;
}

inline void
Wire::notifySender ()
{
    sender.notify();
}

inline void
Wire::notifyReceiver ()
{
    sender.notify();
}

//------------------------------------------------------------------------------

inline
Wire::Loop::Loop (Wire* that, CString name)
    : Thread(name)
    , that(that)
    , disconnected(false)
{
    assert(0 != that);
}

inline void
Wire::Loop::run ()
{
    loop();

    if (that->isConnected()) // Then no other loop has disconnected the endpoint, yet.
    {
        {
            const MutexLocker _(mutex);

            disconnected = true; // Which will prevent other running loops from getting there on disconnection.
        }

        that->endpoint->disconnected();
        that->endpoint->reset();
    }
}

//------------------------------------------------------------------------------

inline void
Wire::Receiver::loop ()
{
    MessageInput messageInput(*that->stream, that->serializer);

    while (isRunning())
    {
        Message* message = messageInput.readMessage();

        report_unless("cannot read network message", 0 != message);

        if (MessageKind_KeepAlive == message->kind())
            continue; // In effect ignoring the keep-alive message.

        uplink_log_debug("Message received: %s (session: %d)", message->name(), message->sessionId);

        report_unless("cannot receive network message", that->endpoint->receiveMessage(message));
    }
}

//------------------------------------------------------------------------------

inline void
Wire::Sender::loop ()
{
    MessageOutput messageOutput(*that->stream, that->serializer);

    while (isRunning())
    {
        bool sent = false;

        report_unless("cannot send network messages", that->endpoint->sendMessages(messageOutput, sent));

        if (sent)
        {
            keepAliveStopWatch.start();
        }
        else
        {
            static const double maxKeepAliveAge = 1.;
        
            const double keepAliveAge = keepAliveStopWatch.elapsed();

            if (maxKeepAliveAge < keepAliveAge)
            {
                KeepAlive keepAlive;
                keepAlive.sessionId = SystemSessionId;
                
                report_unless("cannot send keep-alive message", messageOutput.writeMessage(keepAlive));

                keepAliveStopWatch.start();
            }
            else
            {
                sleep(.001f); // Avoid active loops.
            }
        }

        // FIXME: Rework sender thread notifications.
    }
}

//------------------------------------------------------------------------------

}
