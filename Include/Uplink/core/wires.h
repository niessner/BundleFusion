//
//  network/wires.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "../message.h"
# include "./threads.h"

namespace uplink {

struct Endpoint;

//------------------------------------------------------------------------------

class Wire
{
public:
     Wire (DuplexStream* stream, Endpoint* endpoint);
    ~Wire ();

public:
    void start ();
    void stop ();

    void notifySender ();
    void notifyReceiver ();

public:
    bool isConnected () const
    {
        return
            !sender  .isDisconnected()
         && !receiver.isDisconnected();
    }

private:
    struct Loop : Thread
    {
        Loop (Wire* that, CString name);

        virtual void run  ();

        virtual void loop () = 0;

        bool isDisconnected () const { const MutexLocker _(mutex); return disconnected; }

        Wire* that;

    private:
        mutable Mutex mutex;
        bool          disconnected;
    };

    struct Receiver : Loop
    {
        Receiver (Wire* that) : Loop(that, "uplink::Wire::Receiver") {}

        virtual void loop ();
    };

private:
    struct Sender : Loop
    {
        Sender (Wire* that) : Loop(that, "uplink::Wire::Sender") {}

        virtual void loop ();

    private:
        StopWatch keepAliveStopWatch;
    };

private:
    DuplexStream* stream;
    Endpoint* endpoint;
    MessageSerializer serializer;
    Sender sender;
    Receiver receiver;
};

//------------------------------------------------------------------------------

}

// We can't include the inline definitions from here.
// See: endpoints.h.
// # include "./wires.hpp"
