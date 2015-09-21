//
//  network/messages.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./message.h"
# include "./motion.h"
# include "./camera-frame.h"
# include "./camera-pose.h"
# include <cctype>

namespace uplink {

//------------------------------------------------------------------------------

struct KeepAlive : Message
{
    UPLINK_MESSAGE_CLASS(KeepAlive)

    virtual bool serializeWith (Serializer &serializer)
    {
        return true; // Nothing to read or write, there.
    }
};

//------------------------------------------------------------------------------

// FIXME: Move this where it belongs.

UPLINK_ENUM_BEGIN(SessionSetupStatus)
    SessionSetupStatus_Success,
    SessionSetupStatus_Failure,
// FIXME: SessionSetupStatus_Failure_ColorUnavailable,
UPLINK_ENUM_END(SessionSetupStatus)

struct SessionSetupReply : Message
{
    UPLINK_MESSAGE_CLASS(SessionSetupReply)

    enum { MaxLength = 0xffff }; // FIXME: Factor out short strings.

    SessionSetupReply ()
    : remoteSessionId(InvalidSessionId)
    , status(SessionSetupStatus_Failure)
    , message()
    {
    }

    SessionSetupReply (SessionId remoteSessionId, SessionSetupStatus status, String message = "")
    : remoteSessionId(remoteSessionId)
    , status(status)
    , message(message)
    {
    }

    void swapWith (SessionSetupReply& other)
    {
        Message::swapWith(other);
        
        uplink_swap(remoteSessionId, other.remoteSessionId);
        uplink_swap(status   , other.status);
        uplink_swap(message  , other.message);
    }

    virtual bool serializeWith (Serializer& s)
    {
        return_false_unless(s.put(remoteSessionId));

        return_false_unless(s.put(status));
    
        if (s.isReader())
            return s.asReader().readString(message, MaxLength);
        else
            return s.asWriter().writeString(message, MaxLength);
    }

    SessionId          remoteSessionId;
    SessionSetupStatus status;
    String             message;
};

//------------------------------------------------------------------------------

namespace { // FIXME: Move this where it belongs.

struct NonPrintableFilter
{
    NonPrintableFilter (char replacement = '?') : replacement(replacement) {}

    char operator () (char c) const { return 0 < c ? (std::isprint(c) ? c : replacement) : replacement; }

    char replacement;
};

}

//------------------------------------------------------------------------------

struct CustomCommand : Message
{
    UPLINK_MESSAGE_CLASS(CustomCommand)
   
    enum { MaxLength = 0xffff }; // FIXME: Factor out short strings.

    CustomCommand ()
    {}

    CustomCommand (const String& command_)
    : command(command_)
    {
        assert(command.size() < size_t(MaxLength));
    }

    CustomCommand (CString command_)
    : command(command_)
    {
        assert(command.size() < size_t(MaxLength));
    }

    void swapWith (CustomCommand& other)
    {
        Message::swapWith(other);

        uplink_swap(other.command, command);
    }

    virtual bool serializeWith (Serializer& s)
    {
        if (s.isReader())
            return s.asReader().readString(command, MaxLength);
        else
            return s.asWriter().writeString(command, MaxLength);
    }

    virtual String toString () const
    {
        // Commands might be very long and contain zeroes and binary data.

        static const size_t maxLength = 256;

        String ret(command, 0, maxLength);

        std::transform(ret.begin(), ret.end(), ret.begin(), NonPrintableFilter());

        return ret;
    }

    String command;
};

//------------------------------------------------------------------------------

}

# include "./messages.hpp"
