//
//  network/message.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./core/enum.h"
# include "./core/bytes.h"
# include "./core/chunks.h"
# include "./core/serializers.h"

namespace uplink {

//------------------------------------------------------------------------------

// FIXME: Get rid of the _EXCEPT_COMMAND bit.
// NT: Raaaah! Clean this shit up, dude!
    
# define UPLINK_SYSTEM_MESSAGES() \
    UPLINK_MESSAGE(SessionSetup     , sessionSetup) \
    UPLINK_MESSAGE(SessionSetupReply, sessionSetupReply) \
    UPLINK_MESSAGE(KeepAlive        , keepAlive)

# define UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND() \
    UPLINK_MESSAGE(GyroscopeEvent    , gyroscopeEvent) \
    UPLINK_MESSAGE(AccelerometerEvent, accelerometerEvent) \
    UPLINK_MESSAGE(DeviceMotionEvent , deviceMotionEvent) \
    UPLINK_MESSAGE(CameraPose        , cameraPose) \
    UPLINK_MESSAGE(CameraFixedParams , cameraFixedParams) \
    UPLINK_MESSAGE(Blob              , blob) \
    UPLINK_MESSAGE(Image             , image) \
    UPLINK_MESSAGE(CameraFrame       , cameraFrame)

# define UPLINK_USER_MESSAGES_EXCEPT_CUSTOM_COMMAND() \
    UPLINK_MESSAGE(VersionInfo       , versionInfo) \
    UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND() \

# define UPLINK_MESSAGES() \
    UPLINK_SYSTEM_MESSAGES() \
    UPLINK_MESSAGE(CustomCommand, customCommand) \
    UPLINK_USER_MESSAGES_EXCEPT_CUSTOM_COMMAND()

//------------------------------------------------------------------------------

UPLINK_ENUM_BEGIN(MessageKind)
# define UPLINK_MESSAGE(Name, name) \
    MessageKind_##Name,
         UPLINK_MESSAGES()
# undef  UPLINK_MESSAGE
UPLINK_ENUM_END(MessageKind)

typedef uint32 MessageLength;

static const uint16 InvalidMessageKind = 0xFFFF;

typedef uint32 SessionId;

static const uint32 SystemSessionId  = 0x00000000;
static const uint32 AnySessionId     = 0x00000001;
static const uint32 FirstSessionId   = 0x00000010;
static const uint32 InvalidSessionId = 0xFFFFFFFF;

struct MessageHeader : Header
{
    MessageHeader (Size    magicSize);
    MessageHeader (CString magic);

    MessageKind    kind;
    MessageLength  length;
    SessionId      session;

private:
    void init ();
};

//------------------------------------------------------------------------------

struct Message : Serializable
{
    static const Size MaxSize = 0x4000000; // 64 MB

    virtual ~Message () {}

    virtual Message* clone () const = 0;
    
    virtual MessageKind kind () const = 0;
    
    CString name () const
    {
        switch (kind())
        {
# define UPLINK_MESSAGE(Name, name) \
            case MessageKind_##Name: \
                return #Name;
                UPLINK_MESSAGES()
# undef  UPLINK_MESSAGE
            default:
                assert(false);
                return "Unknown";
        }
    }
    
    virtual String toString () const
    {
        String ret("<");
        ret += name();
        ret += ">";
        return ret;
    }

    void swapWith (Message& other)
    {
        uplink_swap(sessionId, other.sessionId);
    }

#define UPLINK_MESSAGE_CLASS(Name) \
    virtual Name* clone () const   \
    { \
        return new Name(*this);    \
    } \
    \
    virtual MessageKind kind () const \
    { \
        return MessageKind_##Name; \
    }

    template < typename Class >       Class& as ()       { return *downcast<      Class>(this); }
    template < typename Class > const Class& as () const { return *downcast<const Class>(this); }

    SessionId sessionId;
};

//------------------------------------------------------------------------------

struct MessageSerializer
{
public:
    MessageSerializer ()
    : incoming("^lnk")
    , outgoing("^lnk")
    {

    }

    MessageSerializer (CString magic);

    ~MessageSerializer ();

public:
    void setMagic (CString newMagic)
    {
        magic = newMagic;
        incoming.header.magic = newMagic;
        outgoing.header.magic = newMagic;
    }

public:
    MessageKind registerMessage  (Message* message);

public:
    virtual Message* readMessage (InputStream & input );
    virtual bool    writeMessage (OutputStream& output, const Message& message);

private:
    typedef std::vector<Message*> Messages;

private:
    String magic;

    struct Channel
    {
        Channel (CString magic);

        MessageHeader header;
        Buffer        buffer;
        Messages      messages;
    };

    Channel incoming;
    Channel outgoing;
};

//------------------------------------------------------------------------------

struct MessageInput
{
    MessageInput (InputStream& input, MessageSerializer& messageSerializer);

    Message* readMessage ();

    InputStream& input;
    MessageSerializer& messageSerializer;
};

//------------------------------------------------------------------------------

struct MessageOutput
{
    MessageOutput (OutputStream& output, MessageSerializer& messageSerializer);

    bool writeMessage (const Message& message);

    OutputStream& output;
    MessageSerializer&  messageSerializer;
};

//------------------------------------------------------------------------------

struct MessageStream
{
    MessageStream (DuplexStream& stream, MessageSerializer& messageSerializer);

    Message* readMessage ();
    bool    writeMessage (const Message& message);

    DuplexStream& stream;
    MessageSerializer&  messageSerializer;
};

//------------------------------------------------------------------------------

}

# include "./message.hpp"
