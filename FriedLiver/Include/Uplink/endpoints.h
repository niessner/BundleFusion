//
//  network/endpoints.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./messages.h"
# include "./discovery.h"
# include "./sessions-setup.h"
# include "./image-codecs.h"
# include "./core/macros.h"
# include "./core/streams.h"
# include "./core/queues.h"
# include "./core/wires.h"
# include <cctype>

# if ARCTURUS
@class VTEncoder_H264;
@class VTDecoder_H264;
@class VTDecoder_JPEG;
@class VTEncoder_JPEG;
#endif

namespace uplink {

//------------------------------------------------------------------------------

struct Endpoint
{
public:
    Endpoint ();
    virtual ~Endpoint ();

public:
    bool isConnected () const;

public:
    void disconnect ();

public:
    virtual void disconnected () = 0;
    virtual void registerMessages (MessageSerializer& messageSerializer);
    virtual bool sendMessages (MessageOutput& output, bool& sent);
    virtual bool receiveMessage (Message* message);
    virtual void reset ()
    {
        sessionSetupQueue.reset();\
        sessionSetupReplyQueue.reset();

        customCommandQueue.reset();

# define UPLINK_MESSAGE(Name, name) \
        name##Queue.reset()  ;
         UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND()
# undef  UPLINK_MESSAGE
    }

public:
    virtual void onVersionInfo (const VersionInfo& versionInfo) = 0;
    virtual void onSessionSetup (const SessionSetup& sessionSetup) = 0;
    virtual void onSessionSetupReply (const SessionSetupReply& sessionSetupReply) = 0;
    virtual bool onMessage (const Message& message) = 0;
    virtual void onCustomCommand (const String& command) // FIXME: Commands should be part of the messages list.
    {
        uplink_log_error("Unknown custom command: %s.", command.c_str());
    }

private:
    void messagePushed (const Message& message)
    {
        uplink_log_debug("%s pushed.", message.name());

        channelStats[message.kind()].pushing.add(message);
    }

    void messageSent (const Message& message)
    {
        uplink_log_debug("%s sent.", message.name());

        channelStats[message.kind()].sending.add(message);
    }
    
    void messageReceived (const Message& message)
    {
        uplink_log_debug("%s received.", message.name());

        channelStats[message.kind()].receiving.add(message);
    }
    
    void messageDelivered (const Message& message)
    {
        uplink_log_debug("%s delivered.", message.name());

        channelStats[message.kind()].delivering.add(message);
    }

    template < class Message >
    bool sendMessageByCopy (const Message& message, SessionId sessionId, Queue<Message>& queue)
    {
        if (!isConnected())
            return false;
        
        const_cast<Message&>(message).sessionId = sessionId;

        queue.pushByCopy(message);

        uplink_log_debug("%s queued for sending: %s (session: %d)", message.name(), message.toString().c_str(), message.sessionId);

        messagePushed(message);
        
        wire->notifySender();
       
        return true;
    }

    template < class Message >
    bool sendMessageBySwap (Message& message, SessionId sessionId, Queue<Message>& queue)
    {
        if (!isConnected())
            return false;
        
        message.sessionId = sessionId;

        uplink_log_debug("%s queued for sending: %s (session: %d)", message.name(), message.toString().c_str(), message.sessionId);
        messagePushed(message);

        queue.pushBySwap(message);

        // The message has been swapped and is now invalid.
        
        wire->notifySender();

        return true;
    }
    
    bool deliverMessage (const Message& message)
    {
        if (this->onMessage(message))
        {
            messageDelivered(message);
        
            return true;
        }
        else
        {
            uplink_log_error("%s could not be delivered.", message.name());
        
            return false;
        }
    }

private:
    bool sendMessage (MessageOutput& output, const Message& message);

public:
    bool sendVersionInfo        (const  VersionInfo      & versionInfo       ) { return sendMessageByCopy(versionInfo       , SystemSessionId,         versionInfoQueue); }
    bool sendSessionSetup       (const  SessionSetup     & sessionSetup      )
    {
        lastSessionSetup = sessionSetup;
    
        return sendMessageByCopy(sessionSetup, SystemSessionId, sessionSetupQueue);
    }
    
    bool sendSessionSetupReply  (const  SessionSetupReply& sessionSetupReply ) { return sendMessageByCopy(sessionSetupReply , SystemSessionId,   sessionSetupReplyQueue); }
//  bool sendKeepAlive          (const KeepAlive& keepalive = KeepAlive()    ) { return sendMessageByCopy(keepAlive         , SystemSessionId,           keepaliveQueue); }
    bool sendGyroscopeEvent     (const     GyroscopeEvent& gyroscopeEvent    ) { return sendMessageByCopy(    gyroscopeEvent, currentSessionId,     gyroscopeEventQueue); }
    bool sendAccelerometerEvent (const AccelerometerEvent& accelerometerEvent) { return sendMessageByCopy(accelerometerEvent, currentSessionId, accelerometerEventQueue); }
    bool sendDeviceMotionEvent  (const  DeviceMotionEvent& deviceMotionEvent ) { return sendMessageByCopy( deviceMotionEvent, currentSessionId,  deviceMotionEventQueue); }
    bool sendCameraPose         (const         CameraPose& cameraPose        ) { return sendMessageByCopy(        cameraPose, currentSessionId,         cameraPoseQueue); }
    bool sendBlob               (                    Blob& blob              ) { return sendMessageBySwap(blob              , currentSessionId,               blobQueue); }
    bool sendCustomCommand      (           CustomCommand& customCommand     ) { return sendMessageBySwap(customCommand     , currentSessionId,      customCommandQueue); }
    bool sendCameraFrame        (             CameraFrame& cameraFrame       ) { return sendMessageBySwap(cameraFrame       , currentSessionId,        cameraFrameQueue); }
    bool sendImage              (                   Image& image             ) { return sendMessageBySwap(image             ,     AnySessionId,              imageQueue); }

    bool sendCustomCommand (CString command)
    {
        CustomCommand customCommand(command);
        
        return sendCustomCommand(customCommand);
    }
   
# define UPLINK_MESSAGE(Name, name) \
public: \
    Queue<Name>   name##Queue  ;  \

         UPLINK_MESSAGES()
# undef  UPLINK_MESSAGE

    // FIXME: Make queues private.
    // FIXME: Make queues NOT message-specific.
    // FIXME: Queues should be private.

public:
    struct ChannelStat
    {
        struct Stage
        {
            void add (const Message& message)
            {
                traffic += message.serializedSize();

                rate.tick();
            }
            
            String toString () const
            {
                return formatted_copy("Traffic: %s Rate: %s", prettyByteSize(traffic).c_str(), rate.toString().c_str());
            }

            void reset ()
            {
                traffic = 0.;
                rate.reset();
            }
        
            double        traffic;
            RateEstimator rate;
        };

        void logInfo (CString channelName)
        {
            uplink_log_info("%s: Pushing   : %s", channelName, pushing   .toString().c_str());
            uplink_log_info("%s: Sending   : %s", channelName, sending   .toString().c_str());
            uplink_log_info("%s: Receiving : %s", channelName, receiving .toString().c_str());
            uplink_log_info("%s: Delivering: %s", channelName, delivering.toString().c_str());
        }

        void reset ()
        {
            uplink_log_debug("Endpoint reset.");
        
            pushing.reset();
            sending.reset();
            receiving.reset();
            delivering.reset();
        }
        
        Stage pushing;
        Stage sending;
        Stage receiving;
        Stage delivering;
    };

public: // FIXME: This should be private.
    std::vector<ChannelStat> channelStats;

protected:
    template < typename Type>
    void setChannelSettings (const ChannelSettings& settings, Queue<Type>& queue);
    
# define UPLINK_MESSAGE(Name, name) \
    void set##Name##ChannelSettings (const ChannelSettings& settings) \
    { \
        setChannelSettings(settings, name##Queue); \
    }
         UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND()
# undef  UPLINK_MESSAGE
    
    void setAllChannelSettings (const SessionSettings& sessionSettings)
    {
# define UPLINK_MESSAGE(Name, name) \
        set##Name##ChannelSettings(sessionSettings.name##Channel); \
         UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND()
# undef  UPLINK_MESSAGE
    }

protected:
    SessionId currentSessionId;

private:
    bool isActiveSession (SessionId sessionId) const
    {
        return_true_if(SystemSessionId == sessionId);
        
        return_true_if(AnySessionId == sessionId);
    
        return_false_unless(sessionId == currentSessionId);

        return true;
    }

protected:
    friend class Wire;
    Wire* wire;
    
private:
    template < class Message > bool sendSimpleMessage (Queue<Message>& messageQueue, MessageOutput& output, bool& sent);
    bool receiveSimpleMessage (Message* message);

public:
    ImageCodecs imageCodecs;

private:
    bool canCompressFeedbackImage (const Image& image) const
    {
        return currentSessionSettings.feedbackImageCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.feedbackImageCodec].canCompress(image.format)
            ;
    }
    
    bool canDecompressFeedbackImage (const Image& image) const
    {
        return currentSessionSettings.feedbackImageCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.feedbackImageCodec].canDecompress(image.format)
            ;
    }

    bool compressFeedbackImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_CompressImage);
    
        assert(canCompressFeedbackImage(source));

        return imageCodecs.byId[currentSessionSettings.feedbackImageCodec].compress(source, target);
    }

    bool decompressFeedbackImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_DecompressImage);

        assert(canDecompressFeedbackImage(source));
    
        return imageCodecs.byId[currentSessionSettings.feedbackImageCodec].decompress(source, target);
    }

    bool canCompressColorCameraImage (const Image& image) const
    {    
        return currentSessionSettings.colorCameraCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.colorCameraCodec].canCompress(image.format)
            ;
    }

    bool canDecompressColorCameraImage (const Image& image) const
    {
        return currentSessionSettings.colorCameraCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.colorCameraCodec].canDecompress(image.format)
            ;
    }

    bool compressColorCameraImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_CompressImage);
    
        assert(canCompressColorCameraImage(source));

        return imageCodecs.byId[currentSessionSettings.colorCameraCodec].compress(source, target);
    }

    bool decompressColorCameraImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_DecompressImage);

        assert(canDecompressColorCameraImage(source));
    
        return imageCodecs.byId[currentSessionSettings.colorCameraCodec].decompress(source, target);
    }

    bool canCompressDepthCameraImage (const Image& image) const
    {
        return currentSessionSettings.depthCameraCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.depthCameraCodec].canCompress(image.format)
            ;
    }
    
    bool canDecompressDepthCameraImage (const Image& image) const
    {
        return currentSessionSettings.depthCameraCodec != ImageCodecId_Invalid
            && imageCodecs.byId[currentSessionSettings.depthCameraCodec].canDecompress(image.format)
            ;
    }
    
    bool compressDepthCameraImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_CompressImage);

        assert(canCompressDepthCameraImage(source));
    
        return imageCodecs.byId[currentSessionSettings.depthCameraCodec].compress(source, target);
    }

    bool decompressDepthCameraImage (const Image& source, Image& target) const
    {
        ScopedProfiledTask _(ProfilerTask_DecompressImage);
    
        assert(canDecompressDepthCameraImage(source));

        return imageCodecs.byId[currentSessionSettings.depthCameraCodec].decompress(source, target);
    }

# if ARCTURUS
private:
    void requireH264Encoding ();
    void requireH264Decoding ();
    void cleanupH264 ();

private:
    VTDecoder_H264* vtDecoder;
    VTEncoder_H264* vtEncoder;
# endif

public:
    SessionSetup    lastSessionSetup;
    SessionSettings currentSessionSettings;
};

//------------------------------------------------------------------------------

}

# include "./endpoints.hpp"
# include "./core/wires.hpp" // FIXME: Temporary workaround.
