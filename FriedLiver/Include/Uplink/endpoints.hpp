//
//  network/endpoints.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./endpoints.h"

# if ARCTURUS
#   include <Graphics/VTEncoder_JPEG.h>
#   include <Graphics/VTDecoder_JPEG.h>
#   include <Graphics/VTEncoder_H264.h>
#   include <Graphics/VTDecoder_H264.h>
#   include <Graphics/ImageCodecs_Apple.h>
#   include <Core/DebugDefines.h>
# endif

namespace uplink {

//------------------------------------------------------------------------------

inline
Endpoint::Endpoint ()
    : currentSessionId(InvalidSessionId)
    , wire(0)
# if ARCTURUS
    , vtDecoder(nil)
    , vtEncoder(nil)
# endif
{
    // Color implementations will be supplied elsewhere.

    int numChannels = 0;

    ++numChannels; // SessionSetup
    ++numChannels; // SessionSetupReply
    ++numChannels; // KeepAlive

    ++numChannels; // VersionInfo
    ++numChannels; // CustomCommand

# define UPLINK_MESSAGE(Name, name) \
    name##Queue.setMaximumSize(1);  \
    ++numChannels;
         UPLINK_USER_MESSAGES_EXCEPT_VERSION_INFO_AND_CUSTOM_COMMAND()
# undef  UPLINK_MESSAGE

    assert(MessageKind_HowMany == numChannels);
    
    channelStats.resize(numChannels);
        
    // The command queue is kind of a special case. We almost never want to drop them.
    customCommandQueue.setMaximumSize(0); // This should be the default, but making sure.
    customCommandQueue.setDroppingStategy(Queue<CustomCommand>::OldestOneDroppingStrategy);

    // FIXME: All this updater stuff is crappy. Properly refactor channels, already!

    blobQueue.setMaximumSize(0);

//    oldRgbdFrameQueue.setMaximumSize(0); // FIXME: Temporary hack for h264 purposes.

    cameraPoseQueue.setMaximumSize(30); // FIXME: This may not be ideal for all situations.

    deviceMotionEventQueue.setMaximumSize(0);

    // FIXME: Move this to image-codecs.
#if ARCTURUS
    imageCodecs.jpeg.compressInputFormat = ImageFormat_YCbCr;
    imageCodecs.jpeg.compress = [this] (const Image& source, Image& target) -> bool {

        using namespace graphics;

        assert(target.isEmpty());
        assert(!source.isEmpty());
        assert(ImageFormat_YCbCr == source.format);
        assert(source.storage.type == Image::Storage::Type_PixelBuffer);

        OCMemoryBlock* memoryBlock = new OCMemoryBlock;

        bool ret = encode_image (
            ImageCodec_Hardware_JPEG,
            source.pixelBuffer(),
            defaultQuality,
            *memoryBlock
        );

        if (!ret)
            return false;
        
        target.cameraInfo = source.cameraInfo;
        target.width = source.width;
        target.height = source.height;
        target.format = ImageFormat_JPEG;
        target.planes[0].buffer      = memoryBlock->Data;
        target.planes[0].sizeInBytes = memoryBlock->Size;
        target.release = [memoryBlock] () { delete memoryBlock; };
        target.retain  = 0;
        
        return true;
    };

    imageCodecs.jpeg.decompressOutputFormat = ImageFormat_YCbCr;
    imageCodecs.jpeg.decompress = [this] (const Image& source, Image& target) -> bool {

        using namespace graphics;

        assert(target.isEmpty());
        assert(!source.isEmpty());
        assert(ImageFormat_JPEG == source.format);
//        assert(source.storage.type == Image::Storage::Type_BlockBuffer);

        CVImageBufferRef imageBuffer = decode_image (
            ImageCodec_Hardware_JPEG,
            (const uint8_t*)source.planes[0].buffer,
            source.planes[0].sizeInBytes
        );

        if (!imageBuffer)
            return false;
    
        target = Image(imageBuffer, source.cameraInfo);
        
        CFRelease(imageBuffer); // The target camera frame owns it now.

        return true;
    };
#endif

    // FIXME: Move this to image-codecs.
#if ARCTURUS
    imageCodecs.h264.compressInputFormat = ImageFormat_YCbCr;
    imageCodecs.h264.compress = [this] (const Image& source, Image& target) -> bool {

        assert(target.isEmpty());
        assert(!source.isEmpty());
        assert(ImageFormat_YCbCr == source.format);
        assert(source.storage.type == Image::Storage::Type_PixelBuffer);

        requireH264Encoding();
        
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

        __block CMSampleBufferRef sampleBuffer;

        if (![vtEncoder
            encodeFrameWithImageBuffer:source.pixelBuffer()
            presentationTime:CMTimeMake(1000. * source.cameraInfo.timestamp, 1000.)
            frameDuration:CMTimeMake(1000. * source.cameraInfo.duration, 1000.)
            andBlock: ^(CMSampleBufferRef sampleBuffer_) {
                assert(0 != sampleBuffer_);
                CFRetain(sampleBuffer_);
                sampleBuffer = sampleBuffer_;
                dispatch_semaphore_signal(semaphore);
            }
        ])
        {
#   if !__has_feature(objc_arc)
            dispatch_release(semaphore);
#   endif
            return false;
        }

        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

#   if !__has_feature(objc_arc)
        dispatch_release(semaphore);
#   endif

        target = Image(sampleBuffer, source.cameraInfo, source.width, source.height);
        
        {
            target.attachments.blobs.resize([VTEncoder_H264 numH264ParameterSetsForSampleBuffer:sampleBuffer]);

            [VTEncoder_H264 h264ParameterSetsForSampleBuffer:sampleBuffer withBlock:^(size_t index, const uint8_t* data, size_t size) {

                target.attachments.blobs[index].data.assign(data, data + size);

                // Ignore blob tag.
            }];
        }

        return true;
    };

    imageCodecs.h264.decompressOutputFormat = ImageFormat_YCbCr;
    imageCodecs.h264.decompress = [this] (const Image& source, Image& target) -> bool {

        requireH264Decoding();

        assert(target.isEmpty());
        assert(!source.isEmpty());
        assert(ImageFormat_H264 == source.format);

        uint8* sourceBuffer      = (uint8*)source.planes[0].buffer;
        size_t sourceSizeInBytes = source.planes[0].sizeInBytes;

        // Decode parameter sets.
        [vtDecoder setNumH264ParameterSets:source.attachments.blobs.size()];

        for (size_t n = 0; n < source.attachments.blobs.size(); ++n)
        {
            [vtDecoder setH264ParameterSetAtIndex:n
                withData: source.attachments.blobs[n].data.data()
                andSize:  source.attachments.blobs[n].data.size()
            ];
        }

        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

        __block CVImageBufferRef imageBuffer;

        if (![vtDecoder
            decodeFrameFromBytes:sourceBuffer
            size: sourceSizeInBytes
            timestamp:source.cameraInfo.timestamp
            duration:source.cameraInfo.duration
            andBlock: ^(CVImageBufferRef imageBuffer_) {
                assert(0 != imageBuffer_);
                CFRetain(imageBuffer_);
                imageBuffer = imageBuffer_;
                dispatch_semaphore_signal(semaphore);
            }
        ])
        {
#   if !__has_feature(objc_arc)
            dispatch_release(semaphore);
#   endif
            return false;
        }

        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

#   if !__has_feature(objc_arc)
        dispatch_release(semaphore);
#   endif
        
        target = Image(imageBuffer, source.cameraInfo);

        CFRelease(imageBuffer); // The target camera frame owns it now.

        return true;
    };

# endif
}

#if ARCTURUS
inline void
Endpoint::requireH264Encoding()
{
    if (nil != vtEncoder)
        return;

    vtEncoder = [[VTEncoder_H264 alloc] init];

    oc_assert(vtEncoder, "VTEncoder failed to initialize.");
}

inline void
Endpoint::requireH264Decoding()
{
    if (nil != vtDecoder)
        return;

    vtDecoder = [[VTDecoder_H264 alloc] init];
    
    oc_assert(vtDecoder, "VTDecoder failed to initialize.");
}

inline void
Endpoint::cleanupH264()
{
#   if !__has_feature(objc_arc)
    if (vtEncoder)
        [vtEncoder dealloc];
#   endif
    
    vtEncoder = nil;
    
#   if !__has_feature(objc_arc)
    if (vtDecoder)
        [vtDecoder dealloc];
#   endif

    vtDecoder = nil;
}
# endif

inline
Endpoint::~Endpoint()
{
    uplink_log_debug("~Endpoint");
    
    zero_delete(wire);

# if ARCTURUS
    cleanupH264();
# endif
}

inline bool
Endpoint::isConnected () const
{
    return 0 != wire && wire->isConnected();
}

inline void
Endpoint::disconnect ()
{
    if (0 != wire)
        wire->stop();

    wire = 0;
}

inline void
Endpoint::registerMessages (MessageSerializer& messageSerializer)
{
    messageSerializer.setMagic("skan");

# define UPLINK_MESSAGE(Name, name) \
    messageSerializer.registerMessage(new Name());
         UPLINK_MESSAGES()
# undef  UPLINK_MESSAGE
}

inline bool
Endpoint::sendMessages (MessageOutput& output, bool& sent)
{
    // FIXME: Rework message scheduling.

    VersionInfo versionInfo;
    if (versionInfoQueue.popBySwap(versionInfo))
    {
        assert(SystemSessionId == versionInfo.sessionId);
        return_false_unless(sendMessage(output, versionInfo));
        sent = true;
    }

    CustomCommand customCommand;
    if (customCommandQueue.popBySwap(customCommand))
    {
        customCommand.sessionId = AnySessionId;

        return_false_unless(sendMessage(output, customCommand));
       
        uplink_log_debug("Custom command sent: %s", customCommand.toString().c_str());

        if (customCommand.command == "disconnected")
        {
            disconnected();

            return false; // FIXME: This isn't actually an error.
        }
        
        sent = true;
    }

    SessionSetup sessionSetup;
    if (sessionSetupQueue.popBySwap(sessionSetup))
    {
        assert(SystemSessionId == sessionSetup.sessionId);

        return_false_unless(sendMessage(output, sessionSetup));

        uplink_log_debug("SessionSetup sent.");
    }

    SessionSetupReply sessionSetupReply;
    if (sessionSetupReplyQueue.popBySwap(sessionSetupReply))
    {
        assert(SystemSessionId == sessionSetupReply.sessionId);

        return_false_unless(sendMessage(output, sessionSetupReply));

        uplink_log_debug("SessionSetupReply sent.");
    }

    return_false_unless(sendSimpleMessage<GyroscopeEvent    >(gyroscopeEventQueue    , output, sent));
    return_false_unless(sendSimpleMessage<AccelerometerEvent>(accelerometerEventQueue, output, sent));
    return_false_unless(sendSimpleMessage<DeviceMotionEvent >(deviceMotionEventQueue , output, sent));
    return_false_unless(sendSimpleMessage<CameraFixedParams >(cameraFixedParamsQueue , output, sent));
    return_false_unless(sendSimpleMessage<CameraPose        >(cameraPoseQueue        , output, sent));
    return_false_unless(sendSimpleMessage<Blob              >(blobQueue              , output, sent));

    {
        Image image;
        if (imageQueue.popBySwap(image))
        {
            if ((image.isEmpty() || canCompressFeedbackImage(image)))
            {
                if (isActiveSession(image.sessionId))
                {
                    Image compressedImage;

                    if (image.isEmpty())
                    {
                        image.swapWith(compressedImage);
                    }
                    else
                    {
                        if (!compressFeedbackImage(image, compressedImage))
                            return false;
                    }

                    compressedImage.sessionId = image.sessionId;
                
                    return_false_unless(sendMessage(output, compressedImage));

                    uplink_log_debug("%s sent.", compressedImage.name());

                    sent = true;
                }
                else
                {
                    uplink_log_warning("%s not sent (stale session: %d).", image.name(), image.sessionId);
                }
            }
            else
            {
                uplink_log_debug("Current session settings:\n%s", currentSessionSettings.toString().c_str());

                uplink_log_warning("%s not sent (cannot compress).", image.name());
            }
        }
    }

    {
        CameraFrame cameraFrame;
        if (cameraFrameQueue.popBySwap(cameraFrame))
        {
            if ((cameraFrame.colorImage.isEmpty() || canCompressColorCameraImage(cameraFrame.colorImage))
             && (cameraFrame.depthImage.isEmpty() || canCompressDepthCameraImage(cameraFrame.depthImage)))
            {
                if (isActiveSession(cameraFrame.sessionId))
                {
                    CameraFrame compressedCameraFrame;

                    if (cameraFrame.colorImage.isEmpty())
                    {
                        cameraFrame.colorImage.swapWith(compressedCameraFrame.colorImage);
                    }
                    else
                    {
                        if (!compressColorCameraImage(cameraFrame.colorImage, compressedCameraFrame.colorImage))
                            return false;

                        compressedCameraFrame.colorImage.sessionId = cameraFrame.colorImage.sessionId;
                    }
                    
                    if (cameraFrame.depthImage.isEmpty())
                    {
                        cameraFrame.depthImage.swapWith(compressedCameraFrame.depthImage);
                    }
                    else
                    {
                        if (!compressDepthCameraImage(cameraFrame.depthImage, compressedCameraFrame.depthImage))
                            return false;
                        
                        compressedCameraFrame.depthImage.sessionId = cameraFrame.depthImage.sessionId;
                    }

                    return_false_unless(sendMessage(output, compressedCameraFrame));

                    compressedCameraFrame.sessionId = cameraFrame.sessionId;

                    sent = true;
                }
                else
                {
                    uplink_log_warning("%s not sent (stale session: %d).", cameraFrame.name(), cameraFrame.sessionId);
                }
            }
            else
            {
                uplink_log_debug("Current session settings:\n%s", currentSessionSettings.toString().c_str());
            
                uplink_log_warning("%s not sent (cannot compress).", cameraFrame.name());
            }

        }
    }

    return true;
}

inline bool
Endpoint::sendMessage (MessageOutput& output, const Message& message)
{
    return_false_unless(output.writeMessage(message));
    
    messageSent(message);

    return true;
}

inline bool
Endpoint::receiveMessage (Message* message)
{
    assert(0 != message);
    
    assert(MessageKind_Invalid < message->kind() && message->kind() < MessageKind_HowMany);

    messageReceived(*message);

    switch (message->kind())
    {
        case MessageKind_CustomCommand:
        {
            if (!isActiveSession(message->sessionId))
            {
                uplink_log_warning("Custom command discarded (stale session).");
                return true; // Skipping this stale command.
            }
            
            const CustomCommand& customCommand = *downcast<CustomCommand>(message);

            uplink_log_debug("Custom command received: %s", customCommand.toString().c_str());

            if (customCommand.command == "disconnect")
            {
                sendCustomCommand("disconnected");

                return false; // FIXME: See above.
            }
            else if (customCommand.command == "disconnected")
            {
                return false; // Likewise.
            }

            onCustomCommand(customCommand.command);

            return true;
        }

        case MessageKind_SessionSetup:
        {
            assert(SystemSessionId == message->sessionId);
            uplink_log_debug("SessionSetup received.");
            onSessionSetup(*downcast<SessionSetup>(message));
            return true;
        }

        case MessageKind_SessionSetupReply:
        {
            assert(SystemSessionId == message->sessionId);
            uplink_log_debug("SessionSetupReply received.");
            onSessionSetupReply(*downcast<SessionSetupReply>(message));
            return true;
        }

        case MessageKind_VersionInfo:
        {
            assert(SystemSessionId == message->sessionId);
            uplink_log_debug("VersionInfo received.");
            onVersionInfo(*downcast<VersionInfo>(message));
            return true;
        }

        case MessageKind_GyroscopeEvent:
        case MessageKind_AccelerometerEvent:
        case MessageKind_DeviceMotionEvent:
        case MessageKind_CameraPose:
        {
            return receiveSimpleMessage(message);
        }

        case MessageKind_Image:
        {
            assert(0 != message);

            const Image& image = message->as<Image>();
            
            if (!canDecompressFeedbackImage(image))
            {
                uplink_log_warning("%s discarded (cannot decompress: %d).", message->name(), message->sessionId);
                return true;
            }
            
            if (!isActiveSession(message->sessionId))
            {
                uplink_log_warning("%s discarded (stale session: %d).", message->name(), message->sessionId);
                return true;
            }

            Image decompressedImage;

            if (!decompressFeedbackImage(image, decompressedImage))
                return false;

            decompressedImage.sessionId = image.sessionId;

            uplink_log_debug("%s received.", message->name());

            return deliverMessage(decompressedImage);
        }
        
        case MessageKind_CameraFrame:
        {
            assert(0 != message);

            CameraFrame& cameraFrame = message->as<CameraFrame>();

            if ((!cameraFrame.colorImage.isEmpty() && !canDecompressColorCameraImage(cameraFrame.colorImage))
             || (!cameraFrame.depthImage.isEmpty() && !canDecompressDepthCameraImage(cameraFrame.depthImage)))
            {
                uplink_log_warning("%s discarded (cannot decompress: %d).", message->name(), message->sessionId);
                return true;
            }

            if (!isActiveSession(message->sessionId))
            {
                uplink_log_warning("%s discarded (stale session: %d).", message->name(), message->sessionId);
                return true;
            }

            CameraFrame decompressedCameraFrame;

            if (cameraFrame.depthImage.isEmpty())
            {
                cameraFrame.depthImage.swapWith(decompressedCameraFrame.depthImage);
            }
            else
            {
                if (!decompressDepthCameraImage(cameraFrame.depthImage, decompressedCameraFrame.depthImage))
                    return false;
                
                decompressedCameraFrame.depthImage.sessionId = cameraFrame.depthImage.sessionId;
            }

            if (cameraFrame.colorImage.isEmpty())
            {
                cameraFrame.colorImage.swapWith(decompressedCameraFrame.colorImage);
            }
            else
            {
                if (!decompressColorCameraImage(cameraFrame.colorImage, decompressedCameraFrame.colorImage))
                    return false;

                decompressedCameraFrame.colorImage.sessionId = cameraFrame.colorImage.sessionId;
            }

            decompressedCameraFrame.sessionId = cameraFrame.sessionId;

            uplink_log_debug("%s received.", message->name());
            
            return deliverMessage(decompressedCameraFrame);
        }

        case MessageKind_CameraFixedParams:
        case MessageKind_Blob:
        {
            return receiveSimpleMessage(message);
        }

        default:
            return false;
    }
}

template < typename Message >
inline void
Endpoint::setChannelSettings (const ChannelSettings& channelSettings, Queue<Message>& queue)
{
    switch (channelSettings.bufferingStrategy)
    {
        case uplink::BufferingStrategy_One:
            queue.setMaximumSize(1);
            break;
        case uplink::BufferingStrategy_Some:
            queue.setMaximumSize(channelSettings.droppingThreshold);
            break;
        default:
            break;
    }

    switch (channelSettings.droppingStrategy)
    {
        case uplink::DroppingStrategy_RandomOne:
            queue.setDroppingStategy(Queue<Message>::RandomOneDroppingStrategy);
            break;

        case uplink::DroppingStrategy_OldestOne:
            queue.setDroppingStategy(Queue<Message>::OldestOneDroppingStrategy);
            break;
            
        default:
            break;
    }
}

template < class Message >
inline bool
Endpoint::sendSimpleMessage (Queue<Message>& messageQueue, MessageOutput& output, bool& sent)
{
    Message message;
    if (messageQueue.popBySwap(message))
    {
        if (isActiveSession(message.sessionId))
        {
            return_false_unless(sendMessage(output, message));

            sent = true;
        }
        else
        {
            uplink_log_warning("%s not sent (stale session: %d).", message.name(), message.sessionId);
        }
    }

    return true;
}

inline bool
Endpoint::receiveSimpleMessage (Message* message)
{
    assert(0 != message);

    if (!isActiveSession(message->sessionId))
    {
        uplink_log_warning("%s discarded (stale session: %d).", message->name(), message->sessionId);
        return true;
    }

    uplink_log_debug("%s received.", message->name());
    
    return deliverMessage(*message);
}

//------------------------------------------------------------------------------

}
