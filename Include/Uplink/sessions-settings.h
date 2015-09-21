//
//  network/sessions-settings.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/serializers.h"
# include "./core/enum.h"

namespace uplink {

//------------------------------------------------------------------------------

// FIXME: Add gray images support.

UPLINK_ENUM_BEGIN(DepthMode)
    DepthMode_None,
    DepthMode_QVGA,
    DepthMode_VGA,
    DepthMode_QVGA_60FPS,
UPLINK_ENUM_END(DepthMode)

UPLINK_ENUM_BEGIN(ColorMode)
    ColorMode_None,
    ColorMode_640x480  , ColorMode_VGA   = ColorMode_640x480,
    ColorMode_1296x968 ,
UPLINK_ENUM_END(ColorMode)

UPLINK_ENUM_BEGIN(InfraredMode)
    InfraredMode_None,
    InfraredMode_QVGA, // FIXME: Implement.
    InfraredMode_VGA,  // FIXME: Implement.
UPLINK_ENUM_END(InfraredMode)

UPLINK_ENUM_BEGIN(RegistrationMode)
    RegistrationMode_None,
    RegistrationMode_RegisteredDepth,
UPLINK_ENUM_END(RegistrationMode)

UPLINK_ENUM_BEGIN(FrameSyncMode)
    FrameSyncMode_None,
    FrameSyncMode_Depth,
    FrameSyncMode_Infrared, // FIXME: Implement.
UPLINK_ENUM_END(FrameSyncMode)

UPLINK_ENUM_BEGIN(ColorCameraExposureMode)
   ColorCameraExposureMode_Locked,
   ColorCameraExposureMode_Auto,
   ColorCameraExposureMode_ContinuousAuto,
UPLINK_ENUM_END(ColorCameraExposureMode)

UPLINK_ENUM_BEGIN(ColorCameraWhiteBalanceMode)
   ColorCameraWhiteBalanceMode_Locked,
   ColorCameraWhiteBalanceMode_Auto,
   ColorCameraWhiteBalanceMode_ContinuousAuto,
UPLINK_ENUM_END(ColorCameraWhiteBalanceMode)

UPLINK_ENUM_BEGIN(ImageCodecId)
    ImageCodecId_CompressedShifts,
    ImageCodecId_JPEG,
    ImageCodecId_H264,
UPLINK_ENUM_END(ImageCodecId)

UPLINK_ENUM_BEGIN(BufferingStrategy)
    BufferingStrategy_One,
    BufferingStrategy_Some,
UPLINK_ENUM_END(BufferingStrategy)

UPLINK_ENUM_BEGIN(DroppingStrategy)
    DroppingStrategy_RandomOne,
    DroppingStrategy_OldestOne,
UPLINK_ENUM_END(DroppingStrategy)

//------------------------------------------------------------------------------

struct ChannelSettings : Serializable
{
# define MEMBERS() \
         MEMBER(bufferingStrategy) \
         MEMBER(droppingStrategy) \
         MEMBER(droppingThreshold)

    ChannelSettings ()
    {
        bufferingStrategy = BufferingStrategy_One;
        droppingStrategy = DroppingStrategy_RandomOne;
        droppingThreshold = 0; // No dropping.
    }

    bool operator != (const ChannelSettings& rhs) const
    {
#define MEMBER(Name) \
        return_true_if(Name != rhs.Name);
        MEMBERS()
#undef  MEMBER

        return false;
    }
    
    virtual bool serializeWith (Serializer& serializer)
    {
#define MEMBER(Name) \
        return_false_unless(serializer.put(Name));
        MEMBERS()
#undef  MEMBER
    
        return true;
    }

    String toString () const
    {
        String str = "ChannelSettings\n{\n";

#define MEMBER(Name) \
        str += #Name; \
        str += ": "; \
        str += ::uplink::toString(Name); \
        str += "\n";
        MEMBERS()
#undef  MEMBER
        str += "}";

        return str;
    }

    BufferingStrategy  bufferingStrategy;
    DroppingStrategy    droppingStrategy;
    uint16             droppingThreshold;

# undef MEMBERS
};

//------------------------------------------------------------------------------

// FIXME: Remove the hacky "lockColorCameraGainOnRecord".

# define UPLINK_SESSION_SETTING(Type, Name, name)
# define UPLINK_SESSION_SETTINGS() \
         UPLINK_SESSION_SETTING(   DepthMode               , DepthMode                  , depthMode) \
         UPLINK_SESSION_SETTING(   ColorMode               , ColorMode                  , colorMode) \
         UPLINK_SESSION_SETTING(InfraredMode               , InfraredMode               , infraredMode) \
         UPLINK_SESSION_SETTING(RegistrationMode           , RegistrationMode           , registrationMode) \
         UPLINK_SESSION_SETTING(FrameSyncMode              , FrameSyncMode              , frameSyncMode) \
         UPLINK_SESSION_SETTING(ChannelSettings            , DepthImageChannel          , depthImageChannel) \
         UPLINK_SESSION_SETTING(ChannelSettings            , ColorImageChannel          , colorImageChannel) \
         UPLINK_SESSION_SETTING(ChannelSettings            , InfraredImageChannel       , infraredImageChannel) \
         UPLINK_SESSION_SETTING(ChannelSettings            , RGBDFrameChannel           , rgbdFrameChannel) \
         UPLINK_SESSION_SETTING(ChannelSettings            , GrayDFrameChannel          , graydFrameChannel) \
         UPLINK_SESSION_SETTING(bool                       , SporadicFrameColor         , sporadicFrameColor) \
         UPLINK_SESSION_SETTING(uint8                      , SporadicFrameColorDivisor  , sporadicFrameColorDivisor) \
         UPLINK_SESSION_SETTING(bool                       , SendMotion                 , sendMotion) \
         UPLINK_SESSION_SETTING(bool                       , LockColorCameraGainOnRecord, lockColorCameraGainOnRecord) \
         UPLINK_SESSION_SETTING(ColorCameraExposureMode    , ColorCameraExposureMode    , colorCameraExposureMode) \
         UPLINK_SESSION_SETTING(ColorCameraWhiteBalanceMode, ColorCameraWhiteBalanceMode, colorCameraWhiteBalanceMode) \
         UPLINK_SESSION_SETTING(ImageCodecId               , DepthCameraCodec           , depthCameraCodec) \
         UPLINK_SESSION_SETTING(ImageCodecId               , ColorCameraCodec           , colorCameraCodec) \
         UPLINK_SESSION_SETTING(ImageCodecId               , FeedbackImageCodec         , feedbackImageCodec) \
         UPLINK_SESSION_SETTING(uint16                     , MotionRate                 , motionRate)
# undef  UPLINK_SESSION_SETTING

//------------------------------------------------------------------------------

struct SessionSettings : Serializable
{
    SessionSettings ();

# define UPLINK_SESSION_SETTING(Type, Name, name) \
    Type name;
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

    void clear ();

    String toString () const
    {
        String str = "SessionSettings\n{\n";
# define UPLINK_SESSION_SETTING(Type, Name, name) \
        str += #name; \
        str += ": "; \
        str += ::uplink::toString(name); \
        str += "\n";
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING
        str += "}";

        return str;
    }

    virtual bool serializeWith (Serializer& serializer);

# define UPLINK_SESSION_SETTING(Type, Name, name) \
    virtual bool set##Name (const Type& new##Name) \
    { \
        return_false_unless(modified(name, new##Name)); \
        \
        return true; \
    }
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING
};

//------------------------------------------------------------------------------

}

# include "./sessions-settings.hpp"
