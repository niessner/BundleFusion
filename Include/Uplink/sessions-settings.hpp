//
//  network/sessions-settings.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital. All rights reserved.
//

# pragma once

# include "./sessions-settings.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
SessionSettings::SessionSettings ()
{
    // Default to the minimum.

    depthMode = DepthMode_None;

    colorMode = ColorMode_None;

    infraredMode = InfraredMode_None;

    registrationMode = RegistrationMode_None;

    frameSyncMode = FrameSyncMode_None;

    sporadicFrameColor = false;
    sporadicFrameColorDivisor = 1;

    sendMotion = false;
    motionRate = 50;

    lockColorCameraGainOnRecord = false;

    colorCameraExposureMode     = ColorCameraExposureMode_ContinuousAuto;
    colorCameraWhiteBalanceMode = ColorCameraWhiteBalanceMode_ContinuousAuto;

    // Reset to safe defaults. Wrong image formats have happened and will happen again, otherwise.
    depthCameraCodec = ImageCodecId_Invalid;
    colorCameraCodec = ImageCodecId_Invalid;
    feedbackImageCodec = ImageCodecId_Invalid;

    // Channel settings are initialized in their default-constructor.
}

inline void
SessionSettings::clear ()
{
    *this = SessionSettings();
}

inline bool
SessionSettings::serializeWith (Serializer& serializer)
{
# define UPLINK_SESSION_SETTING(Type, Name, name) \
    return_false_unless(serializer.put(name));
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

    return true;
}

//------------------------------------------------------------------------------

}
