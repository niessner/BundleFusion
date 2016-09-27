//
//  network/sessions-setup-presets.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./sessions-setup.h"

namespace uplink {

//------------------------------------------------------------------------------

struct SporadicColorSessionSetup : SessionSetup
{
    SporadicColorSessionSetup()
    {
        addSetColorModeAction(ColorMode_VGA);
        addSetDepthModeAction(DepthMode_VGA);
        addSetRegistrationModeAction(RegistrationMode_RegisteredDepth);
        addSetFrameSyncModeAction(FrameSyncMode_Depth);

        addSetSporadicFrameColorAction(true);
        addSetSporadicFrameColorDivisorAction(6);

        ChannelSettings channelSettings;
        channelSettings.droppingStrategy = DroppingStrategy_RandomOne;
        channelSettings.droppingThreshold = 90;
        channelSettings.bufferingStrategy = BufferingStrategy_Some;

        addSetRGBDFrameChannelAction(channelSettings);

        addSetSendMotionAction(false);
        addSetMotionRateAction(100);

        addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
        addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);
    }
};

//------------------------------------------------------------------------------

struct Depth60FPSSessionSetup : SessionSetup
{
    Depth60FPSSessionSetup()
    {
        addSetColorModeAction(ColorMode_None);
        addSetDepthModeAction(DepthMode_QVGA_60FPS);
        addSetRegistrationModeAction(RegistrationMode_None);
        addSetFrameSyncModeAction(FrameSyncMode_None);

        addSetSporadicFrameColorAction(false);
        addSetSporadicFrameColorDivisorAction(6);

        ChannelSettings channelSettings;
        channelSettings.droppingStrategy = DroppingStrategy_RandomOne;
        channelSettings.droppingThreshold = 90;
        channelSettings.bufferingStrategy = BufferingStrategy_Some;

        addSetRGBDFrameChannelAction(channelSettings);

        addSetSendMotionAction(false);
        addSetMotionRateAction(100);

        addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
        addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);
    }
};

//------------------------------------------------------------------------------

struct WXGASessionSetup : SessionSetup
{
    WXGASessionSetup()
    {
        addSetColorModeAction(ColorMode_1296x968);
        addSetDepthModeAction(DepthMode_VGA);
        addSetRegistrationModeAction(RegistrationMode_RegisteredDepth);
        addSetFrameSyncModeAction(FrameSyncMode_Depth);

        addSetSporadicFrameColorAction(false);
        addSetSporadicFrameColorDivisorAction(6);

        ChannelSettings channelSettings;
        channelSettings.droppingStrategy = DroppingStrategy_RandomOne;
        channelSettings.droppingThreshold = 90;
        channelSettings.bufferingStrategy = BufferingStrategy_Some;

        addSetRGBDFrameChannelAction(channelSettings);

        addSetSendMotionAction(false);
        addSetMotionRateAction(100);

        addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
        addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);
    }
};

//------------------------------------------------------------------------------

}

# include "./sessions-setup-presets.hpp"
