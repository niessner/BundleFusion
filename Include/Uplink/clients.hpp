//
//  network/clients.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./clients.h"

namespace uplink {

//------------------------------------------------------------------------------

inline void
ClientEndpoint::onVersionInfo (const VersionInfo& clientVersionInfo)
{
    uplink_log_info("Server version: %i.%i", clientVersionInfo.major, clientVersionInfo.minor);

    // Nothing else to do, for now.
}

inline void
ClientEndpoint::onSessionSetup(const uplink::SessionSetup &sessionSetup)
{
    SessionSettings nextSessionSettings = currentSessionSettings;

    if (!sessionSetup.applyTo(nextSessionSettings))
        return; // Session settings did not change.
    
    if (!setupSession(nextSessionSettings))
    {
        onSessionSetupFailure();

        sendSessionSetupReply(SessionSetupReply(InvalidSessionId, SessionSetupStatus_Failure));

        uplink_log_error("Session setup failed.");
    }
    else
    {
        static unsigned sessionCounter = 0;
        
        // Next messages will be sent and received as part of the new session.
        // FIXME: Let the remote side choose the id and return that.
        currentSessionId = FirstSessionId + sessionCounter++;

        currentSessionSettings = nextSessionSettings;

        sendSessionSetupReply(SessionSetupReply(currentSessionId, SessionSetupStatus_Success));
    
        onSessionSetupSuccess();

        uplink_log_info("Session setup succeeded.");
    }
}

inline bool
ClientEndpoint::setupSession (const SessionSettings& nextSessionSettings)
{
    setAllChannelSettings(nextSessionSettings);
    
    return true;
}

inline void
ClientEndpoint::reset ()
{
    Endpoint::reset();

    currentSessionSettings.clear();

    uplink_log_debug("Client endpoint reset.");
}

//------------------------------------------------------------------------------

}
