//
//  network/servers.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./endpoints.h"

namespace uplink {

//------------------------------------------------------------------------------

class Server;
class ServerDelegate;
struct DesktopServerUI;

//------------------------------------------------------------------------------

struct ServerSession : Endpoint
{
public:
    ServerSession (int socketDescriptor, Server* server);

    void start ();
    void stop ();

    virtual void disconnected ();
    virtual void onSessionSetup (const SessionSetup& sessionSetup);

    virtual void onRemoteSessionSetupSuccess ()
    {
        uplink_log_info("Remote session setup successful.");
    }

    virtual void onRemoteSessionSetupFailure ()
    {
        uplink_log_error("Remote session setup failed.");
    }

    Server& server() { return *_server; }

private:
    int     _socketDescriptor;
    Wire*   _wire;
    Server* _server;

    virtual void onVersionInfo (const VersionInfo& clientVersionInfo)
    {
        uplink_log_info("Client version: %i.%i", clientVersionInfo.major, clientVersionInfo.minor);

        // Nothing else to do, for now.
    }

    virtual void onSessionSetupReply (const SessionSetupReply& reply)
    {
        switch (reply.status)
        {
            case SessionSetupStatus_Success:
                
                // Store the session id that will be subsequently used.
                currentSessionId = reply.remoteSessionId;

                // Apply the pending setup to the current settings.
                lastSessionSetup.applyTo(currentSessionSettings);

                reset();

                onRemoteSessionSetupSuccess();

                break;

            case SessionSetupStatus_Failure:

                // FIXME: Call back the server.
                // FIXME: Try other setups.

                onRemoteSessionSetupFailure();

                break;

            default:
                break;
        }
    }
};

// FIXME: Remove this once all the dependent code has been updated.
typedef ServerSession ServerSessionBase;

//------------------------------------------------------------------------------

class ServerDelegate
{
protected:
    virtual ~ServerDelegate () {}
    ServerDelegate () {}

public:
    virtual ServerSession* newSession (int socketDescriptor, Server* server) = 0;

    virtual void onConnect (uintptr_t sessionId) = 0;
    virtual void onDisconnect(uintptr_t sessionId) {}
};

//------------------------------------------------------------------------------

class Server : public ServicePublisher
{
public:
    Server (const std::string& serviceName, int servicePort, objc_weak ServerDelegate* serverDelegate);

    virtual ~Server ();

public:
    bool startListening (const char* ipAddress = 0);

    void onConnect (int socketDescriptor);
    void onDisconnect (int socketDescriptor, ServerSession* session);

protected:
    void clear();

private:
    class ConnectionListener;

private:
    ConnectionListener* _listener;
    int _port;
    std::vector<ServerSession*> _sessions;
    ServerDelegate* _serverDelegate;

public:
    ServerSession* _currentSession;
    
public:
    void* userdata;
};

//------------------------------------------------------------------------------

}

# include "./servers.hpp"
