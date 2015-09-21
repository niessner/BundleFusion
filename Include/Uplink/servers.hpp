//
//  network/servers.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./servers.h"

namespace uplink {

//------------------------------------------------------------------------------

class Server::ConnectionListener : public Thread
{
public:
    ConnectionListener (const char* ipAddress, int port, Server* server);
    virtual ~ConnectionListener ();

public:
    bool startListening ();

private:
    virtual void run ();

private:
    Server* _server;
    int _port;
    bool _startedListening;
    bool _isListening;
    bool _shouldStopListening;
    struct sockaddr_in _local;

# if _WIN32
    SOCKET sockd;
# else
    int sockd;
# endif
};

//------------------------------------------------------------------------------

inline
Server::ConnectionListener::ConnectionListener (const char* ipAddress, int port, Server* server)
: _server (server)
, _port (port)
, _startedListening(true)
, _isListening(false)
, _shouldStopListening(false)
{
    assert(0 != _server);

    if (ipAddress)
        _local.sin_addr.s_addr = inet_addr(ipAddress);
    else
        _local.sin_addr.s_addr = INADDR_ANY;
    _local.sin_family = AF_INET;
    _local.sin_port = htons(_port);
}

inline
Server::ConnectionListener::~ConnectionListener ()
{}

inline bool
Server::ConnectionListener::startListening ()
{
    int status;

    /* create a nonblocking-socket */

# if _WIN32
    sockd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
# else
    sockd = socket(AF_INET, SOCK_STREAM, 0);
# endif

# if _WIN32
    if (INVALID_SOCKET == sockd)
# else
    if (-1 == sockd)
# endif
    {
# if _WIN32
        last_socket_error();
# else
        perror("Socket creation error");
# endif

        return false;
    }

    status = ::bind(sockd, (struct sockaddr*)&_local, sizeof(_local));
    if (status == -1)
    {
# if _WIN32
        last_socket_error();
# else
        perror("Binding error");
# endif

        return false;
    }

    uplink_log_info("Server started listening.");

    status = listen(sockd, 5);
    if (status == -1)
    {
# if _WIN32
        last_socket_error();
# else
        perror("Listening error");
# endif
        return false;
    }

# if _WIN32
    u_long val = 1;
    ioctlsocket(sockd, FIONBIO, &val);
# else
    // set listen to be non-blocking
    int socketFlags = fcntl(sockd, F_GETFL, 0);
    fcntl(sockd, F_SETFL, socketFlags | O_NONBLOCK);
# endif

    _startedListening = true;

    start(); // Start the thread.

    return true;
}

inline void
Server::ConnectionListener::run ()
{
    assert(_startedListening); // Call ConnectionListener::startListening instead of Thread::start, and check on its return value.

    _isListening = true;
    _shouldStopListening = false;

    struct sockaddr_in peer_name;

# if _WIN32
    int addrlen;
# else
    unsigned int addrlen;
# endif

    /* wait for a connection */
    addrlen = sizeof(peer_name);

# if _WIN32
    SOCKET newServerSocket = INVALID_SOCKET;
# else
    int newServerSocket = -1;
# endif

    while (isRunning())
    {
        newServerSocket = accept(sockd, (struct sockaddr*)&peer_name, &addrlen);
# if _WIN32
        if (INVALID_SOCKET == newServerSocket)
# else
        if (-1 == newServerSocket)
# endif
        {
# if _WIN32
            Sleep(500);
# else
            usleep(500000);
# endif
        }
        else
        {
            _server->onConnect(int(newServerSocket));
        }
    }

# if _WIN32
    closesocket(sockd);
# else
    close (sockd);
# endif

    uplink_log_info("Server stopped listening.");
    _isListening = false;
}

//------------------------------------------------------------------------------

inline
ServerSession::ServerSession (int socketDescriptor, Server* server)
: _socketDescriptor (socketDescriptor)
, _server(server)
{
    assert(0 != _server);

    imageCodecs.jpeg.compressInputFormat = ImageFormat_RGB;
    imageCodecs.jpeg.compress   = compress_image_RGB_JPEG;
    imageCodecs.jpeg.decompressOutputFormat = ImageFormat_RGB;
    imageCodecs.jpeg.decompress = decompress_image_JPEG_RGB;

    _wire = new Wire(TCPConnection::create(_socketDescriptor), this);
}

inline void
ServerSession::start ()
{
    assert(0 != _wire);

    _wire->start();
}

inline void
ServerSession::stop ()
{
    assert(0 != _wire);

    _wire->stop();
}

inline void
ServerSession::disconnected ()
{
    uplink_log_info("Server session disconnected.");

    _server->onDisconnect(_socketDescriptor, this);
}

inline void
ServerSession::onSessionSetup (const SessionSetup& sessionSetup)
{
    // Nothing to do.
}

//------------------------------------------------------------------------------

inline
Server::Server (const std::string& serviceName, int servicePort, objc_weak ServerDelegate* serverDelegate)
: ServicePublisher(servicePort)
, _listener (0)
, _port (servicePort)
, _serverDelegate (serverDelegate)
, _currentSession(0)
{
    ServiceEntry entry (serviceName, servicePort, VersionRange(
        UPLINK_SERVER_MINIMUM_VERSION_MAJOR,
        UPLINK_SERVER_MINIMUM_VERSION_MINOR,
        UPLINK_SERVER_MINIMUM_VERSION_MAJOR,
        UPLINK_SERVER_MINIMUM_VERSION_MINOR)
                                );
    serviceList.push_back (entry);
    ServicePublisher::start ();

    assert(0 != serverDelegate);
}

inline
Server::~Server ()
{
    clear();
}

inline void
Server::clear ()
{
    for (int i = 0; i < _sessions.size(); ++i)
    {
        _sessions[i]->stop();
        delete _sessions[i];
    }
    _sessions.clear();

    delete _listener; _listener = 0;
}

inline bool
Server::startListening (const char* ipAddress)
{
    if (_listener)
        delete _listener;

    _listener = new ConnectionListener (ipAddress, _port, this);

    return _listener->startListening ();
}

// Callback from the listener
inline void
Server::onConnect (int socketDescriptor)
{
    //session->colorImageDropping = false;
    //session->colorImageQueue.setLimit(90);

    if (_currentSession && _currentSession->isConnected ())
    {
        uplink_log_info("New connection, but we are already talking to someone, closing it.\n");
        std::string msg = "Sorry but a client is already connected!\n";
# if _WIN32
        send (socketDescriptor, msg.c_str(), int(msg.size()), 0);
        closesocket (socketDescriptor);
# else
        write (socketDescriptor, msg.c_str(), msg.size());
        close (socketDescriptor);
# endif
        return;
    }

    ServerSession* session = _serverDelegate->newSession(socketDescriptor, this);

    assert(0 != session);

    session->start ();
    _sessions.push_back (session);

    _currentSession = session;

    uplink_log_info("Incoming client.");

    uplink_log_info("Disabling service publishing.");
    setBroadcastingEnabled(false);

    _serverDelegate->onConnect(socketDescriptor);

    // Do not request the desired session setup yet, we don't know which program will call this.
    //    StructureEngineSessionSetup sessionSetup;
    //    _currentSession->sendSessionSetup(sessionSetup);
}

// FIXME: this will never be called!
inline void
Server::onDisconnect (int socketDescriptor, ServerSession* session)
{
    uplink_log_info("Client disconnected.");

    // Restart service publishing since we don't have a client anymore.
    uplink_log_info("Enabling service publishing.");
    setBroadcastingEnabled (true);

    if (_serverDelegate)
        _serverDelegate->onDisconnect(uintptr_t(session));

# if _WIN32
    closesocket (socketDescriptor);
# else
    close (socketDescriptor);
# endif
}

//------------------------------------------------------------------------------

}
