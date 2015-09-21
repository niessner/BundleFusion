//
//  network/backends/bsd-sockets.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./bsd-sockets.h"
# include <unistd.h>
# include <fcntl.h>
# include <arpa/inet.h>
# include <netinet/tcp.h>
# include <unistd.h>
# include <netdb.h>
# include <errno.h>
# include <cstdio>
# include <cstring>
# include <iostream>
# include <cassert>

# define ERROR_MESSAGE(Str) (formatted(Str ": %s", strerror(errno)).get())

namespace uplink {

//------------------------------------------------------------------------------

inline int
select_single_read (int fd, int timeout_ms)
{
    ::fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    ::fd_set errs;
    FD_ZERO(&errs);
    FD_SET(fd, &errs);

    timeval timeout;
    timeout.tv_sec  = timeout_ms / 1000;
    timeout.tv_usec = 1000 * (timeout_ms % 1000);

    const int ret = ::select(fd + 1, &fds, 0, &errs, &timeout);

    if (FD_ISSET(fd, &errs))
        return -1;

    return ret;
}

inline int
select_single_write (int fd, int timeout_ms)
{
    ::fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);

    ::fd_set errs;
    FD_ZERO(&errs);
    FD_SET(fd, &errs);

    timeval timeout;
    timeout.tv_sec  = timeout_ms / 1000;
    timeout.tv_usec = 1000 * (timeout_ms % 1000);

    const int ret = ::select(fd + 1, 0, &fds, &errs, &timeout);

    if (FD_ISSET(fd, &errs))
        return -1;

    return ret;
}

//------------------------------------------------------------------------------

inline bool
BSDSocket::enableOption (int option)
{
    int val = 1;

    report_false_if_non_zero("Cannot set socket option.", setsockopt(fd, SOL_SOCKET, option, &val, sizeof val));

    return true;
}

inline bool
BSDSocket::setTimeOption (int option, int sec, int usec)
{
    struct timeval tv;
    tv.tv_sec  = sec;
    tv.tv_usec = usec;

    int result = setsockopt(fd, SOL_SOCKET, option, (char*) &tv, sizeof(struct timeval));
    report_false_if("set socket time option", result < 0);

    return true;
}

//------------------------------------------------------------------------------

inline UDPBroadcaster*
UDPBroadcaster_create (int port)
{
    std::auto_ptr<BSDUDPBroadcaster> ret(new BSDUDPBroadcaster());

    assert(0 < port && port < 65536);

    ret->socket.fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

    report_zero_if("Failed to create UDP broadcaster socket.", -1 == ret->socket.fd);

    return_zero_unless(ret->socket.enableOption(SO_REUSEADDR));
    return_zero_unless(ret->socket.enableOption(SO_REUSEPORT));
    return_zero_unless(ret->socket.enableOption(SO_BROADCAST));

    std::memset(&ret->socket.addr, 0, sizeof(ret->socket.addr));
    ret->socket.addr.sin_family      = AF_INET;
    ret->socket.addr.sin_addr.s_addr = INADDR_BROADCAST;
    ret->socket.addr.sin_port        = htons(port);

    return ret.release();
}

inline bool
BSDUDPBroadcaster::broadcast (const Byte* bytes, Size length)
{
    assert(0 != bytes);

    ssize_t howmany = sendto(socket.fd, bytes, length, 0, (sockaddr*) &socket.addr, sizeof socket.addr);

    if (howmany != length)
    {
        // Also try ad-hoc broadcast.
        inet_aton("169.254.255.255", &socket.addr.sin_addr);
        howmany = sendto(socket.fd, bytes, length, 0, (sockaddr*) &socket.addr, sizeof socket.addr);

        // Reset to default broadcast.
        socket.addr.sin_addr.s_addr = INADDR_BROADCAST;
    }

    report_false_unless("Cannot broadcast datagram.", howmany == length);

    return true;
}

//------------------------------------------------------------------------------

inline UDPListener*
UDPListener_create (int port)
{
    std::auto_ptr<BSDUDPListener> ret(new BSDUDPListener());

    assert(0 < port && port < 65536);

    ret->socket.fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

    report_zero_if("Failed to create UDP listener socket.", -1 == ret->socket.fd);

    return_zero_unless(ret->socket.enableOption(SO_REUSEADDR));
    return_zero_unless(ret->socket.enableOption(SO_REUSEPORT));

    // Set a timeout on the listener socket.
    // This will allow the listener thread to properly exit instead of blocking on read forever.
    return_zero_unless(ret->socket.setTimeOption(SO_RCVTIMEO, 2, 0)); // Two seconds is an arbitrary choice.
    return_zero_unless(ret->socket.setTimeOption(SO_SNDTIMEO, 2, 0)); // Likewise.

    std::memset(&ret->socket.addr, 0, sizeof ret->socket.addr);
    ret->socket.addr.sin_family      = AF_INET;
    ret->socket.addr.sin_addr.s_addr = INADDR_ANY;
    ret->socket.addr.sin_port        = htons(port);

    report_zero_if("Cannot bind datagram listener socket.", -1 == ::bind(ret->socket.fd, (sockaddr*) &ret->socket.addr, sizeof ret->socket.addr));

    return ret.release();
}

inline bool
BSDUDPListener::receive (Byte* bytes, Size size, NetworkAddress& sender)
{
    assert(0 != bytes);
    assert(0 < size);

    // FIXME: Generalize this.
    const bool valid = fcntl(socket.fd, F_GETFD) != -1 || errno != EBADF;
    report_false_if( "udp listening socket is invalid", !valid );

    sockaddr_in origin;
    socklen_t   origin_size = sizeof origin;

    ssize_t howmany = recvfrom(socket.fd, bytes, size, MSG_WAITALL, (sockaddr*) &origin, &origin_size);

    // Now that the recvfrom is on a timeout, we don't need to write errors to console if the size comes back as -1.
    return_false_if(-1 == howmany);

    sender = origin.sin_addr.s_addr;

    return true;
}

//------------------------------------------------------------------------------

inline TCPConnection*
TCPConnection_connect (CString host, uint16 port)
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);

    report_zero_if(ERROR_MESSAGE("socket"), fd == -1);

    // Make it so that we can early-return from this scope using one-liners.
    struct Cleanup
    {
        explicit Cleanup (int fd)
            : ok(false), fd(fd)
        {}

        ~Cleanup ()
        {
            return_if(ok);

            close(fd);
        }

        bool ok;
        int  fd;
    }
    cleanup(fd);

    // FIXME: Use the IPv6-compliant getaddrinfo.
    struct hostent* ip = ::gethostbyname(host);

    report_zero_if(ERROR_MESSAGE("gethostbyname"), ip == NULL);

    sockaddr_in address;
    address.sin_family = AF_INET;
    memcpy(&address.sin_addr, ip->h_addr_list[0], ip->h_length);
    address.sin_port = htons(port);

    {
        // Put the socket into non-blocking mode.

        int flags = fcntl(fd, F_GETFL, 0);
        int status = fcntl(fd, F_SETFL, flags | O_NONBLOCK);

        report_zero_if(ERROR_MESSAGE("fcntl"), status < 0);
    }

    {
        // Perform the non-blocking connection dance.
    
        int status = 0;
        do
        {
            status = ::connect(fd, (sockaddr*) &address, sizeof(address));
            
        }
        while (-1 == status && EINTR == errno); // In the extreme case where connect got interrupted by a signal.

        if (0 != status) // TCPConnection did not immediately succeed.
        {
            switch (errno) // Investigate the error, and take action.
            {
                case EINPROGRESS: // The connection is in progress.
                {
                    static const int connectionTimeoutInMilliseconds = 500;
                
                    // Poll the socket for writing, with a timeout.
                    const int ret = select_single_write(fd, connectionTimeoutInMilliseconds);

                    // Select might actually have returned EINPROGRESS, here.
                    // Currently, we behave as if the connection failed when it times out.
                    // FIXME: Introduce an asynchronous callback connection mechanism in parallel to this blocking one.

                    // Abort if select failed.
                    report_zero_unless(ERROR_MESSAGE("select"), 0 < ret);
                    
                    // Also abort if select returned with a descriptor not ready for writing.
                    report_zero_if("Connection timed out.", 0 == ret);

                    // Make sure that the connection is error-free.
                    int       err = 0;
                    socklen_t len = sizeof(int);
                    report_zero_unless(ERROR_MESSAGE("getsockopt"), 0 == getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len));
                    report_zero_unless(formatted("connect: %s", strerror(err)).get(), 0 == err);

                    break; // The connection is now established.
                }
                
                default: // The connection failed for other reasons.
                {
                    uplink_log_error(ERROR_MESSAGE("connect"));

                    return 0;
                }
            }
        }
    }

    {
        // Put the socket back into blocking mode.
        
        int flags = fcntl(fd, F_GETFL, 0);
        int status = fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);

        report_zero_if(ERROR_MESSAGE("fcntl"), status < 0);
    }

    cleanup.ok = true;

    return TCPConnection_create(fd);
}

inline TCPConnection*
TCPConnection_create (int descriptor)
{
    assert(0 < descriptor);

    // The passed descriptor might have non-default socket options set.

    // Enforce blocking reads and writes.
    int flags = fcntl(descriptor, F_GETFL, 0);
    int status = fcntl(descriptor, F_SETFL, flags & ~O_NONBLOCK);

    report_zero_if(ERROR_MESSAGE("fcntl"), status < 0);

    // Ignore the SIGPIPE signal.
    int val = 1;
    setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &val, sizeof(int));


    // Set TCP_NODELAY in hopes of having lower latency streaming.
    int flag = 1;
    int result = setsockopt(descriptor,       /* socket descriptor */
                             IPPROTO_TCP,     /* set option at TCP level */
                             TCP_NODELAY,     /* name of option */
                             (char *) &flag,  /* the cast is historical cruft */
                             sizeof(int));    /* length of option value */

    report_zero_if(ERROR_MESSAGE("set TCP nodelay socket option"), result < 0);

    {
        struct timeval tv;
        tv.tv_sec  = 2;
        tv.tv_usec = 0;

        result = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, (char*) &tv, sizeof(struct timeval));
        report_zero_if(ERROR_MESSAGE("set socket receive timeout"), result < 0);

        result = setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, (char*) &tv, sizeof(struct timeval));
        report_zero_if(ERROR_MESSAGE("set socket send timeout"), result < 0);
    }

    return new BSDTCPConnection(descriptor);
}

//------------------------------------------------------------------------------

inline
BSDTCPConnection::BSDTCPConnection (int descriptor)
{
    assert(0 < descriptor);

    socket.fd = descriptor;
}

inline
BSDTCPConnection::~BSDTCPConnection ()
{
    close(socket.fd);
}

inline bool
BSDTCPConnection::read (Byte* bytes, Size size)
{
    assert(0 != bytes);
    assert(0 < size);

    do
    {
        ScopedProfiledTask _(ProfilerTask_SocketReadLoop);
    
        report_false_if("connection: read: disconnecting", disconnecting);

        const int status = select_single_read(socket.fd, 100);

        if (0 == status)
        {
            continue; // Nothing to read, yet.
        }
        else if (status < 0)
        {
            uplink_log_error("connection: read: select: %s", strerror(errno));
            return false;
        }

        const ssize_t count = ::read(socket.fd, bytes, size);

        if (0 == count)
        {
            uplink_log_error("connection: read: EOF");
            return false;
        }
        else if (count < 0)
        {
            uplink_log_error("connection: read: %s", strerror(errno));
            return false;
        }

        size  -= count;
        bytes += count;
    }
    while (0 < size);

    return true;
}

inline bool
BSDTCPConnection::write (const Byte* bytes, Size size)
{
    assert(0 != bytes);
    assert(0 < size);

    do
    {
        ScopedProfiledTask _(ProfilerTask_SocketWriteLoop);
    
        report_false_if("connection: write: disconnecting", disconnecting);

        const int status = select_single_write(socket.fd, 100);

        if (0 == status)
        {
            continue; // Cannot write just yet.
        }
        else if (status < 0)
        {
            uplink_log_error("connection: write: select failed: %s", strerror(errno));

            return false;
        }

        const ssize_t count = ::write(socket.fd, bytes, size);

        if (0 == count)
        {
            uplink_log_error("connection: write failed: EOF");

            return false;
        }
        else if (count < 0)
        {
            uplink_log_error("connection: write failed: %s", strerror(errno));

            return false;
        }

        size  -= count;
        bytes += count;
    }
    while (0 < size);

    return true;
}

//------------------------------------------------------------------------------

}

# undef ERROR_MESSAGE
