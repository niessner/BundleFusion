//
//  network/backends/windows-sockets.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./windows-sockets.h"
# include <cstdio>
# include <cstring>
# include <iostream>

namespace uplink {

inline void
last_socket_error()
{
    int lastError = WSAGetLastError();

    wchar_t* s = 0;
    FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        lastError,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPWSTR)&s,
        0,
        NULL
        );
    fprintf(stderr, "%S\n", s);
    LocalFree(s);
}

//------------------------------------------------------------------------------

inline int
select_single_read (SOCKET fd, int timeout_ms)
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

    const int ret = ::select(0 /*ignored*/, &fds, 0, &errs, &timeout);

    if (FD_ISSET(fd, &errs))
        return -1;

    return ret;

    return 1;
}

inline int
select_single_write (SOCKET fd, int timeout_ms)
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

    const int ret = ::select(0 /*ignored*/, 0, &fds, &errs, &timeout);

    if (FD_ISSET(fd, &errs))
        return -1;

    return ret;

    return 1;
}

//------------------------------------------------------------------------------

inline bool
WindowsSocket::enableOption (int option)
{
    int val = 1;

    report_false_if_non_zero("Cannot set socket option.",
        setsockopt(fd, SOL_SOCKET, option, reinterpret_cast<const char*>(&val), sizeof val));

    return true;
}

inline bool
WindowsSocket::setTimeOption (int option, int msec)
{
    DWORD val = msec;

    report_false_if_non_zero("Cannot set socket time option",
        setsockopt(fd, SOL_SOCKET, option, reinterpret_cast<const char*>(&val), sizeof val));

    return true;
}

//------------------------------------------------------------------------------

inline TCPConnection*
TCPConnection_connect (CString host, uint16 port)
{
//    const int fd = socket(AF_INET, SOCK_STREAM, 0);

//    report_zero_if("socket", fd == -1);

//    // FIXME: Use the IPv6-compliant getaddrinfo.
//    struct hostent* ip = ::gethostbyname(host);

//    report_zero_if("gethostbyname", ip == NULL);

//    sockaddr_in address;
//    address.sin_family = AF_INET;
//    memcpy(&address.sin_addr, ip->h_addr_list[0], ip->h_length);
//    address.sin_port = htons(port);

//    int status = ::connect(fd, (sockaddr*) &address, sizeof(address));

//    report_zero_if("connect", status < 0);

//    return TCPConnection_create(fd);

    return 0; // FIXME: Implement.
}

inline TCPConnection*
TCPConnection_create (int descriptor)
{   
    assert(INVALID_SOCKET != descriptor);

    // The passed descriptor might have non-default socket options set.

    // Ensure that sockets are in blocking mode.
    {
        unsigned long val = 0;
        report_zero_if("disable non-blocking socket mode", SOCKET_ERROR == ioctlsocket(descriptor, FIONBIO, &val));
    }

    // Set TCP_NODELAY in hopes of having lower latency streaming.
    {
        BOOL val = 1;
        int result = setsockopt(descriptor, IPPROTO_TCP, TCP_NODELAY, (char*) &val, sizeof val);
        report_zero_if("set TCP nodelay socket option", result == SOCKET_ERROR);
    }

    // Set read & write timeouts.
    {
        DWORD val = 2000; // 2 seconds timeout.
        int result = setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, (char*) &val, sizeof val);
        report_zero_if("set socket receive timeout", result == SOCKET_ERROR);
        result = setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, (char*) &val, sizeof val);
        report_zero_if("set socket send timeout", result == SOCKET_ERROR);
    }

    // Set send and receive buffer sizes.
    if (false) // This is a generally bad idea. See: http://msdn.microsoft.com/en-us/library/windows/desktop/ms738551
    {
        int val = 256; // Buffer size, in bytes.
        int result = setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, (char*) &val, sizeof val);
        report_zero_if("set socket receive buffer size", result == SOCKET_ERROR);
        result = setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, (char*) &val, sizeof val);
        report_zero_if("set socket send buffer size", result == SOCKET_ERROR);
    }

    return new WindowsTCPConnection(descriptor);
}

//------------------------------------------------------------------------------

inline
WindowsTCPConnection::WindowsTCPConnection (int descriptor)
{
    assert(INVALID_SOCKET != descriptor);

    socket.fd = descriptor;
}

inline
WindowsTCPConnection::~WindowsTCPConnection ()
{
    closesocket(socket.fd);
}

inline bool
WindowsTCPConnection::read (Byte* bytes, Size size)
{
    assert(0 != bytes);
    assert(0 < size);

    do
    {
        report_false_if("TCPConnection: read: disconnecting", disconnecting);

        const int status = select_single_read(socket.fd, 100);

        if (0 == status)
        {
            uplink_log_debug("TCPConnection: Nothing to read, yet.");
            continue;
        }
        else if (status < 0)
        {
            uplink_log_error("TCPConnection: select: %s", strerror(errno));
            return false;
        }

//        const ssize_t count = ::read(descriptor, bytes, size);
        const int count = recv(socket.fd, reinterpret_cast<char*>(bytes), int(size), 0);

        if (0 == count)
        {
            uplink_log_error("TCPConnection: read: EOF");
            return false;
        }
        else if (count == SOCKET_ERROR)
        {
            uplink_log_error("TCPConnection: read: %s", strerror(errno));
            return false;
        }

        size  -= count;
        bytes += count;

        uplink_log_debug("TCPConnection: Received %i bytes", count);
    }
    while (0 < size);

    return true;
}

inline bool
WindowsTCPConnection::write (const Byte* bytes, Size size)
{
    assert(0 != bytes);
    assert(0 < size);

    do
    {
        report_false_if("TCPConnection: write: disconnecting", disconnecting);

        const int status = select_single_write(socket.fd, 100);

        if (0 == status)
        {
            uplink_log_debug("TCPConnection: Cannot write, yet.");
            continue;
        }
        else if (status < 0)
        {
            uplink_log_error("TCPConnection: select: %s", strerror(errno));
        }

        // const ssize_t count = ::write(descriptor, bytes, size);
        const int count = send(socket.fd, reinterpret_cast<const char*>(bytes), int(size), 0);

        if (0 == count)
        {
            uplink_log_error("TCPConnection: write: EOF");
            return false;
        }
        else if (count < 0)
        {
            uplink_log_error("TCPConnection: write: %s", strerror(errno));
            return false;
        }

        size  -= count;
        bytes += count;

        uplink_log_debug("TCPConnection: Sent %i bytes.", count);
    }
    while (0 < size);

    return true;
}

//------------------------------------------------------------------------------

inline UDPBroadcaster*
UDPBroadcaster_create (int port)
{
    std::auto_ptr<WindowsUDPBroadcaster> ret(new WindowsUDPBroadcaster());

    assert(0 < port && port < 65536);

    ret->socket.fd = WSASocket(PF_INET, SOCK_DGRAM, IPPROTO_UDP, 0, 0, 0);

    report_zero_if("Failed to create UDP broadcaster socket.", INVALID_SOCKET == ret->socket.fd);

    return_zero_unless(ret->socket.enableOption(SO_REUSEADDR));
    // SO_REUSEPORT is unavailable on Windows, but SO_REUSEADDR seems to do the trick.
    // See: https://groups.google.com/d/msg/alt.winsock.programming/f5AO_zTUn3I/M3ccZKgcne8J
    // FIXME: Investigate further.
    return_zero_unless(ret->socket.enableOption(SO_BROADCAST));

    memset(&ret->socket.addr, 0, sizeof(ret->socket.addr));
    ret->socket.addr.sin_family      = AF_INET;
    ret->socket.addr.sin_addr.s_addr = INADDR_BROADCAST;
    ret->socket.addr.sin_port        = htons(port);

    return ret.release();
}

inline bool
WindowsUDPBroadcaster::broadcast (const Byte* bytes, Size length)
{
    assert(0 != bytes);

    int howmany = sendto(socket.fd, reinterpret_cast<const char*>(bytes), int(length), 0, (sockaddr*) &socket.addr, sizeof socket.addr);

    // FIXME: Make windows ad-hoc networking work.
//    if (howmany != length)
//    {
//        // Also try ad-hoc broadcast.
//        inet_aton("169.254.255.255", &socket.addr.sin_addr);
//        howmany = sendto(socket.fd, bytes, length, 0, (sockaddr*) &socket.addr, sizeof socket.addr);

//        // Reset to default broadcast.
//        addr.sin_addr.s_addr = INADDR_BROADCAST;
//    }

    report_false_unless("Cannot broadcast datagram.", howmany == length);

    return true;
}

//------------------------------------------------------------------------------

inline UDPListener*
UDPListener_create (int port)
{
    std::auto_ptr<WindowsUDPListener> ret(new WindowsUDPListener());

    assert(0 < port && port < 65536);

    ret->socket.fd = WSASocket(PF_INET, SOCK_DGRAM, IPPROTO_UDP, 0, 0, 0);

    report_zero_if("Failed to create UDP listener socket.", INVALID_SOCKET == ret->socket.fd);

    return_zero_unless(ret->socket.enableOption(SO_REUSEADDR));
    // FIXME: See above comment about SO_REUSEPORT.

    // Set a timeout on the listener socket.
    // This will allow the listener thread to properly exit instead of blocking on read forever.
    return_zero_unless(ret->socket.setTimeOption(SO_RCVTIMEO, 2000)); // Two seconds is an arbitrary choice.
    return_zero_unless(ret->socket.setTimeOption(SO_SNDTIMEO, 2000)); // Likewise.

    memset(&ret->socket.addr, 0, sizeof ret->socket.addr);
    ret->socket.addr.sin_family      = AF_INET;
    ret->socket.addr.sin_addr.s_addr = INADDR_ANY;
    ret->socket.addr.sin_port        = htons(port);

    report_zero_if("Cannot bind datagram listener socket.",
        SOCKET_ERROR == ::bind(ret->socket.fd, (sockaddr*) &ret->socket.addr, sizeof ret->socket.addr));

    return ret.release();
}

inline bool
WindowsUDPListener::receive (Byte* bytes, Size size, NetworkAddress& sender)
{
    assert(0 != bytes);
    assert(0 < size);

    // FIXME: Generalize this.
    // FIXME: Make this work on windows.
//    const bool valid = fcntl(fd, F_GETFD) != -1 || errno != EBADF;
//    report_false_if( "udp listening socket is invalid", !valid );

    sockaddr_in origin;
    socklen_t   origin_size = sizeof origin;

    int howmany = recvfrom(socket.fd, reinterpret_cast<char*>(bytes), int(size), MSG_WAITALL, (sockaddr*) &origin, &origin_size);

    // Now that the recvfrom is on a timeout, we don't need to write errors to console if the size comes back as -1.
    return_false_if(SOCKET_ERROR == howmany);

    sender = origin.sin_addr.s_addr;

    return true;
}

//------------------------------------------------------------------------------

}
