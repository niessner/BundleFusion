//
//  network/services.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/datagrams.h"
# include "./core/threads.h"
# include <cstring>
# include <vector>

namespace uplink {

//------------------------------------------------------------------------------

struct VersionInfo : Message
{
    UPLINK_MESSAGE_CLASS(VersionInfo)

    VersionInfo ()
    : major(0)
    , minor(0)
    {

    }

    VersionInfo (uint8 major, uint8 minor)
    : major(major), minor(minor)
    {

    }

    virtual bool serializeWith (Serializer& s)
    {
        return_false_unless(s.put(major));
        return_false_unless(s.put(minor));

        return true;
    }

    bool isValid () const
    {
        return major != 0 || minor != 0;
    }

    bool operator == (const VersionInfo& rhs) const
    {
        return major == rhs.major && minor == rhs.minor;
    }

    bool operator < (const VersionInfo& rhs) const
    {
        return major == rhs.major
            ? minor < rhs.minor
            : major < rhs.major
            ;
    }

    bool operator <= (const VersionInfo& rhs) const
    {
        return *this < rhs || *this == rhs;
    }

    void swapWith (VersionInfo& other)
    {
        Message::swapWith(other);

        uplink_swap(major, other.major);
        uplink_swap(minor, other.minor);
    }

    uint8 major;
    uint8 minor;
};

struct VersionRange
{
    VersionRange ()
    : minimum()
    , maximum()
    {

    }

    VersionRange (
        const uint8 minimum_major, const uint8 minimum_minor,
        const uint8 maximum_major, const uint8 maximum_minor
    )
    : minimum(minimum_major, minimum_minor)
    , maximum(maximum_major, maximum_minor)
    {

    }

    bool isValid () const
    {
        return minimum.isValid() && maximum.isValid();
    }

    enum Compatibility
    {
        VersionIsOlder,
        VersionIsCompatible,
        VersionIsNewer
    };

    Compatibility compatibilityWith (const VersionInfo& versionInfo) const
    {
        if (versionInfo < minimum)
            return VersionIsOlder;
        else if (maximum < versionInfo)
            return VersionIsNewer;
        else
            return VersionIsCompatible;
    }

    VersionInfo minimum;
    VersionInfo maximum;
};

//------------------------------------------------------------------------------

struct ServiceEntry
{
    //FIXME: This should be removed. Passing an improperly initialized ServiceEntry around will cause bad access errors
    ServiceEntry ()
    {}

    ServiceEntry (const String& serviceName, uint16 port, const VersionRange& versionRange)
    : serviceName(serviceName)
    , port(port)
    , versionRange(versionRange)
    {
    }

    String         serviceName;
    uint16         port;
    VersionRange   versionRange;
};

//------------------------------------------------------------------------------

struct Service
{
    //FIXME: This should be removed. Passing an improperly initialized Services around will cause bad access errors
    Service ()
    {}

    Service (const ServiceEntry &entry, const NetworkAddress& address)
    : entry(entry)
    , address(address)
    {
    }

    VersionRange::Compatibility compatibilityWith (const VersionInfo& versionInfo) const
    {
        return entry.versionRange.compatibilityWith(versionInfo);
    }

    VersionRange::Compatibility compatibilityWithClientVersion () const;

    ServiceEntry   entry;
    NetworkAddress address;
};

//------------------------------------------------------------------------------

inline const VersionRange&
serverVersionRange ()
{
    static const VersionRange instance = VersionRange(
        UPLINK_SERVER_MINIMUM_VERSION_MAJOR,
        UPLINK_SERVER_MINIMUM_VERSION_MINOR,
        UPLINK_SERVER_MAXIMUM_VERSION_MAJOR,
        UPLINK_SERVER_MAXIMUM_VERSION_MINOR
    );

    return instance;
}

inline const VersionInfo&
clientVersion ()
{
    static const VersionInfo instance = VersionInfo(
        UPLINK_CLIENT_VERSION_MAJOR,
        UPLINK_CLIENT_VERSION_MINOR
    );

    return instance;
}

inline
VersionRange::Compatibility
Service::compatibilityWithClientVersion () const
{
    return compatibilityWith(clientVersion());
}

}

# include "./services.hpp"
