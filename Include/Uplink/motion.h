//
//  motion/events.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/enum.h"
# include "./core/serializers.h"

namespace uplink {

//------------------------------------------------------------------------------

// The following C++ types are modeled after the Objective-C types defined in Apple's CoreMotion framework.
// See: http://developer.apple.com/library/ios/documentation/CoreMotion/Reference/CoreMotion_Reference

//------------------------------------------------------------------------------

struct Attitude : Serializable
{
    // Quaternion.
    double x;
    double y;
    double z;
    double w;

    // Matrix
    double m11, m12, m13;
    double m21, m22, m23;
    double m31, m32, m33;

    void swapWith (Attitude& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

struct RotationRate : Serializable
{
    double x;
    double y;
    double z;

    void swapWith (RotationRate& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

struct Acceleration : Serializable
{
    double x;
    double y;
    double z;

    void swapWith (Acceleration& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

struct MagneticField : Serializable
{
    UPLINK_ENUM_BEGIN(Accuracy)
        Accuracy_Uncalibrated,
        Accuracy_Low,
        Accuracy_Medium,
        Accuracy_High,
    UPLINK_ENUM_END(Accuracy)

    double x;
    double y;
    double z;

    Accuracy accuracy;

    void swapWith (MagneticField& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

struct DeviceMotion : Serializable
{
    Attitude      attitude;
    RotationRate  rotationRate;
    Acceleration  gravity;
    Acceleration  userAcceleration;
    MagneticField magneticField;

    void swapWith (DeviceMotion& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

struct Event : Message
{
    Event ()
    {}

    Event (double timestamp)
    : timestamp(timestamp)
    {
    }

    void swapWith (Event& other)
    {
        Message::swapWith(other);
        uplink_swap(timestamp, other.timestamp);
    }

    virtual bool serializeWith (Serializer& serializer);

    double timestamp;
};

//------------------------------------------------------------------------------

template < class Type >
struct UnaryEvent : Event
{
    virtual MessageKind kind () const;

    virtual Message* clone () const;

    UnaryEvent ()
    {}

    UnaryEvent (const UnaryEvent& copy)
        : Event(copy.timestamp)
        , value(copy.value)
    {}

    UnaryEvent (double timestamp, const Type& value)
        : Event(timestamp)
        , value(value)
    {}

    Type value;

    void swapWith (UnaryEvent& other);

    virtual bool serializeWith (Serializer& serializer);
};

//------------------------------------------------------------------------------

typedef UnaryEvent<RotationRate>     GyroscopeEvent;
typedef UnaryEvent<Acceleration> AccelerometerEvent;
typedef UnaryEvent<DeviceMotion>  DeviceMotionEvent;

//------------------------------------------------------------------------------

}

# include "./motion.hpp"
