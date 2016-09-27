//
//  motion/events.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./motion.h"
# include "./core/macros.h"
# include "./core/serializers.h"

namespace uplink {

//------------------------------------------------------------------------------

# define MEMBERS() \
         MEMBER(x) \
         MEMBER(y) \
         MEMBER(z) \
         MEMBER(w) \
         MEMBER(m11) \
         MEMBER(m12) \
         MEMBER(m13) \
         MEMBER(m21) \
         MEMBER(m22) \
         MEMBER(m23) \
         MEMBER(m31) \
         MEMBER(m32) \
         MEMBER(m33)

inline void
Attitude::swapWith (Attitude& other)
{
# define MEMBER(Name) uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
Attitude::serializeWith (Serializer& s)
{
# define MEMBER(Name) report_false_unless("Cannot archive Attitude::" #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef  MEMBERS

//------------------------------------------------------------------------------

# define MEMBERS() \
         MEMBER(x) \
         MEMBER(y) \
         MEMBER(z)

inline void
RotationRate::swapWith (RotationRate& other)
{
# define MEMBER(Name) uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
RotationRate::serializeWith (Serializer& s)
{
# define MEMBER(Name) report_false_unless("Cannot archive RotationRate::" #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef  MEMBERS

//------------------------------------------------------------------------------

# define MEMBERS() \
         MEMBER(x) \
         MEMBER(y) \
         MEMBER(z)

inline void
Acceleration::swapWith (Acceleration& other)
{
# define MEMBER(Name) uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
Acceleration::serializeWith (Serializer& s)
{
# define MEMBER(Name) report_false_unless("Cannot archive Acceleration::" #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef  MEMBERS

//------------------------------------------------------------------------------

# define MEMBERS() \
         MEMBER(x) \
         MEMBER(y) \
         MEMBER(z) \
         MEMBER(accuracy)

inline void
MagneticField::swapWith (MagneticField& other)
{
# define MEMBER(Name) uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
MagneticField::serializeWith (Serializer& s)
{
# define MEMBER(Name) report_false_unless("cannot archive MagneticField::" #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef  MEMBERS

//------------------------------------------------------------------------------

# define MEMBERS()                \
         MEMBER(attitude)         \
         MEMBER(rotationRate)     \
         MEMBER(gravity)          \
         MEMBER(userAcceleration) \
         MEMBER(magneticField)

inline void
DeviceMotion::swapWith (DeviceMotion& other)
{
# define MEMBER(Name) uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
DeviceMotion::serializeWith (Serializer& s)
{
# define MEMBER(Name) report_false_unless("cannot archive DeviceMotion::" #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef  MEMBERS

//------------------------------------------------------------------------------

template <> inline MessageKind UnaryEvent<RotationRate>::kind () const { return MessageKind_GyroscopeEvent; }
template <> inline MessageKind UnaryEvent<Acceleration>::kind () const { return MessageKind_AccelerometerEvent; }
template <> inline MessageKind UnaryEvent<DeviceMotion>::kind () const { return MessageKind_DeviceMotionEvent; }

template <> inline Message* UnaryEvent<RotationRate>::clone () const { return new UnaryEvent<RotationRate>(*this); }
template <> inline Message* UnaryEvent<Acceleration>::clone () const { return new UnaryEvent<Acceleration>(*this); }
template <> inline Message* UnaryEvent<DeviceMotion>::clone () const { return new UnaryEvent<DeviceMotion>(*this); }


//------------------------------------------------------------------------------

inline bool
Event::serializeWith (Serializer& s)
{
    report_false_unless("cannot archive imu event timestamp", s.put(timestamp));

    return true;
}

//------------------------------------------------------------------------------

# define UNARY_EVENT_METHOD(...) \
template < class Type > \
inline __VA_ARGS__ \
UnaryEvent<Type>::

UNARY_EVENT_METHOD(void)
swapWith (UnaryEvent& other)
{
    Event::swapWith(other);
    
    uplink_swap(value, other.value);
}

UNARY_EVENT_METHOD(bool)
serializeWith (Serializer& s)
{
    return_false_unless(Event::serializeWith(s));

    report_false_unless("cannot archive imu event value", s.put(value));

    return true;
}
# undef UNARY_EVENT_METHOD

//------------------------------------------------------------------------------

}
