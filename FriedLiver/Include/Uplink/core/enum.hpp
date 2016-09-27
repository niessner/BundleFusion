//
//  binary/enum.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./enum.h"

namespace uplink {

//------------------------------------------------------------------------------

# define ENUM_METHOD(...) \
template < typename Type, Type Invalid, Type HowMany > \
inline __VA_ARGS__ \
Enum<Type, Invalid, HowMany>::

ENUM_METHOD()
Enum ()
: value(Invalid)
{
}

ENUM_METHOD()
Enum (Type value)
: value(value)
{
}

ENUM_METHOD()
operator const Type () const
{
    return value;
}

ENUM_METHOD(Enum<Type, Invalid, HowMany>&)
operator = (const Type& rhs)
{
    value = rhs; return *this;
}

ENUM_METHOD(void)
swapWith (Enum& other)
{
    ::std::swap(value, other.value);
}

ENUM_METHOD(bool)
serializeWith (Serializer& serializer)
{
    Unsigned n = 0;

    if (serializer.isWriter())
    {
        return_false_unless(Invalid < value && value < HowMany);

        n = value;
    }

    return_false_unless(serializer.put(n));

    if (serializer.isReader())
    {
        return_false_unless(Invalid < Type(n) && Type(n) < HowMany);

        value = Type(n);
    }

    return true;
}

ENUM_METHOD(Size)
chunkSize () const
{
    return sizeof(Unsigned);
}

ENUM_METHOD(void)
storeChunk (Byte*& bytes) const
{
    Unsigned val = value;
    store(val, bytes);
}

ENUM_METHOD(void)
fetchChunk (const Byte*& bytes)
{
    Unsigned val;
    fetch(val, bytes);
    value = Type(val);
}

ENUM_METHOD(bool)
operator == (const Enum& rhs) const
{
    return value == rhs.value;
}

ENUM_METHOD(bool)
operator != (const Enum& rhs) const
{
    return value != rhs.value;
}

ENUM_METHOD(bool)
operator == (const Type& otherValue) const
{
    return value == otherValue;
}

ENUM_METHOD(bool)
operator != (const Type& otherValue) const
{
    return value != otherValue;
}

# undef ENUM_METHOD

//------------------------------------------------------------------------------

template < typename Type, Type Invalid, Type HowMany >
inline bool
operator == (const Type& value, const Enum<Type, Invalid, HowMany>& that)
{
    return that == value;
}

template < typename Type, Type Invalid, Type HowMany >
inline bool
operator != (const Type& value, const Enum<Type, Invalid, HowMany>& that)
{
    return that != value;
}

//------------------------------------------------------------------------------

}
