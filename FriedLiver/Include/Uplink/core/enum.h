//
//  binary/enum.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./chunks.h"
# include "./serializers.h"

namespace uplink {

//------------------------------------------------------------------------------

template < typename Type, Type Invalid, Type HowMany >
struct Enum : Serializable, Chunk
{
    typedef uint32 Unsigned; // FIXME: 4 bytes is too big.

    Enum ();
    Enum (Type value);

    operator const Type () const;

    Enum& operator = (const Type& rhs);

    void swapWith (Enum& other);

    virtual bool serializeWith (Serializer& serializer);

    virtual Size      chunkSize  () const;
    virtual void storeChunk (      Byte*& bytes) const;
    virtual void fetchChunk (const Byte*& bytes);

    bool operator ==  (const Enum& rhs) const;
    bool operator !=  (const Enum& rhs) const;

    bool operator != (const Type& otherValue) const;
    bool operator == (const Type& otherValue) const;

    String toString () const
    {
        return ::uplink::toString(int(value)); // For now. FIXME: Return enum names instead.
    }

    Type value;
};

//------------------------------------------------------------------------------

template < typename Type, Type Invalid, Type HowMany >
inline bool
operator == (const Type& value, const Enum<Type, Invalid, HowMany>& that);

template < typename Type, Type Invalid, Type HowMany >
inline bool
operator != (const Type& value, const Enum<Type, Invalid, HowMany>& that);

//------------------------------------------------------------------------------

#define UPLINK_ENUM_BEGIN(Name) \
enum Name##_Enum \
{ \
    Name##_Invalid = -1, \

#define UPLINK_ENUM_END(Name) \
    Name##_HowMany \
}; \
typedef Enum<Name##_Enum, Name##_##Invalid, Name##_##HowMany> Name;

//------------------------------------------------------------------------------

}

# include "./enum.hpp"
