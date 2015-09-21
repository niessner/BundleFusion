//
//  binary/bytes.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./bytes.h"
# include <cassert>

namespace uplink {

//------------------------------------------------------------------------------

inline uint64
byteOrderSwapped (uint64 val)
{
    uint64 ret;

    setByte<0>(getByte<7>(val), ret);
    setByte<1>(getByte<6>(val), ret);
    setByte<2>(getByte<5>(val), ret);
    setByte<3>(getByte<4>(val), ret);
    setByte<4>(getByte<3>(val), ret);
    setByte<5>(getByte<2>(val), ret);
    setByte<6>(getByte<1>(val), ret);
    setByte<7>(getByte<0>(val), ret);

    return ret;
}

inline uint32
byteOrderSwapped (uint32 val)
{
    uint32 ret;

    setByte<0>(getByte<3>(val), ret);
    setByte<1>(getByte<2>(val), ret);
    setByte<2>(getByte<1>(val), ret);
    setByte<3>(getByte<0>(val), ret);

    return ret;
}

inline uint16
byteOrderSwapped (uint16 val)
{
    uint16 ret;

    setByte<0>(getByte<1>(val), ret);
    setByte<1>(getByte<0>(val), ret);

    return ret;
}

inline void
appendBuffer (Buffer& buffer, const Byte* bytes, Size size)
{
    buffer.insert(buffer.end(), bytes, bytes + size);
}

inline Byte*
growBuffer (Buffer& buffer, Size increase)
{
    assert(0 <= increase);
    Size at = buffer.size();
    buffer.resize(at + increase);
    assert(!buffer.empty()); // Or we cannot return any valid pointer.
    return &buffer[at];
}

inline Byte*
mutableBufferBytes (Buffer& buffer, Index offset)
{
    assert(size_t(offset) < buffer.size());

    return &buffer[offset];
}

inline const Byte*
bufferBytes (const Buffer& buffer, Index offset)
{
    return mutableBufferBytes(const_cast<Buffer&>(buffer), offset);
}

//------------------------------------------------------------------------------

}
