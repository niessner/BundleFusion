//
//  binary/chunks.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./chunks.h"

namespace uplink {

//------------------------------------------------------------------------------

inline Byte*
Chunk::appendTo (Buffer& buffer) const
{
    Byte* ret = growBuffer(buffer, chunkSize());
    Byte* tmp = ret;
    storeChunk(tmp);
    return ret;
}

inline void
Chunk::fetchFrom (const Buffer& buffer, Index at)
{
    assert(at + chunkSize() <= buffer.size());
    const Byte* bytes = &buffer[at];
    fetchChunk(bytes);
}

inline void
Chunk::fetchFrom (const Byte* bytes)
{
    fetchChunk(bytes);
}

//------------------------------------------------------------------------------

inline
Fields::Fields ()
    : size(0)
{

}

inline
Fields::~Fields ()
{
    for (int i = 0; i < owned.size(); ++i)
        delete owned[i];
}

inline Size
Fields::chunkSize () const
{
    return size;
}

inline void
Fields::storeChunk (Byte*& bytes) const
{
    for (int i = 0; i < chunks.size(); ++i)
        chunks[i]->storeChunk(bytes);
}

inline void
Fields::fetchChunk (const Byte*& bytes)
{
    for (int i = 0; i < chunks.size(); ++i)
        chunks[i]->fetchChunk(bytes);
}

//------------------------------------------------------------------------------

inline Size
Header::chunkSize () const
{
    return magic.size() + fields.chunkSize();
}

inline void
Header::storeChunk (Byte*& bytes) const
{
    std::memcpy(bytes, &magic[0], magic.size());
    bytes += magic.size();
    fields.storeChunk(bytes);
}

inline void
Header::fetchChunk (const Byte*& bytes)
{
    magic.assign(CString(bytes), magic.size());
    bytes += magic.size();
    fields.fetchChunk(bytes);
}

//------------------------------------------------------------------------------

}
