//
//  binary/chunks.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

// This file was intentionally written using some wide text lines.
// Please keep it this way.

# pragma once

# include "./bytes.h"
# include "./streams.h"
# include "./serializers.h"
# include <vector>

namespace uplink {

//------------------------------------------------------------------------------

struct Chunk
{
    // Fixed-sized data segment.

public:
             Chunk () {}
    virtual ~Chunk () {}

public:
    virtual Size chunkSize  ()                   const = 0;
    virtual void storeChunk (      Byte*& bytes) const = 0;
    virtual void fetchChunk (const Byte*& bytes)       = 0;

public:
    Byte* appendTo   (      Buffer& buffer) const;
    void   fetchFrom (const Buffer& buffer, Index at = 0);
    void   fetchFrom (const Byte*   bytes);
};

//------------------------------------------------------------------------------

class Fields : public Chunk
{
public:
            Fields ();
   virtual ~Fields ();

public: // Chunk
    virtual Size chunkSize  ()                  const;
    virtual void storeChunk (      Byte*& bytes) const;
    virtual void fetchChunk (const Byte*& bytes);

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
private: \
    struct Name##Atom : Chunk \
    { \
        Name##Atom (Type* instance) : instance(instance) { assert(instance != 0); } \
        \
        virtual Size chunkSize  ()                   const { return sizeof(Type);     } \
        virtual void storeChunk (      Byte*& bytes) const { store(*instance, bytes); } \
        virtual void fetchChunk (const Byte*& bytes)       { fetch(*instance, bytes); } \
    \
        Type* instance; \
    }; \
public: \
    void addChunk (Type* instance)  { addChunk(new Name##Atom(instance), true); }
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

public:
    void addChunk (Chunk* chunk, bool own = false)
    {
        chunks.push_back(chunk);
        
        size += chunks.back()->chunkSize();

        if (own)
            owned.push_back(chunk);
    }

private:
    typedef std::vector<Chunk*> Chunks;

private:
    Size   size;
    Chunks chunks;
    Chunks owned;
};

//------------------------------------------------------------------------------

struct Header : Chunk
{
public:
    Header (Size    magicSize) : magic() { magic.resize(magicSize, '\0'); }
    Header (CString magic)     : magic(magic) {}
    virtual ~Header () {}

public: // Chunk
    virtual Size chunkSize  ()                  const;
    virtual void storeChunk (      Byte*& bytes) const;
    virtual void fetchChunk (const Byte*& bytes);

public:
    String magic ;
    Fields fields;
};

//------------------------------------------------------------------------------

}

# include "./chunks.hpp"
