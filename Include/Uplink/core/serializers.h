//
//  binary/archives.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./types.h"
# include "./streams.h"

namespace uplink {

//------------------------------------------------------------------------------

#define UPLINK_SERIALIZER_ATOMS()          \
    UPLINK_SERIALIZER_ATOM(Uint8 , uint8 ) \
    UPLINK_SERIALIZER_ATOM(Uint16, uint16) \
    UPLINK_SERIALIZER_ATOM(Uint32, uint32) \
    UPLINK_SERIALIZER_ATOM(Uint64, uint64) \
    UPLINK_SERIALIZER_ATOM( Int8 ,  int8 ) \
    UPLINK_SERIALIZER_ATOM( Int16,  int16) \
    UPLINK_SERIALIZER_ATOM( Int32,  int32) \
    UPLINK_SERIALIZER_ATOM( Int64,  int64) \
    UPLINK_SERIALIZER_ATOM(Bool  , bool  ) \
    UPLINK_SERIALIZER_ATOM(Float , float ) \
    UPLINK_SERIALIZER_ATOM(Double, double)

void store (uint8   val,       Byte*& bytes);
void store (uint16  val,       Byte*& bytes);
void store (uint32  val,       Byte*& bytes);
void store (uint64  val,       Byte*& bytes);

void store ( int8   val,       Byte*& bytes);
void store ( int16  val,       Byte*& bytes);
void store ( int32  val,       Byte*& bytes);
void store ( int64  val,       Byte*& bytes);

void store (bool    val,       Byte*& bytes);
void store (float   val,       Byte*& bytes);
void store (double  val,       Byte*& bytes);

void fetch (uint8 & val, const Byte*& bytes);
void fetch (uint16& val, const Byte*& bytes);
void fetch (uint32& val, const Byte*& bytes);
void fetch (uint64& val, const Byte*& bytes);

void fetch ( int8 & val, const Byte*& bytes);
void fetch ( int16& val, const Byte*& bytes);
void fetch ( int32& val, const Byte*& bytes);
void fetch ( int64& val, const Byte*& bytes);

void fetch (  bool& val, const Byte*& bytes);
void fetch ( float& val, const Byte*& bytes);
void fetch (double& val, const Byte*& bytes);

//------------------------------------------------------------------------------

struct Reader;
struct Writer;
struct Serializable;

//------------------------------------------------------------------------------

struct Serializer
{
    virtual ~Serializer ();

    virtual bool isReader () const = 0;
            bool isWriter () const { return !isReader(); }

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
    virtual bool put (Type& val) = 0;
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool putBytes  (Byte*   bytes , Size size) = 0;

    bool put (Serializable& serializable);

    Reader& asReader ();
    Writer& asWriter ();
};

//------------------------------------------------------------------------------

struct Reader : Serializer
{
    virtual bool isReader () const { return true; }

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
    virtual bool put  (Type& val) { return read(val); } \
    virtual bool read (Type& val) = 0;
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool  putBytes (Byte* bytes , Size size) { return readBytes(bytes, size); }
    virtual bool readBytes (Byte* bytes , Size size) = 0;

    bool read (Serializable& serializable);

    bool readMagic (CString magic);
    bool readString (String& string, uint32 maxLength = 0xffff);
};

//------------------------------------------------------------------------------

struct InputStreamReader : Reader
{
    InputStreamReader (InputStream& input);
    InputStreamReader (Buffer& buffer);

   ~InputStreamReader ();

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
            bool read (Type& val);
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool readBytes (Byte* bytes , Size size);

    bool readAll (Buffer& buffer);

    bool   owned;
    InputStream* input;
};

//------------------------------------------------------------------------------

struct Writer : Serializer
{
    virtual bool isReader () const { return false; }

# define UPLINK_SERIALIZER_ATOM(Name, Type)  \
    virtual bool put   (      Type& val) { return write(val); } \
    virtual bool write (const Type& val) = 0;
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool   putBytes (Byte* bytes , Size size) { return writeBytes(bytes, size); }
    virtual bool writeBytes (const Byte* bytes , Size size) = 0;

    bool write (const Serializable& serializable);

    bool writeMagic (CString magic);
    bool writeString (const String& string, uint32 maxLength = 0xffff);

private: // Prevent accidental bool coercions.
    template < typename T > bool write (T*);
};

//------------------------------------------------------------------------------

struct OutputStreamWriter : Writer
{
     OutputStreamWriter (OutputStream& output_);
     OutputStreamWriter (Buffer& buffer);
    ~OutputStreamWriter ();

# define UPLINK_SERIALIZER_ATOM(Name, Type)  \
    virtual bool write (const Type& val);
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool writeBytes (const Byte* bytes , Size size);

    bool    owned;
    OutputStream* output;
};

//------------------------------------------------------------------------------

struct SerializedSizeCalculator : Writer
{
    SerializedSizeCalculator ()
        : serializedSize(0)
    {
    }

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
    virtual bool write (const Type& val) { serializedSize += sizeof(val); return true; }
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

    virtual bool writeBytes (const Byte*, Size size)
    {
        serializedSize += size;

        return true;
    }

    size_t serializedSize;
};

//------------------------------------------------------------------------------

struct Serializable : Streamable
{
public:
     Serializable ();
    ~Serializable ();

public:
    
public:
    virtual bool serializeWith (Serializer& serializer) = 0;

public:
    size_t serializedSize () const;
    virtual bool  readFrom (InputStream & input );
    virtual bool writeTo   (OutputStream& output) const;
};

//------------------------------------------------------------------------------

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
    size_t serializedSize (const Type&);
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

inline size_t
serializedSize (const String& str)
{
    return sizeof(uint32) + str.length();
}

}

# include "./serializers.hpp"
