# pragma once

# include "message.h"

namespace uplink {

//------------------------------------------------------------------------------

struct Blob : Message
{
    UPLINK_MESSAGE_CLASS(Blob)

    Blob ()
    : tag(0)
    {
    }

    virtual bool serializeWith (Serializer &serializer)
    {
        if(serializer.isReader())
        {
            return_false_unless(serializer.asReader().read(tag));

            uint32 size;
            return_false_unless(serializer.asReader().read(size));

            // FIXME: Maximum size check.

            data.resize(size);
            serializer.asReader().readBytes(data.data(), size);
        }
        else
        {
            return_false_unless(serializer.asWriter().write(tag));
            size_t size = data.size();
            return_false_unless(serializer.asWriter().write(uint32(size)));
            return_false_unless(serializer.asWriter().writeBytes(data.data(), size));
        }

        return true;
    }

    void swapWith (Blob& other)
    {
        Message::swapWith(other);

        uplink_swap(tag , other.tag);
        uplink_swap(data, other.data);
    }

    uint8              tag;
    std::vector<uint8> data;
};


//------------------------------------------------------------------------------

// FIXME: Move this somewhere else.

struct Blobs : Serializable
{
    enum { MaxNumBlobs = 0x10000 };

    Blobs ()
    {
    }

    virtual bool serializeWith (Serializer &s)
    {
        if(s.isReader())
        {
            Reader& r = s.asReader();

            uint16 numBlobs;

            return_false_unless(r.read(numBlobs));

            blobs.resize(numBlobs);
        }
        else
        {
            Writer& w = s.asWriter();

            size_t numBlobs_ = blobs.size();

            assert (0 <= numBlobs_ && numBlobs_ < MaxNumBlobs);

            uint16 numBlobs = uint16(numBlobs_);

            return_false_unless(w.write(numBlobs));
        }

        for (int n = 0; n < blobs.size(); ++n)
            return_false_unless(blobs[n].serializeWith(s));

        return true;
    }

    void swapWith (Blobs& other)
    {
        uplink_swap(blobs, other.blobs);
    }

    std::vector<Blob> blobs;
};

//------------------------------------------------------------------------------

} // uplink namespace

# include "blobs.h"
