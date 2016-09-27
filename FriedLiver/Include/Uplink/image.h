//
//  image.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./blobs.h"
# include "./camera-calibration.h"
# include <functional>

# if __APPLE__
# include <CoreMedia/CMSampleBuffer.h>
# include <CoreVideo/CVPixelBuffer.h>
# endif

namespace uplink {

//------------------------------------------------------------------------------

UPLINK_ENUM_BEGIN(ImageFormat)

    ImageFormat_Empty,

    ImageFormat_Shifts,
    ImageFormat_CompressedShifts,

    ImageFormat_Gray, // FIXME: Implement

    ImageFormat_RGB,
    ImageFormat_YCbCr,
    ImageFormat_JPEG,
    ImageFormat_H264,

UPLINK_ENUM_END(ImageFormat)

inline bool
isValidImageFormat (ImageFormat imageFormat)
{
    return ImageFormat_Invalid < imageFormat && imageFormat < ImageFormat_HowMany;
}

//------------------------------------------------------------------------------

struct Image : Message
{
public:
    UPLINK_MESSAGE_CLASS(Image);

    enum { MaxNumPlanes = 2 };

public:
    Image ();
    Image (const Image& copy);

public:
    struct Shifts {};
    struct CompressedShifts {};

public:
    Image (uint16* shifts, size_t size, Shifts);
    Image (uint8*  shifts, size_t size, CompressedShifts);

public:
    ~Image ();

    Image& operator = (const Image& other);

    void clear ();

    void swapWith (Image& other);
   
    virtual bool serializeWith (Serializer& s);

public:
    bool isEmpty () const
    {
        return 0 == width * height;
    }

    bool isCompressed () const
    {
        return
               ImageFormat_CompressedShifts == format
            || ImageFormat_JPEG             == format
            || ImageFormat_H264             == format
            ;
    }

    bool isCompressedOrEmpty () const
    {
        return isEmpty() || isCompressed();
    }

# if __APPLE__
public:
    Image (CMSampleBufferRef sampleBuffer, const CameraInfo& cameraInfo_, size_t width = 0, size_t height = 0);
    Image (CVPixelBufferRef   pixelBuffer, const CameraInfo& cameraInfo_);

private:
    void initializeWithPixelBuffer_RGB   (CVPixelBufferRef pixelBuffer);
    void initializeWithPixelBuffer_YCbCr (CVPixelBufferRef pixelBuffer);
    void initializeWithBlockBuffer       (CMBlockBufferRef blockBuffer, ImageFormat format_, size_t width, size_t height);

public:
    CVPixelBufferRef pixelBuffer () const;
    CMBlockBufferRef blockBuffer () const;
# endif

public: // For now.
    struct Storage
    {
        Storage ();

        enum Type
        {
            Type_Invalid,
        
# if __APPLE__
            Type_BlockBuffer,
            Type_PixelBuffer,
# endif
        };

        union Handle
        {
            void*            invalid;

# if __APPLE__
            CMBlockBufferRef blockBuffer;
            CVPixelBufferRef pixelBuffer;
# endif
        };
    
        Type   type;
        Handle handle;
    };

public: // For now.
    struct Plane
    {
        void*  buffer;
        size_t sizeInBytes;
        size_t bytesPerRow;
    };

    ImageFormat            format;
    size_t                 width;
    size_t                 height;
    Plane                  planes[MaxNumPlanes];
    CameraInfo             cameraInfo;
    std::function<void ()> release;
    std::function<void ()> retain;
    Storage                storage;
    Blobs                  attachments;
};

//------------------------------------------------------------------------------

}

# include "./image.hpp"
