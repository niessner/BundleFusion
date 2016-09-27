//
//  image.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./image.h"
# include <utility>

# if __APPLE__
#   include <TargetConditionals.h>
#   if TARGET_OS_IPHONE
#       include <CoreImage/CoreImage.h>
#   else
#       include <QuartzCore/CoreImage.h>
#   endif
#   include <ImageIO/ImageIO.h>
# endif

namespace uplink {

namespace {

# if __APPLE__

// FIXME: Unused.
inline std::pair<size_t, size_t>
getCompressedImageBufferSize (uint8* buffer, size_t size)
{
    size_t width = 0;
    size_t height = 0;
    {
        CFDataRef imageData  = CFDataCreateWithBytesNoCopy(NULL, buffer, size, kCFAllocatorNull);

        CFStringRef       imageSourceCreateKeys[1];
        CFTypeRef         imageSourceCreateValues[1];
        CFDictionaryRef   imageSourceCreateOptions;
        
        imageSourceCreateKeys[0] = kCGImageSourceShouldCache;
        imageSourceCreateValues[0] = (CFTypeRef)kCFBooleanFalse;
        imageSourceCreateOptions = CFDictionaryCreate(
            kCFAllocatorDefault,
            reinterpret_cast<const void**>(imageSourceCreateKeys),
            reinterpret_cast<const void**>(imageSourceCreateValues),
            1,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks
        );

        CGImageSourceRef imageSource = CGImageSourceCreateWithData (imageData, imageSourceCreateOptions);

        if (0 == imageSource)
            return std::make_pair(width, height);

        CFDictionaryRef imageProperties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, imageSourceCreateOptions);

        if (0 == imageProperties)
        {
            CFRelease(imageSource);
            return std::make_pair(width, height);
        };
        
        CFNumberRef widthNumber = (CFNumberRef) CFDictionaryGetValue(imageProperties, kCGImagePropertyPixelWidth);
        CFNumberRef heightNumber = (CFNumberRef) CFDictionaryGetValue(imageProperties, kCGImagePropertyPixelHeight);
        
        int val = 0;
        if (NULL != widthNumber)
        {
            CFNumberGetValue(widthNumber, kCFNumberIntType, &val);
            width = val;
        }
        if (NULL != heightNumber)
        {
            CFNumberGetValue(heightNumber, kCFNumberIntType, &val);
            height = val;
        }
        
        CFRelease(imageSource);
        CFRelease(imageProperties);
    }
    
    return std::make_pair(width, height);
}

# endif

}

//------------------------------------------------------------------------------

inline
Image::Image ()
: format(ImageFormat_Empty)
, width(0)
, height(0)
{
    for (int n = 0; n < MaxNumPlanes; ++n)
    {
        planes[n].buffer = 0;
        planes[n].sizeInBytes = 0;
        planes[n].bytesPerRow = 0;
    }
}

inline
Image::Image (const Image& copy)
: Message(copy)
, format(copy.format)
, width(copy.width)
, height(copy.height)
, cameraInfo(copy.cameraInfo)
, release(copy.release)
, retain(copy.retain)
, storage(copy.storage)
, attachments(copy.attachments)
{
    // NOTE: Image copies are shallow.

    std::copy(copy.planes, copy.planes + MaxNumPlanes, planes);

    assert(copy.isEmpty() || copy.retain);

    if (retain)
        retain();
}

# if __APPLE__
inline
Image::Image (CMSampleBufferRef sampleBuffer, const CameraInfo& cameraInfo_, size_t width, size_t height)
: Image()
{
    cameraInfo = cameraInfo_;

    CMFormatDescriptionRef formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer);
    CMMediaType mediaType = CMFormatDescriptionGetMediaType(formatDescription);
    assert(kCMMediaType_Video == mediaType);
    FourCharCode mediaSubType = CMFormatDescriptionGetMediaSubType(formatDescription);
    
    if (kCMPixelFormat_24RGB == mediaSubType)
        initializeWithPixelBuffer_RGB(CMSampleBufferGetImageBuffer(sampleBuffer));
    else if (kCMVideoCodecType_422YpCbCr8 == mediaSubType || kCVPixelFormatType_420YpCbCr8BiPlanarFullRange == mediaSubType)
        initializeWithPixelBuffer_YCbCr(CMSampleBufferGetImageBuffer(sampleBuffer));
    else if (kCMVideoCodecType_JPEG == mediaSubType)
        initializeWithBlockBuffer(CMSampleBufferGetDataBuffer(sampleBuffer), ImageFormat_JPEG, width, height);
    else if (kCMVideoCodecType_H264 == mediaSubType)
        initializeWithBlockBuffer(CMSampleBufferGetDataBuffer(sampleBuffer), ImageFormat_H264, width, height);
    else
        assert(false); // Unsupported sample buffer format.
}

inline
Image::Image (CVPixelBufferRef pixelBuffer, const CameraInfo& cameraInfo_)
: Image()
{
    cameraInfo = cameraInfo_;

    OSType pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    
    if (kCMPixelFormat_24RGB == pixelFormat)
        initializeWithPixelBuffer_YCbCr(pixelBuffer);
    else if (kCMPixelFormat_422YpCbCr8 == pixelFormat || kCVPixelFormatType_420YpCbCr8BiPlanarFullRange == pixelFormat)
        initializeWithPixelBuffer_YCbCr(pixelBuffer);
    else
        assert(false); // Unsupported sample buffer format.
}

inline void
Image::initializeWithPixelBuffer_RGB (CVPixelBufferRef pixelBuffer)
{
    format = ImageFormat_RGB;

    assert(0 != pixelBuffer);

    assert(!CVPixelBufferIsPlanar(pixelBuffer));

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    planes[0].buffer = CVPixelBufferGetBaseAddress(pixelBuffer);

    planes[0].bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);

    width  = CVPixelBufferGetWidth(pixelBuffer);
    height = CVPixelBufferGetHeight(pixelBuffer);

    planes[0].sizeInBytes = planes[0].bytesPerRow * height;
   
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    CFRetain(pixelBuffer);

    release = [pixelBuffer] () { CFRelease(pixelBuffer); };
    retain  = [pixelBuffer] () { CFRetain(pixelBuffer);  };
    
    storage.type = Storage::Type_PixelBuffer;
    storage.handle.pixelBuffer = pixelBuffer;
}

inline void
Image::initializeWithPixelBuffer_YCbCr (CVPixelBufferRef pixelBuffer)
{
    assert(0 != pixelBuffer);

    assert(CVPixelBufferIsPlanar(pixelBuffer));
    assert(2 == CVPixelBufferGetPlaneCount(pixelBuffer));

    format = ImageFormat_YCbCr;

    width  = CVPixelBufferGetWidth(pixelBuffer);
    height = CVPixelBufferGetHeight(pixelBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    planes[0].buffer = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
    planes[1].buffer = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 1);

    planes[0].bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
    planes[1].bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 1);

    planes[0].sizeInBytes = planes[0].bytesPerRow * height;
    planes[1].sizeInBytes = planes[1].bytesPerRow * height;
   
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    CFRetain(pixelBuffer);

    release = [pixelBuffer] () { CFRelease(pixelBuffer); };
    retain  = [pixelBuffer] () { CFRetain(pixelBuffer);  };
    
    storage.type = Storage::Type_PixelBuffer;
    storage.handle.pixelBuffer = pixelBuffer;
}

inline void
Image::initializeWithBlockBuffer (CMBlockBufferRef blockBuffer, ImageFormat format_, size_t width_, size_t height_)
{
    assert(0 != blockBuffer); // Could not retrieve the block buffer.
    assert(format_ != ImageFormat_Invalid);
    assert(0 < width_ * height_); // You need to specify a valid image size, when providing compressed buffers.

    format = format_;

    width = width_;
    height = height_;

    char* buffer = 0;
    size_t size = 0;
    size_t sizeAtOffset = 0;

    OSStatus status = CMBlockBufferGetDataPointer(blockBuffer, 0, &sizeAtOffset, &size, &buffer);
    assert(kCMBlockBufferNoErr == status); // Could not retrieve the data pointer.
    assert(size == sizeAtOffset); // Not contiguous, this is unsupported.

    planes[0].buffer = buffer;
    planes[0].sizeInBytes = size;

    CFRetain(blockBuffer);

    release = [blockBuffer] () { CFRelease(blockBuffer); };
    retain  = [blockBuffer] () { CFRetain(blockBuffer);  };
    
    storage.type = Storage::Type_BlockBuffer;
    storage.handle.blockBuffer = blockBuffer;
}

inline
CVPixelBufferRef
Image::pixelBuffer () const
{
    assert(Storage::Type_PixelBuffer == storage.type);
    assert(0 != storage.handle.pixelBuffer);
    
    return storage.handle.pixelBuffer;
}

inline CMBlockBufferRef
Image::blockBuffer () const
{
    assert(Storage::Type_BlockBuffer == storage.type);
    assert(0 != storage.handle.blockBuffer);

    return storage.handle.blockBuffer;
}
# endif

inline
Image::~Image ()
{
    if (0 != release)
        release();
}

inline Image&
Image::operator = (const Image& other)
{
    Image that(other);

    this->swapWith(that);

    return *this;
}

inline void
Image::clear ()
{
    if (0 != release)
        release();

    width = 0;
    height = 0;
    format = ImageFormat_Empty;
    cameraInfo = CameraInfo();
    release =  std::function<void ()>();
    retain =  std::function<void ()>();
    storage = Storage();

//        compress = 0;
//        decompress = 0;
}

inline void
Image::swapWith (Image& other)
{
    Message::swapWith(other);
    
    std::swap(format, other.format);

    std::swap(width, other.width);
    std::swap(height, other.height);

    cameraInfo.swapWith(other.cameraInfo);

    for (int n = 0; n < MaxNumPlanes; ++n)
    {
        std::swap(planes[n].buffer       , other.planes[n].buffer);
        std::swap(planes[n].sizeInBytes  , other.planes[n].sizeInBytes);
        std::swap(planes[n].bytesPerRow  , other.planes[n].bytesPerRow);
    }

    std::swap(release    , other.release);
    std::swap(retain     , other.retain );
    std::swap(storage    , other.storage );

    attachments.swapWith(other.attachments);
}

inline bool
Image::serializeWith (Serializer& s)
{
    static const int maxBufferSize = 0x4000000;

    if (s.isReader())
        clear();

    report_false_unless("cannot archive image format"  , s.put(format));
    report_false_unless("invalid archived image format", isValidImageFormat(format));

    uint32 width_  = uint32(width);
    uint32 height_ = uint32(height);
    report_false_unless("cannot archive image width"   , s.put(width_));
    report_false_unless("cannot archive image height"  , s.put(height_));
    report_false_unless("bad archived image size", 4096 > width_ && 4096 > height_);

    report_false_unless("cannot archive image camera info", s.put(cameraInfo));

    report_false_unless("cannot archive image attachments", attachments.serializeWith(s));

    if (s.isReader())
    {
        Reader& r = s.asReader();
        
        width  = width_;
        height = height_;

        if (0 == width * height)
            return true; // No further data to read.

        uint32 bufferSize = 0;
        report_false_unless("cannot read image buffer size", r.read(bufferSize));
        assert(bufferSize < maxBufferSize);

        typedef std::unique_ptr<uint8, void(*)(uint8*)> BufferPtr;

        // Make sure we don't leak buffer memory on read failures.
        BufferPtr bufferPtr(new uint8 [bufferSize], [](uint8* buffer){ delete [] buffer; });

        report_false_unless("cannot read image buffer data", r.readBytes(bufferPtr.get(), bufferSize));

        planes[0].buffer      = bufferPtr.release();
        planes[0].sizeInBytes = bufferSize;
        release = [this] () { delete [] (uint8*)planes[0].buffer; };
        retain = std::function<void ()>();
    }
    else
    {
        Writer& w = s.asWriter();

        if (0 == width * height)
            return true; // No further data to write.

        uint8* buffer = (uint8*) planes[0].buffer;
        uint32 bufferSize = uint32(planes[0].sizeInBytes);
        assert(bufferSize < maxBufferSize);

        report_false_unless("cannot write image buffer size", w.write(bufferSize));
        report_false_unless("cannot write image buffer data", w.writeBytes(buffer, bufferSize));
    }

    return true;
}

//------------------------------------------------------------------------------

inline
Image::Storage::Storage ()
: type(Type_Invalid)
{
    handle.invalid = 0;
}

//------------------------------------------------------------------------------
    
}
