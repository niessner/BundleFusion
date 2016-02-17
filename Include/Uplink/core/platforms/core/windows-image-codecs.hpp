//
//  ImageCodecs_GdiPlus.cpp
//  Graphics
//
//  Created by Nicolas Tisserand on 10/31/14.
//  Copyright (c) 2014 Occipital. All rights reserved.
//

// WARNING: This source file is compiled on Mac OS X, in a cross-platform C++ project (ie: Skanect).
// Please keep the platform-specific dependencies in control and refrain from using any Objective-C.

// FIXME: There ARE memory leaks in these codec functions.

# pragma once

#if !_WIN32
#   error "This file can only be compiled on Windows."
#endif

//------------------------------------------------------------------------------

#ifndef _SCL_SECURE_NO_WARNINGS
#	define _SCL_SECURE_NO_WARNINGS 1
#endif

//------------------------------------------------------------------------------

# include "windows-image-codecs.h"
# include <Unknwn.h>
# include <algorithm>
namespace Gdiplus
{
    using std::min;
    using std::max;
}
# define GDIPVER 0x110 // This won't work on Windows XP.
# include <gdiplus.h>
# include <Gdiplusimaging.h>
# include <Shlwapi.h>
# include <map>
# include <vector>
# include <cassert>
# include <algorithm>

//------------------------------------------------------------------------------

//inline bool
//operator < (const GUID& lhs, const GUID& rhs)
//{
//    if (lhs.Data1 < rhs.Data1) return true; else if (rhs.Data1 < lhs.Data1) return false;
//    if (lhs.Data2 < rhs.Data2) return true; else if (rhs.Data2 < lhs.Data2) return false;
//    if (lhs.Data3 < rhs.Data3) return true; else if (rhs.Data3 < lhs.Data3) return false;
//
//	return std::memcmp(lhs.Data4, rhs.Data4, 8) < 0;
//}

//------------------------------------------------------------------------------

namespace uplink {

inline float
GdiPlusJpegQualityFromNormalizedLibJpegQuality (float normalizedLibJpegQuality)
{
    // The GDI+ JPEG compression quality behavior follows libjpeg's one very closely, so we just consider them equivalent, here.
    // The equivalence is not perfect, though, as GDI+ encodes slightly more compact JPEG payloads at higher quality values.
    // FIXME: Make up for these small discrepancies.
    // More info: Dropbox/CV Team/JPEG Compression Test/
 
    return normalizedLibJpegQuality;
}

struct GdiPlus
{
private:
    typedef std::vector<Gdiplus::ImageCodecInfo>    CodecsList;

private:
    struct Codec
    {
        Codec (
            const Gdiplus::ImageCodecInfo& encoderInfo_,
            const Gdiplus::ImageCodecInfo& decoderInfo_
        )
            : encoderInfo(encoderInfo_)
            , decoderInfo(decoderInfo_)
        {

        }

        Gdiplus::ImageCodecInfo encoderInfo;
        Gdiplus::ImageCodecInfo decoderInfo;
    };

private:
    typedef Gdiplus::Status GetImageCodecsSize(UINT*, UINT*);
    typedef Gdiplus::Status GetImageCodecs(UINT, UINT, Gdiplus::ImageCodecInfo*);

private:
    static void populateCodecsInfo (CodecsList& codecsList, GetImageCodecsSize getImageCodecsSize, GetImageCodecs getImageCodecs)
    {
        UINT num  = 0;
        UINT size = 0;

        Gdiplus::Status status = Gdiplus::Ok;

        status = getImageCodecsSize(&num, &size);
		assert(Gdiplus::Ok == status);

        assert(sizeof(Gdiplus::ImageCodecInfo) * num == size);

        codecsList.resize(num);

        status = getImageCodecs(num, size, codecsList.data());
        assert(Gdiplus::Ok == status);
	}

private:
    GdiPlus ()
	{
		return; // FIXME: Ditch or fix codec enumeration.

        CodecsList encodersList;
        CodecsList decodersList;

        populateCodecsInfo(encodersList, Gdiplus::GetImageEncodersSize, Gdiplus::GetImageEncoders);
        populateCodecsInfo(decodersList, Gdiplus::GetImageDecodersSize, Gdiplus::GetImageDecoders);

        typedef std::map<GUID, Gdiplus::ImageCodecInfo*> CodecsDict;

        CodecsDict encodersDict;
        CodecsDict decodersDict;

        for (size_t n = 0; n < encodersList.size(); ++n)
        {
            Gdiplus::ImageCodecInfo* info = &encodersList[n];
            encodersDict[info->FormatID] = info;
        }

        for (size_t n = 0; n < decodersList.size(); ++n)
        {
            Gdiplus::ImageCodecInfo* info = &decodersList[n];
            decodersDict[info->FormatID] = info;
        }

        static const GUID guids [] =
        {
            // Defined in the same sequence as the ImageCodec enum definition.
            Gdiplus::ImageFormatJPEG,
            Gdiplus::ImageFormatPNG,
            Gdiplus::ImageFormatBMP,
        };

        for (size_t n = 0; n < sizeof_array(guids); ++n)
        {
            const GUID guid = guids[n];

            CodecsDict::const_iterator enc = encodersDict.find(guid);
            CodecsDict::const_iterator dec = decodersDict.find(guid);

            assert(enc != encodersDict.end());
            assert(dec != decodersDict.end());

            codecs.push_back(Codec(*enc->second, *enc->second));
        }
	}

public:
    static GdiPlus& getInstance ()
    {
        static GdiPlus that;

        return that;
    }

    const Codec& getCodecInfo (graphics_ImageCodec imageCodec)
    {
        assert(
            graphics_ImageCodec_PNG  == imageCodec
         || graphics_ImageCodec_JPEG == imageCodec
         || graphics_ImageCodec_BMP  == imageCodec
        );

        return codecs[imageCodec];
    }

private:
    struct Handle
    {
        Handle ()
        {
            Gdiplus::GdiplusStartupInput startup;
            Gdiplus::GdiplusStartup(&token, &startup, NULL);
        }

        ~Handle ()
        {
            Gdiplus::GdiplusShutdown(token);
        }

        ULONG_PTR           token;
    };

public:
    static bool getEncoderClsid (const WCHAR* format, CLSID& clsid)
    {
        using namespace Gdiplus;

        UINT  num = 0;  // Number of image encoders.
        UINT  size = 0; // Size of the image encoder array in bytes.

        ImageCodecInfo* codecsInfo = NULL;

        GetImageEncodersSize(&num, &size);
        if(size == 0)
            return false;

        codecsInfo = (ImageCodecInfo*)(malloc(size));
        if(NULL == codecsInfo)
            return false;

        GetImageEncoders(num, size, codecsInfo);

        for(UINT j = 0; j < num; ++j)
        {
            if(0 == wcscmp(codecsInfo[j].MimeType, format))
            {
                clsid = codecsInfo[j].Clsid;

                // FIXME: Cache clsids.

                free(codecsInfo);
                return true;
            }
        }

        free(codecsInfo);
        return false;
    }

    static const WCHAR* getEncoderMimeType (graphics_ImageCodec imageCodec)
    {
        switch (imageCodec)
        {
            case graphics_ImageCodec_JPEG: return L"image/jpeg";
            case graphics_ImageCodec_PNG:  return L"image/png";
            case graphics_ImageCodec_BMP:  return L"image/bmp";

			default: return L"";
        }
    }

    static bool getEncoderClsid (graphics_ImageCodec imageCodec, CLSID& clsid)
    {
        return getEncoderClsid(getEncoderMimeType(imageCodec), clsid);
    }

	static Gdiplus::ColorPalette* defaultPalette ()
	{
		static Gdiplus::ColorPalette ret;

		return &ret;
	}

	static Gdiplus::ColorPalette* grayPalette ()
	{
		static bool initialized = false;

		// TRICK: Allocate the returned palette instance from a static buffer. The total buffer size includes additional space for 255 extra palette entries.
		static uint8_t paletteBytes [sizeof(Gdiplus::ColorPalette) + 255 * sizeof(Gdiplus::ARGB)];
		Gdiplus::ColorPalette* ret = reinterpret_cast<Gdiplus::ColorPalette*>(paletteBytes);

		if (!initialized)
		{
			ret->Flags = Gdiplus::PaletteFlagsGrayScale;
			ret->Count = 256;
			for(int n = 0; n < 256; ++n)
				ret->Entries[n] = Gdiplus::Color::MakeARGB(255, n, n, n);

			initialized = true;
		}

		return ret;
	}

private:
	typedef std::vector<Codec> Codecs;

private:
    Handle handle;
    Codecs codecs;
};


//------------------------------------------------------------------------------

inline bool
encode_image (
    graphics_ImageCodec imageCodec,
    const uint8_t* inputBuffer,
    size_t         inputSize,
    graphics_PixelFormat inputFormat,
    size_t         inputWidth,
    size_t         inputHeight,
    MemoryBlock&   outputMemoryBlock,
    float          outputQuality
)
{
    assert(0 != inputBuffer);
    assert(0 < inputSize);
    assert(
        graphics_PixelFormat_Gray == inputFormat
     || graphics_PixelFormat_RGB  == inputFormat
     || graphics_PixelFormat_RGBA == inputFormat
    );
    assert(0 < inputWidth);
    assert(0 < inputHeight);
    assert(0. < outputQuality && outputQuality < 1.);

    outputQuality = GdiPlusJpegQualityFromNormalizedLibJpegQuality(outputQuality);

    // Lazy-initialize GDI+.
    // FIXME: Can this fail? Check status if so.
    GdiPlus::getInstance();

    Gdiplus::Status status = Gdiplus::Ok;
    HRESULT result = S_OK;

	uint8_t* temporaryBuffer = 0;
	const uint8_t* actualBuffer = 0;

	const size_t numPixels = inputWidth * inputHeight;

    int rowStride = 0;
	Gdiplus::PixelFormat bitmapFormat = PixelFormatUndefined;
    switch (inputFormat)
    {
        case graphics_PixelFormat_Gray:
		{
			bitmapFormat = PixelFormat24bppRGB;
            rowStride = int(inputWidth) * 3;
			temporaryBuffer = new uint8_t [3 * numPixels];
			for (size_t n = 0; n < numPixels; ++n)
			{
				const uint8_t lum = inputBuffer[n];

				temporaryBuffer[n * 3    ] = lum;
				temporaryBuffer[n * 3 + 1] = lum;
				temporaryBuffer[n * 3 + 2] = lum;
			}
			actualBuffer = temporaryBuffer;
            break;
		}

        case graphics_PixelFormat_RGB:
		{
            // Even though the enumeration name makes you think otherwise, GDI+'s PixelFormat24bppRGB is BGR (RGB, little endian).
			bitmapFormat = PixelFormat24bppRGB;
            rowStride = int(inputWidth) * 3;
            temporaryBuffer = new uint8_t[3 * numPixels];
            for (size_t n = 0; n < numPixels; ++n)
            {
                const uint8_t* rgb = inputBuffer + n * 3;

                temporaryBuffer[n * 3    ] = rgb[2];
                temporaryBuffer[n * 3 + 1] = rgb[1];
                temporaryBuffer[n * 3 + 2] = rgb[0];
            }
            actualBuffer = temporaryBuffer;
            break;
		}

        case graphics_PixelFormat_RGBA:
		{
			bitmapFormat = PixelFormat32bppARGB;
            rowStride = int(inputWidth) * 4;
			temporaryBuffer = new uint8_t [4 * numPixels];
			for (size_t n = 0; n < numPixels; ++n)
			{
                // Even though the enumeration name makes you think otherwise, GDI+'s PixelFormat32bppARGB is BGRA (ARGB, little endian).
				const uint8_t* rgba = inputBuffer + n * 4;

				temporaryBuffer[n * 4    ] = rgba[2];
                temporaryBuffer[n * 4 + 1] = rgba[1];
                temporaryBuffer[n * 4 + 2] = rgba[0];
                temporaryBuffer[n * 4 + 3] = rgba[3];
			}
			actualBuffer = temporaryBuffer;
            break;
		}
    }

    // Construct the encoder input bitmap on the stack.
    Gdiplus::Bitmap bitmap (
        INT(inputWidth),
        INT(inputHeight),
        INT(rowStride),
        bitmapFormat,
        const_cast<BYTE*>(actualBuffer)
    );

	status = bitmap.GetLastStatus();
    assert(Gdiplus::Ok == status);

    // Create the encoder output stream.
	IStream* stream = SHCreateMemStream(NULL, 0);            
    assert(0 != stream);

	CLSID clsid;
    return_false_unless(GdiPlus::getEncoderClsid(imageCodec, clsid));

    // Initialize the encoder parameters.
    Gdiplus::EncoderParameters parameters;
    parameters.Count = 1;

    // FIXME: Normalize output quality with other implementations.
    ULONG qualityParameter = ULONG(outputQuality * 100.f);
    parameters.Parameter[0].Guid = Gdiplus::EncoderQuality;
    parameters.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
    parameters.Parameter[0].NumberOfValues = 1;
    parameters.Parameter[0].Value = &qualityParameter;

    // Encode the input bitmap into the output stream.
    status = bitmap.Save(stream, &clsid, &parameters);
    return_false_unless(Gdiplus::Ok == status);

    // Seek the output stream cursor back to its first byte.
    LARGE_INTEGER position = { 0 };
    result = stream->Seek(position, STREAM_SEEK_SET, NULL);
    return_false_unless(S_OK == result);

    // Retrieve the encoded stream size.
    ULARGE_INTEGER streamSize = { 0 };
    result = IStream_Size(stream, &streamSize);
    return_false_unless(S_OK == result);

    // Set and check the output buffer size.
    size_t outputSize = size_t(streamSize.QuadPart);
    static size_t maxOutputSize = 1024 * 1024 * 64; // FIXME: This is arbitrary.
    assert(outputSize < maxOutputSize);

    outputMemoryBlock.Resize(outputSize);

    // Allocate the output buffer.
    uint8_t* outputBuffer = outputMemoryBlock.Data;

    // Copy the encoded bytes to the output buffer.
    result = IStream_Read(stream, outputBuffer, ULONG(outputSize));

	if (0 != temporaryBuffer)
		delete [] temporaryBuffer;

    IUnknown_AtomicRelease((void**)&stream);

    return S_OK == result;
}

//------------------------------------------------------------------------------

inline bool
decode_image (
    graphics_ImageCodec imageCodec,
    const uint8_t* inputBuffer,
    size_t         inputSize,
    graphics_PixelFormat outputFormat,
    size_t&        outputWidth,
    size_t&        outputHeight,
    MemoryBlock&   outputMemoryBlock
)
{
    assert(0 != inputBuffer);
    assert(0 != inputSize);
    assert(
        graphics_PixelFormat_Gray == outputFormat
     || graphics_PixelFormat_RGB  == outputFormat
     || graphics_PixelFormat_RGBA == outputFormat
    );
    assert(0 == outputWidth); // Just in case the caller made false assumptions about the calling semantics of this function.
    assert(0 == outputHeight); // Just in case the caller made false assumptions about the calling semantics of this function.

    // Lazy-initialize GDI+.
    // FIXME: Can this fail? Check status if so.
    GdiPlus::getInstance();

    Gdiplus::Status status = Gdiplus::Ok;

    // Create the decoder input stream.
    IStream* stream = SHCreateMemStream(inputBuffer, UINT(inputSize));
    assert(0 != stream);

    // Construct the encoder input bitmap on the stack.
    Gdiplus::Bitmap bitmap(stream, FALSE);
    status = bitmap.GetLastStatus();

    IUnknown_AtomicRelease((void**)&stream);

    return_false_unless(Gdiplus::Ok == status);

    Gdiplus::Rect rect;
    rect.X = 0;
    rect.Y = 0;
    rect.Width  = bitmap.GetWidth();
    rect.Height = bitmap.GetHeight();

    Gdiplus::BitmapData bitmapData;

	status = bitmap.LockBits(
        &rect,
        Gdiplus::ImageLockModeRead,
        PixelFormat32bppARGB,
        &bitmapData
    );

	// Set output image size.
    outputWidth  = bitmapData.Width;
    outputHeight = bitmapData.Height;

    const size_t numPixels = outputWidth * outputHeight;

    assert(bitmapData.Stride * bitmapData.Height == 4 * numPixels);

    return_false_unless(Gdiplus::Ok == status);

    int bytesPerPixel = 0;
    switch (outputFormat)
    {
        case graphics_PixelFormat_Gray:
            bytesPerPixel = 1;
            break;

        case graphics_PixelFormat_RGB:
            bytesPerPixel = 3;
            break;

        case graphics_PixelFormat_RGBA:
            bytesPerPixel = 4;
            break;
    }

    outputMemoryBlock.Resize(numPixels * bytesPerPixel);

    // Even though the enumeration name makes you think otherwise, GDI+'s PixelFormat32bppARGB is BGRA (ARGB, little endian).
    uint8_t* bgraBuffer = reinterpret_cast<uint8_t*>(bitmapData.Scan0);

    uint8_t* outputBuffer = outputMemoryBlock.Data;

    switch (outputFormat)
    {
        case graphics_PixelFormat_Gray:
        {
            // In-place convert from ARGB to 8-bit grayscale, using the Rec. 601 luma formula and 8-bit truncation.
            // FIXME: There might be a more accurate way.
            for (size_t n = 0; n < numPixels; ++n)
            {
				const uint8_t* bgra = bgraBuffer + n * 4;

                outputBuffer[n] = uint8_t(
                    0.299 * bgra[2] +
                    0.587 * bgra[1] +
                    0.114 * bgra[0]
                );
            }
        }
        break;

        case graphics_PixelFormat_RGB:
        {
            // In-place convert from ARGB to RGB.
            // FIXME: There might be a faster way.
            for (size_t n = 0; n < numPixels; ++n)
            {

                const uint8_t* bgra = bgraBuffer + n * 4;

                outputBuffer[n * 3    ] = bgra[2];
                outputBuffer[n * 3 + 1] = bgra[1];
                outputBuffer[n * 3 + 2] = bgra[0];
            }
        }
        break;

        case graphics_PixelFormat_RGBA:
        {
            // In-place convert from ARGB to RGBA.
            // FIXME: There might be a faster way.
            for (size_t n = 0; n < numPixels; ++n)
            {
                const uint8_t* bgra =  bgraBuffer + n * 4;

                outputBuffer[n * 4    ] = bgra[2];
                outputBuffer[n * 4 + 1] = bgra[1];
                outputBuffer[n * 4 + 2] = bgra[0];
                outputBuffer[n * 4 + 3] = bgra[3];
            }
        }
        break;
    }

	status = bitmap.UnlockBits(&bitmapData);

    return Gdiplus::Ok == status;
}

//------------------------------------------------------------------------------

} // uplink namespace
