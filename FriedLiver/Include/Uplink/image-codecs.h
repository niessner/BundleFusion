//
//  image/image-codecs.h
//  Uplink
//
//  Copyright (c) 2013 Occipital, Inc. All rights reserved.
//

# pragma once

#include "./core/bitstream.h"
#include "./image.h"
#include "./core/memory.h"
#include "./core/shift2depth.h"
#include <functional>

namespace uplink {

//------------------------------------------------------------------------------

// -----------------------------------------------
// OCCIPITAL DEPTH FRAME COMPRESSION ALGORITHM 0.1
// -----------------------------------------------

// TYPICAL COMPRESSION RATIO ON 640x480 test image:  0.17

// DECODE:
// Step 0. Last value is initialized to 0.  Frame size is known in advance.
// Step 1. Proceed by decoding following bitstream until all pixels are decoded.

// 00 - Next value is same as last value.
// 11 - Next value is last value + 1.
// 10 - Next value is last value - 1.
// 010 - bbbbb - Next N values are same as last value.  (N encoded w/ 5 bits)
// 0111 - bbbbbbbbbbb - Next value is X.  (X encoded w/ 11 bits)
// 01101 - Next value is last value + 2.
// 01100 - Next value is last value - 2.

inline uint16 * decode (const uint8 * bitstream_data, unsigned int bitstream_length_bytes, int numelements, uint16* output)
{

    uint16_t lastVal = 0;
    uint16_t curVal = 0;

    bitstream_t bs;
    bs_init(&bs);
    bs_attach(&bs, const_cast<uint8_t*>(bitstream_data), bitstream_length_bytes);

    uint16_t * depthimage = 0 == output
        ? (uint16_t*)malloc(numelements * sizeof(uint16_t))
        : output
        ;

    uint16_t * depth_ptr = depthimage;

    while(numelements > 0)
    {
        uint8_t bit0 = bs_get(&bs, 1);
        uint8_t bit1 = bs_get(&bs, 1);

        if(bit0 == 0 && bit1 == 0) // 00
        {
            curVal = lastVal;

            *(depth_ptr++) = curVal; lastVal = curVal;

            numelements-=1;
        }
        else if(bit0 == 1) // 1 prefix
        {

            if(bit1 == 0) {
                curVal = lastVal - 1;
            }
            else {
                curVal = lastVal + 1;
            }

            *(depth_ptr++) = curVal; lastVal = curVal;

            numelements-=1;

        }
        else  // must be 01 prefix
        {
            uint8_t bit2 = bs_get(&bs, 1);

            if(bit2 == 0) // 010 --> multiple zeros!
            {
                uint16_t numZeros = bs_get(&bs, 5);

                numZeros += 5; // We never encode less than 5.

                for(int i = 0; i < numZeros; i++) {
                    *(depth_ptr++) = curVal;
                }

                numelements-=numZeros;
            }
            else
            {
                uint8_t bit3 = bs_get(&bs, 1);

                if(bit3 == 0) // 0110 -- DELTA!
                {
                    uint8_t delta_bit = bs_get(&bs, 1);

                    if(delta_bit == 0) {
                        curVal = lastVal - 2;
                    }
                    else {
                        curVal = lastVal + 2;
                    }

                    *(depth_ptr++) = curVal; lastVal = curVal;

                    numelements-=1;

                }
                else // 0111 -- RESET!
                {
                    uint16_t value = (bs_get(&bs, 3) << 8) | bs_get(&bs, 8); // 11 bits total.

                    curVal = value;

                    *(depth_ptr++) = curVal; lastVal = curVal;
                    numelements-=1;
                }

            }

        }

    }

    return depthimage;

}

//------------------------------------------------------------------------------

inline uint32_t encode(const uint16_t * data_in, int numelements,
                                  uint8_t* out_buffer, uint32_t out_buffer_size)
{
    int numZeros = 0;
    int lastVal = 0;
    
    bitstream_t bs;
    bs_init(&bs);
    bs_attach(&bs, out_buffer, out_buffer_size);

    // Loop over pixels.
    while (numelements > 0) {

        int curVal = *(data_in++);
        int delta = curVal - lastVal;

        if(delta == 0)
        {
            numZeros++;
        }
        else
        {
            if(numZeros > 0)
            {
                // MUST BURN ZEROS!
                while( numZeros > 0 )
                {
                    if(numZeros <= 4)
                    {
                        // Ternary is fastest way of deciding how many zeros to encode (2 * numZeros)
                        bs_put(&bs, 0x0000, numZeros == 1 ? 2 : numZeros == 2 ? 4 : numZeros == 3 ? 6 : 8);
                        numZeros = 0;
                    }
                    else
                    {
                        bs_put(&bs, 0x2, 3); // 010bbbbb

                        // We never encode less than 5 because in that case
                        //  we'll just use multiple 2-bit single zeros.
                        unsigned int numberToEncode = numZeros - 5;

                        // We're only using 5 bits, so we can't encode greater than 31.
                        if(numberToEncode > 31) numberToEncode = 31;

                        bs_put(&bs, numberToEncode, 5); // 0b 010

                        numZeros -= (numberToEncode+5);
                    }
                }

                // numZeros is now zero.
            }

            if(delta == 1 || delta == -1)
            {
                bs_put(&bs, delta == 1 ? 0x3 : 0x2, 2); // 0b 11
            }
            else if (delta >= -2 && delta <= 2)
            {
                bs_put(&bs, delta == 2 ? 0xD : 0xC, 5);
            }
            else // Reset == 1111 bbbbbbbbbbb
            {
                bs_put(&bs, 0x7, 4); // 0111
                bs_put(&bs, curVal >> 8, 3);
                bs_put(&bs, curVal , 8);
            }

        } // end else block of if (delta == 0)

        lastVal = curVal;

        numelements--;
    }

    // FINISH Up -- repeat zeros check.

    if(numZeros > 0)
    {
        // MUST BURN ZEROS!
        while(numZeros > 0)
        {
            if(numZeros <= 4)
            {
                // Ternary is fastest way of deciding how many zeros to encode (2 * numZeros)
                bs_put(&bs, 0x0000, numZeros == 1 ? 2 : numZeros == 2 ? 4 : numZeros == 3 ? 6 : 8);
                numZeros = 0;
            }
            else
            {
                bs_put(&bs, 0x2, 3); // 010bbbbb

                // We never encode less than 5 because in that case
                //  we'll just use multiple 2-bit single zeros.
                unsigned int numberToEncode = numZeros - 5;

                // We're only using 5 bits, so we can't encode greater than 31.
                if(numberToEncode > 31) numberToEncode = 31;

                bs_put(&bs, numberToEncode, 5); // 0b 010

                numZeros -= (numberToEncode+5);
            }
        }
    }

    // numZeros is now zero.

    // END FINISH UP


    bs_flush(&bs);
    return bs_bytes_used(&bs);

}

//------------------------------------------------------------------------------

inline bool
compress_image_Shifts_CompressedShifts (const Image& source, Image& target)
{
    assert(target.isEmpty());
    assert(!source.isEmpty());
    assert(ImageFormat_Shifts == source.format);

    const void*   sourceBuffer      = source.planes[0].buffer;
    const size_t  sourceSizeInBytes = source.planes[0].sizeInBytes;

    uint8* targetBuffer = new uint8[sourceSizeInBytes];

    const size_t compressedSize = encode(
        (uint16_t*) sourceBuffer,
        int(sourceSizeInBytes / 2),
        targetBuffer,
        uint32(sourceSizeInBytes)
    );
    
    if (compressedSize < 1)
    {
        uplink_log_error("OCC compression failed.");
        return false;
    }

    target.width  = source.width;
    target.height = source.height;
    target.format = ImageFormat_CompressedShifts;
    target.planes[0].buffer      = targetBuffer;
    target.planes[0].sizeInBytes = compressedSize;

    target.cameraInfo = source.cameraInfo;

    target.release = [targetBuffer] () { delete [] targetBuffer; };
    target.retain  = std::function<void ()>();

    return true;
}

inline bool
decompress_image_CompressedShifts_Shifts (const Image& source, Image& target)
{
    assert(target.isEmpty());
    assert(!source.isEmpty());
    assert(ImageFormat_CompressedShifts == source.format);
 
    const uint8* sourceBuffer = (uint8*) source.planes[0].buffer;
    const size_t sourceSizeInBytes = source.planes[0].sizeInBytes;
    const size_t numTargetElements = source.width * source.height;

    uint16_t* targetBuffer = decode(sourceBuffer, unsigned(sourceSizeInBytes), int(numTargetElements), 0);

    if (0 == targetBuffer)
        return false;

    target.width  = source.width;
    target.height = source.height;
    target.format = ImageFormat_Shifts;
    target.planes[0].buffer      = targetBuffer;
    target.planes[0].sizeInBytes = numTargetElements * 2;

    target.cameraInfo = source.cameraInfo;

    target.release = [targetBuffer] () { free(targetBuffer); };
    target.retain  = std::function<void ()>();

    return true;
}

struct ImageCodec
{
    std::function<bool (const Image&, Image&)> compress;
    std::function<bool (const Image&, Image&)> decompress;

    ImageFormat compressInputFormat;
    ImageFormat compressOutputFormat;

    ImageFormat decompressInputFormat;
    ImageFormat decompressOutputFormat;

    bool canCompress (ImageFormat imageFormat) const
    {
        uplink_log_debug("ImageCodec::canCompress: compress: %d compressInputFormat: %s imageFormat: %s",
            bool(compress),
            uplink::toString(compressInputFormat).c_str(),
            uplink::toString(imageFormat).c_str()
        );

        return compress && compressInputFormat == imageFormat;
    }

    bool canDecompress (ImageFormat imageFormat) const
    {
        uplink_log_debug("ImageCodec::canDecompress: decompress: %d decompressInputFormat: %s imageFormat: %s",
            bool(decompress),
            uplink::toString(decompressInputFormat).c_str(),
            uplink::toString(imageFormat).c_str()
        );

        return decompress && decompressInputFormat == imageFormat;
    }
};

struct ImageCodecs
{
    ImageCodecs ()
    {
        compressedShifts.compress               =   compress_image_Shifts_CompressedShifts;
        compressedShifts.decompress             = decompress_image_CompressedShifts_Shifts;
        compressedShifts.compressInputFormat    = ImageFormat_Shifts;
        compressedShifts.compressOutputFormat   = ImageFormat_CompressedShifts;
        compressedShifts.decompressInputFormat  = ImageFormat_CompressedShifts;
        compressedShifts.decompressOutputFormat = ImageFormat_Shifts;

        h264.compressOutputFormat  = ImageFormat_H264;
        h264.decompressInputFormat = ImageFormat_H264;
        // The remainder of the H264 codec members will be specified elsewhere.

        jpeg.compressOutputFormat  = ImageFormat_JPEG;
        jpeg.decompressInputFormat = ImageFormat_JPEG;
        // The remainder of the JPEG codec members will be specified elsewhere.
    }

    ImageCodec byId [ImageCodecId_HowMany];

    ImageCodec&  compressedShifts = byId[ImageCodecId_CompressedShifts];
    ImageCodec&  jpeg             = byId[ImageCodecId_JPEG];
    ImageCodec&  h264             = byId[ImageCodecId_H264];

    bool canCompress (ImageFormat imageFormat) const
    {
        for (int n = 0; n < ImageCodecId_HowMany; ++n)
            if (byId[n].canCompress(imageFormat))
                return true;

        return false;
    }

    bool canDecompress (ImageFormat imageFormat) const
    {
        for (int n = 0; n < ImageCodecId_HowMany; ++n)
            if (byId[n].canDecompress(imageFormat))
                return true;

        return false;
    }
};

//------------------------------------------------------------------------------

template < typename FStream >
inline bool
utf8_open (FStream& fs, const char* path /* UTF-8 */,  std::ios_base::openmode mode)
{
#ifndef _WIN32
    // Just pass the UTF-8 path on Unices.
    fs.open(path, mode);
    return true;
#else
    // On Windows, first convert UTF-8 to wide char, then open the converted path.

    const int size = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);

    std::vector<wchar_t> buffer(size, 0);

    const int status = MultiByteToWideChar(CP_UTF8, 0, path, -1, buffer.data(), int(buffer.size()));

    if (0 == status)
        return false;

    fs.open(buffer.data(), mode);
#endif
}

//------------------------------------------------------------------------------

inline bool
read_image (
    graphics_ImageCodec      imageCodec,
    const char*     inputPath,
    graphics_PixelFormat     outputFormat,
    size_t&         outputWidth,
    size_t&         outputHeight,
    MemoryBlock&  outputMemoryBlock
)
{
    std::ifstream inputStream;

    utf8_open(inputStream, inputPath, std::ios::binary | std::ios::ate);

    if (!inputStream)
        return false;

    const size_t inputSize = inputStream.tellg();
    inputStream.seekg(0); // Rewind to the beginning, since the file was opened at the end position.

    if (0 == inputSize)
        return false; // Empty files are invalid.

    MemoryBlock input;
    input.Resize(inputSize);

    inputStream.read((char*)input.Data, inputSize);

    return decode_image(
        imageCodec,
        input.Data,
        input.Size,
        outputFormat,
        outputWidth,
        outputHeight,
        outputMemoryBlock
    );
}

//------------------------------------------------------------------------------

inline bool
write_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t* inputBuffer,
    size_t         inputSize,
    graphics_PixelFormat    inputFormat,
    size_t         inputWidth,
    size_t         inputHeight,
    const char*    outputPath,
    float          outputQuality
)
{
    std::ofstream outputStream;

    utf8_open(outputStream, outputPath, std::ios::binary);

    if (!outputStream)
        return false;

    MemoryBlock outMemoryBlock;

    const bool ok = encode_image(
        imageCodec,
        inputBuffer,
        inputSize,
        inputFormat,
        inputWidth,
        inputHeight,
        outMemoryBlock,
        outputQuality
    );

    if (!ok)
        return false;

    size_t   outputSize = outMemoryBlock.Size;
    uint8_t* outputBuffer = outMemoryBlock.Data;

    // write to file
    std::copy(
        outputBuffer,
        outputBuffer + outputSize,
        std::ostreambuf_iterator<char>(outputStream)
    );

    return ok;
}

//------------------------------------------------------------------------------

inline bool
compress_image_RGB_JPEG(const Image& source, Image& target)
{
    assert(target.isEmpty());
    assert(!source.isEmpty());
    assert(ImageFormat_RGB == source.format);

    uint8_t* outputBuffer = 0;
    size_t outputSize = 0;

    MemoryBlock outputMemoryBlock;

    const bool ret = encode_image(
        graphics_ImageCodec_JPEG,
        (const uint8*)source.planes[0].buffer,
        source.planes[0].sizeInBytes,
        graphics_PixelFormat_RGB,
        source.width,
        source.height,
        outputMemoryBlock,
        defaultQuality
    );

    if (!ret)
    {
        uplink_log_error("Could not compress JPEG image.");
        return false;
    }

    target.width = source.width;
    target.height = source.height;
    target.format = ImageFormat_JPEG;

    target.planes[0].buffer = outputMemoryBlock.Data;
    target.planes[0].sizeInBytes = outputMemoryBlock.Size;

    target.cameraInfo = source.cameraInfo;

    outputMemoryBlock.transferOwnership(target.release);
    target.retain = std::function<void()>();

    return true;
}

inline bool
decompress_image_JPEG_RGB(const Image& source, Image& target)
{
    assert(target.isEmpty());
    assert(!source.isEmpty());
    assert(ImageFormat_JPEG == source.format);

    const uint8_t* inputBuffer = (uint8*)source.planes[0].buffer;
    size_t inputSize = source.planes[0].sizeInBytes;

    size_t outputWidth = 0;
    size_t outputHeight = 0;

    MemoryBlock outputMemoryBlock;

    const bool ret = decode_image(
        graphics_ImageCodec_JPEG,
        inputBuffer,
        inputSize,
        graphics_PixelFormat_RGB,
        outputWidth,
        outputHeight,
        outputMemoryBlock
    );

    if (!ret)
    {
        uplink_log_error("Could not decompress JPEG image.");
        return false;
    }

    target.width = outputWidth;
    target.height = outputHeight;
    target.format = ImageFormat_RGB;

    target.planes[0].buffer = outputMemoryBlock.Data;
    target.planes[0].sizeInBytes = outputMemoryBlock.Size;

    target.cameraInfo = source.cameraInfo;

    outputMemoryBlock.transferOwnership(target.release);
    target.retain = std::function<void()>();

    return true;
}

//------------------------------------------------------------------------------

} // uplink namespace

# include "./image-codecs.hpp"
