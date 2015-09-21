//
//  ImageCodecs.h
//  Graphics
//
//  Created by Nicolas Tisserand on 10/31/14.
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

// FIXME: This code was extracted verbatim from arcturus. Please try and maintain both versions of the code for the time being.

#pragma once

#include <memory>
#include <cstdint>

namespace uplink {

//------------------------------------------------------------------------------

class MemoryBlock;

bool
encode_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t*        inputBuffer,
    size_t                inputSize,
    graphics_PixelFormat    inputFormat,
    size_t                inputWidth,
    size_t                inputHeight,
    MemoryBlock&          outputMemoryBlock,
    float                 outputQuality
);

bool
decode_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t*        inputBuffer,
    size_t                inputSize,
    graphics_PixelFormat    outputFormat,
    size_t&               outputWidth,
    size_t&               outputHeight,
    MemoryBlock&          outputMemoryBlock
);

//------------------------------------------------------------------------------

} // uplink namespace
