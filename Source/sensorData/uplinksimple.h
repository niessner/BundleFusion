//
//  core/platform.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

//------------------------------------------------------------------------------

// Platform-independent definitions

# if __cplusplus > 199711L
#   define UPLINK_HAS_CXX11 1
# endif

//------------------------------------------------------------------------------

// Platform-specific workarounds

# if _MSC_VER
#   define _WINSOCK_DEPRECATED_NO_WARNINGS 1
#ifndef _CRT_SECURE_NO_WARNINGS
#   define _CRT_SECURE_NO_WARNINGS 1
#endif
#ifndef _SCL_SECURE_NO_WARNINGS
#   define _SCL_SECURE_NO_WARNINGS 1
#endif
# endif

//------------------------------------------------------------------------------

// Platform-independent API


//------------------------------------------------------------------------------

# define header_only void __dummy_symbol__##__COUNTER__ () {}

# define zero_delete(Ptr) delete (Ptr); (Ptr) = 0;

# define sizeof_array(Array) sizeof(Array) / sizeof(Array[0])

# define non_copyable(Class)              \
private:                                  \
    Class (const Class& copy);            \
    Class& operator = (const Class& rhs);

# define unimplemented assert(false)

// FIXME: Generate all the combinations, eventually.

# define break_if(...)     if ( (__VA_ARGS__)) break;
# define break_unless(...) if (!(__VA_ARGS__)) break;

# define continue_if(...)      if ( (__VA_ARGS__)) continue;
# define continue_unless(...)  if (!(__VA_ARGS__)) continue;

# define return_if(...)           if ( (__VA_ARGS__)) { return; }
# define return_unless(...)       if (!(__VA_ARGS__)) { return; }

# define return_true_if(...)      if ( (__VA_ARGS__)) { return true; }
# define return_true_unless(...)  if (!(__VA_ARGS__)) { return true; }

# define return_false_if(...)     if ( (__VA_ARGS__)) { return false; }
# define return_false_unless(...) if (!(__VA_ARGS__)) { return false; }

# define return_zero_if(...)      if ( (__VA_ARGS__)) { return 0; }
# define return_zero_unless(...)  if (!(__VA_ARGS__)) { return 0; }

# define return_if_zero(...)           if (0 == (__VA_ARGS__)) { return; }
# define return_if_non_zero(...)       if (0 != (__VA_ARGS__)) { return; }

# define return_true_if_zero(...)      if (0 == (__VA_ARGS__)) { return true; }
# define return_true_if_non_zero(...)  if (0 != (__VA_ARGS__)) { return true; }

# define return_false_if_zero(...)     if (0 == (__VA_ARGS__)) { return false; }
# define return_false_if_non_zero(...) if (0 != (__VA_ARGS__)) { return false; }

# define return_zero_if_zero(...)      if (0 == (__VA_ARGS__)) { return 0; }
# define return_zero_if_non_zero(...)  if (0 != (__VA_ARGS__)) { return 0; }

# define report_continue_if(What, ...)      if ( (__VA_ARGS__)) { uplink_log_error(What); continue; }
# define report_continue_unless(What, ...)  if (!(__VA_ARGS__)) { uplink_log_error(What); continue; }

# define report_if(What, ...)       if ((__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_false_if(What, ...) if ((__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_zero_if(What, ...)  if ((__VA_ARGS__)) { uplink_log_error(What); return 0; }

# define report_unless(What, ...)       if (!(__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_false_unless(What, ...) if (!(__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_zero_unless(What, ...)  if (!(__VA_ARGS__)) { uplink_log_error(What); return 0; }

# define report_if_zero(What, ...)       if (0 == (__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_if_non_zero(What, ...)   if (0 != (__VA_ARGS__)) { uplink_log_error(What); return; }

# define report_false_if_zero(What, ...)     if (0 == (__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_false_if_non_zero(What, ...) if (0 != (__VA_ARGS__)) { uplink_log_error(What); return false; }

# define report_zero_if_zero(What, ...)      if (0 == (__VA_ARGS__)) { uplink_log_error(What); return 0; }
# define report_zero_if_non_zero(What, ...)  if (0 != (__VA_ARGS__)) { uplink_log_error(What); return 0; }

//------------------------------------------------------------------------------

# include "uplinksimple_memory.h"
# include "uplinksimple_image-codecs.h"

namespace uplinksimple {


//------------------------------------------------------------------------------

// NOTE: JPEG quality values are remapped across platform-specific implementations in order to consistently mimic IJG's libjpeg quality scale.
// Some typical values used at Occipital are:
//     .8162f for Structure app's Uplink color camera frames.
//     .8f for Skanect's Uplink feedback images.
//     .95f for Skanect's on-disk color camera frames.

static const float defaultQuality = .95f;
static const float occQuality = 0.98f;
static const float UseDefaultQuality = -12345.0;

enum graphics_PixelFormat // FIXME: Solve the tension with Image and use a shorter name.
{
    graphics_PixelFormat_Gray,
    graphics_PixelFormat_RGB,
    graphics_PixelFormat_RGBA,
    graphics_PixelFormat_YCbCr,
};

enum graphics_ImageCodec // FIXME: Solve the tension with Image and use a shorter name.
{
    graphics_ImageCodec_JPEG,
    graphics_ImageCodec_PNG,
    graphics_ImageCodec_BMP,
    graphics_ImageCodec_Hardware_JPEG
};

bool
encode_image (
    graphics_ImageCodec     imageCodec,
    const uint8_t*        inputBuffer,
    size_t                inputSize,
    graphics_PixelFormat    inputFormat,
    size_t                inputWidth,
    size_t                inputHeight,
    MemoryBlock&          outputMemoryBlock,
    float                 outputQuality = defaultQuality // ] 0.f .. 1.f [
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


#include "uplinksimple_windows-image-codecs.h"

