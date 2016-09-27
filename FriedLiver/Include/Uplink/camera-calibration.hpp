//
//  camera/camera-calibration.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./camera-calibration.h"
# include "./core/macros.h"
# include <cfloat>

namespace uplink {

//------------------------------------------------------------------------------

# define MEMBERS() \
         MEMBER(fx) \
         MEMBER(fy) \
         MEMBER(cx) \
         MEMBER(cy) \
         MEMBER(tx) \
         MEMBER(ty) \
         MEMBER(tz) \
         MEMBER(qx) \
         MEMBER(qy) \
         MEMBER(qz) \
         MEMBER(qw) \
         MEMBER(k1) \
         MEMBER(k2) \
         MEMBER(k3) \
         MEMBER(p1) \
         MEMBER(p2)

inline
CameraCalibration::CameraCalibration ()
    : Serializable()
# define MEMBER(Name) \
    , Name(-FLT_MAX)
         MEMBERS()
# undef  MEMBER
{}

inline bool
CameraCalibration::isValid () const
{
    return fx > 0.f;
}

inline void
CameraCalibration::reset ()
{
    *this = CameraCalibration();
}

inline bool
CameraCalibration::operator== (const CameraCalibration& rhs) const
{
// If two members are NAN, we consider they are the same.
    
# define MEMBER(Name) \
    if (Name != rhs.Name && !(isnan(Name) && isnan(rhs.Name))) \
        return false;
         MEMBERS()
# undef  MEMBER
    
    return true;
}

inline void
CameraCalibration::swapWith (CameraCalibration& other)
{
# define MEMBER(Name) \
    uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
CameraCalibration::serializeWith (Serializer& s)
{
# define MEMBER(Name) \
    report_false_unless("cannot archive camera calibration member: " #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef MEMBERS

//------------------------------------------------------------------------------

inline
CameraProperties::CameraProperties ()
: exposureTime(-1.)
, whiteBalanceMode(0xFFFF)
{}

# define MEMBERS() \
         MEMBER(exposureTime) \
         MEMBER(whiteBalanceMode) \
         MEMBER(focusPosition)

inline void
CameraProperties::swapWith (CameraProperties& other)
{
# define MEMBER(Name) \
    uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
CameraProperties::serializeWith (Serializer& s)
{
# define MEMBER(Name) \
    report_false_unless("cannot archive camera properties member: " #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef MEMBERS

//------------------------------------------------------------------------------

inline
CameraInfo::CameraInfo ()
: timestamp(-1.)
, duration(-1)
, calibration()
, properties()
, cmosAndEmitterDistance(NAN)
, referencePlaneDistance(NAN)
, planePixelSize(NAN)
, pixelSizeFactor(-1)
, streamConfig(-1)
, isRegisteredToColor(false)
, gmcEnabled(false)
, gmcSmooth(false)
, gmcStartB(NAN)
, gmcDeltaB(NAN)
{}

inline
CameraInfo::CameraInfo (double timestamp, double duration, CameraCalibration calibration, CameraProperties properties)
: timestamp(timestamp)
, duration(duration)
, calibration(calibration)
, properties(properties)
{}

# define MEMBERS() \
         MEMBER(timestamp) \
         MEMBER(duration) \
         MEMBER(calibration) \
         MEMBER(properties) \
         MEMBER(cmosAndEmitterDistance) \
         MEMBER(referencePlaneDistance) \
         MEMBER(planePixelSize) \
         MEMBER(pixelSizeFactor) \
         MEMBER(streamConfig) \
         MEMBER(isRegisteredToColor) \
         MEMBER(gmcEnabled) \
         MEMBER(gmcSmooth) \
         MEMBER(gmcStartB) \
         MEMBER(gmcDeltaB)

inline void
CameraInfo::swapWith (CameraInfo& other)
{
# define MEMBER(Name) \
    uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
CameraInfo::serializeWith (Serializer& s)
{
# define MEMBER(Name) \
    report_false_unless("cannot archive camera info member: " #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef MEMBERS

//------------------------------------------------------------------------------


}
