//
//  camera/camera-calibration.h
//  Uplink
//
//  Copyright (c) 2013 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/types.h"
# include "./message.h"
# include <cmath>

namespace uplink {

//------------------------------------------------------------------------------

class CameraCalibration : public Serializable
{
public:
    CameraCalibration ();

public:
    bool isValid () const;

public:
    void reset ();

public:
    void swapWith (CameraCalibration&);

public:
    bool serializeWith (Serializer& serializer);
    
public:
    bool operator== (const CameraCalibration& rhs) const;

public:
    float fx;
    float fy;
    float cx;
    float cy;
    float tx, ty, tz;
    float qx, qy, qz, qw;
    float k1, k2, k3;
    float p1, p2;
};

//------------------------------------------------------------------------------

// FIXME: This should be a more free-form property bag, eventually.
struct CameraProperties : Serializable
{
    CameraProperties ();
    
    virtual void swapWith (CameraProperties& other);

    virtual bool serializeWith (Serializer& serializer);

    double exposureTime;
    uint16 whiteBalanceMode;
    float  focusPosition;
};

//------------------------------------------------------------------------------

struct CameraInfo : Serializable
{
    CameraInfo ();
    CameraInfo (double timestamp, double duration, CameraCalibration calibration, CameraProperties properties);
    
    virtual void swapWith (CameraInfo& other);
    
    virtual bool serializeWith (Serializer& serializer);

    double            timestamp;
    double            duration;
    CameraCalibration calibration;
    CameraProperties  properties;

    // FIXME: Split depth and color camera info.
    
    // Fixed params
    float cmosAndEmitterDistance;
    float referencePlaneDistance;
    float planePixelSize;

    // Depth stream params
    int32 pixelSizeFactor;
    int32 streamConfig;
    bool  isRegisteredToColor;

    // GMC
    bool  gmcEnabled;
    bool  gmcSmooth;
    float gmcStartB;
    float gmcDeltaB;
};

//------------------------------------------------------------------------------

}

# include "camera-calibration.hpp"
