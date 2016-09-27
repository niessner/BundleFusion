//
//  camera/camera-pose.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/types.h"
# include "./core/serializers.h"

#if UPLINK_HAS_EIGEN
# include <Eigen/Geometry>
#endif

namespace uplink {

//------------------------------------------------------------------------------
    
// Row-major fixed size matrix. Uses the Uplink prefix to avoid name collisions
// with other libraries likely to use FixedSizeMatrix.
template <class T, int Rows, int Cols>
struct UplinkFixedSizeMatrix : public Serializable
{
public:
    UplinkFixedSizeMatrix (T defaultValue = 0) { std::fill (values, values + Rows*Cols, defaultValue); }
    
public:
    void swapWith (UplinkFixedSizeMatrix& other);
    
public:
    bool serializeWith (Serializer& serializer);
    
public:
    T values[Rows*Cols];
};

typedef UplinkFixedSizeMatrix<float,6,6> UplinkMatrix6x6f;
    
//------------------------------------------------------------------------------

class CameraPose : public Message
{
public:
    UPLINK_MESSAGE_CLASS(CameraPose)

public:
    enum CameraPoseStatusCode
    {
        CameraPoseSuccess = 0,
         CameraPoseFailed = 1,
          CameraPoseDodgy = 2,
            CameraPoseBad = 3,
    };
    
public:
    CameraPose ();

public:
    void swapWith (CameraPose& other);

public:
    bool serializeWith (Serializer& serializer);

public:
    bool isValid () const { return timestamp > -1e-5; }
    
public:
    float rx, ry, rz; // 3-2-1 euler angles in degrees.
    float tx, ty, tz;
    UplinkMatrix6x6f covarianceRt; // 6x6 covariance matrix. (0,0) is var(rx).
    int statusCode;
    double timestamp;
};

//------------------------------------------------------------------------------

#if UPLINK_HAS_EIGEN
    inline void uplinkCameraPoseToMatrix (const uplink::CameraPose& cameraPose, Eigen::Matrix4f& matrixRt);
    inline void matrixToUplinkCameraPose (const Eigen::Matrix4f& matrixRt, uplink::CameraPose& cameraPose);
    inline uplink::CameraPose toUplink (const Eigen::Isometry3f& cameraRt, double timestamp);
    inline Eigen::Isometry3f toIsometry3f (const uplink::CameraPose& cameraPose);
#endif
    
}

# include "./camera-pose.hpp"
