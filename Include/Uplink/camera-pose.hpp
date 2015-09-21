//
//  camera/camera-pose.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./camera-pose.h"
# include "./core/macros.h"

namespace uplink {

//------------------------------------------------------------------------------
    
template <class T, int Rows, int Cols>
inline void
UplinkFixedSizeMatrix<T,Rows,Cols> :: swapWith (UplinkFixedSizeMatrix<T,Rows,Cols>& other)
{
    for (int i = 0; i < Rows*Cols; ++i)
        uplink_swap(values[i], other.values[i]);
}

template <class T, int Rows, int Cols>
inline bool
UplinkFixedSizeMatrix<T,Rows,Cols> :: serializeWith (Serializer& serializer)
{
    bool ok = serializer.putBytes (reinterpret_cast<Byte*>(values), sizeof(values));
    report_false_unless ("cannot archive fixed size matrix", ok);
    return true;
}
    
//------------------------------------------------------------------------------

# define MEMBERS()  \
         MEMBER(rx) \
         MEMBER(ry) \
         MEMBER(rz) \
         MEMBER(tx) \
         MEMBER(ty) \
         MEMBER(tz) \
         MEMBER(covarianceRt) \
         MEMBER(statusCode) \
         MEMBER(timestamp)

inline
CameraPose::CameraPose ()
    : Message()
# define MEMBER(Name) \
    , Name(0)
         MEMBERS()
# undef  MEMBER
{
    // Mark is as invalid.
    timestamp = -1.0;
    statusCode = CameraPoseFailed;
}

inline void
CameraPose::swapWith (CameraPose& other)
{
    Message::swapWith(other);

# define MEMBER(Name) \
    uplink_swap(Name, other.Name);
         MEMBERS()
# undef  MEMBER
}

inline bool
CameraPose::serializeWith (Serializer& s)
{
# define MEMBER(Name) \
    report_false_unless("cannot archive camera pose " #Name, s.put(Name));
         MEMBERS()
# undef  MEMBER

    return true;
}

# undef MEMBERS

//------------------------------------------------------------------------------

#if UPLINK_HAS_EIGEN
    
    inline void uplinkCameraPoseToMatrix (const uplink::CameraPose& cameraPose, Eigen::Matrix4f& matrixRt)
    {
        Eigen::Isometry3f rbtGuess = Eigen::Isometry3f::Identity();
        Eigen::Vector3f translation;
        translation << cameraPose.tx, cameraPose.ty, cameraPose.tz;
        
        // TODO: Euler angles are not ideal. A real implementation should be over SO3.
        // This corresponds directly to the rot2euler and euler2rot functions in canopus.
        Eigen::Matrix3f rotation;
        rotation = Eigen::AngleAxisf(cameraPose.rz*DEG2RAD, Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(cameraPose.ry*DEG2RAD, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxisf(cameraPose.rx*DEG2RAD, Eigen::Vector3f::UnitX());
        
        rbtGuess.translate(translation);
        rbtGuess.rotate(rotation);
        matrixRt = rbtGuess.matrix ();
    }
    
    inline void matrixToUplinkCameraPose (const Eigen::Matrix4f& matrixRt, uplink::CameraPose& cameraPose)
    {
        const Eigen::Matrix3f& R = matrixRt.block<3,3>(0,0);
        
        // Heading
        float h = atan2(R(1,0), R(0,0));
        
        // Compute cos & sin of heading
        float ch = cos(h);
        float sh = sin(h);
        
        //Pitch
        float p = atan2(-R(2,0), R(0,0)*ch + R(1,0)*sh);
        
        // Roll
        float r = atan2(R(0,2)*sh - R(1,2)*ch, -R(0,1)*sh + R(1,1)*ch);
        
        Eigen::Vector3f euler;
        euler << r*RAD2DEG, p*RAD2DEG, h*RAD2DEG;
        
        cameraPose.rx = euler[0];
        cameraPose.ry = euler[1];
        cameraPose.rz = euler[2];
        cameraPose.tx = matrixRt(0,3);
        cameraPose.ty = matrixRt(1,3);
        cameraPose.tz = matrixRt(2,3);
    }
    
    inline uplink::CameraPose toUplink (const Eigen::Isometry3f& cameraRt, double timestamp)
    {
        uplink::CameraPose output;
        output.timestamp = timestamp;
        matrixToUplinkCameraPose(cameraRt.matrix(), output);
        return output;
    }
    
    inline Eigen::Isometry3f toIsometry3f (const uplink::CameraPose& cameraPose)
    {
        Eigen::Isometry3f output;
        uplinkCameraPoseToMatrix(cameraPose, output.matrix());
        return output;
    }
    
#endif // UPLINK_HAS_EIGEN

//------------------------------------------------------------------------------
    
}
