#pragma once

#include "cuda_SimpleMatrixUtil.h"
#include "mLib.h"

namespace MatrixConversion
{
	static ml::mat4f toMlib(const float4x4& m) {
		return ml::mat4f(m.ptr());
	}
	static ml::mat3f toMlib(const float3x3& m) {
		return ml::mat3f(m.ptr());
	}
	static ml::vec4f toMlib(const float4& v) {
		return ml::vec4f(v.x, v.y, v.z, v.w);
	}
	static ml::vec3f toMlib(const float3& v) {
		return ml::vec3f(v.x, v.y, v.z);
	}
	static ml::vec4i toMlib(const int4& v) {
		return ml::vec4i(v.x, v.y, v.z, v.w);
	}
	static ml::vec3i toMlib(const int3& v) {
		return ml::vec3i(v.x, v.y, v.z);
	}
	static float4x4 toCUDA(const ml::mat4f& m) {
		return float4x4(m.getData());
	}
	static float3x3 toCUDA(const ml::mat3f& m) {
		return float3x3(m.getData());
	}

	// dx/cuda conversion
	static vec3f toMlib(const D3DXVECTOR3& v) {
		return vec3f(v.x, v.y, v.z);
	}
	static vec4f toMlib(const D3DXVECTOR4& v) {
		return vec4f(v.x, v.y, v.z, v.w);
	}
	static mat4f toMlib(const D3DXMATRIX& m) {
		mat4f c((const float*)&m);
		return c.getTranspose();
	}

	static float4 toCUDA(const ml::vec4f& v) {
		return make_float4(v.x, v.y, v.z, v.w);
	}
	static float3 toCUDA(const ml::vec3f& v) {
		return make_float3(v.x, v.y, v.z);
	}
	static int4 toCUDA(const ml::vec4i& v) {
		return make_int4(v.x, v.y, v.z, v.w);
	}
	static int3 toCUDA(const ml::vec3i& v) {
		return make_int3(v.x, v.y, v.z);
	}


	//static matNxM<3,3> toCUDA(const Eigen::Matrix3f& m) {
	//	return matNxM<3,3>(m.data()).getTranspose();
	//}
	//static float4x4 toCUDA(const Eigen::Matrix4f& mat) {
	//	return float4x4(mat.data()).getTranspose();
	//}
	//static ml::mat4f EigToMat(const Eigen::Matrix4f& mat)
	//{
	//	return ml::mat4f(mat.data()).getTranspose();
	//}

	//static Eigen::Matrix4f MatToEig(const ml::mat4f& mat)
	//{
	//	return Eigen::Matrix4f(mat.getData()).transpose();
	//}

	//static Eigen::Vector4f VecH(const Eigen::Vector3f& v)
	//{
	//	return Eigen::Vector4f(v[0], v[1], v[2], 1.0);
	//}

	//static Eigen::Vector3f VecDH(const Eigen::Vector4f& v)
	//{
	//	return Eigen::Vector3f(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
	//}

	//static Eigen::Vector3f VecToEig(const ml::vec3f& v)
	//{
	//	return Eigen::Vector3f(v[0], v[1], v[2]);
	//}

	//static ml::vec3f EigToVec(const Eigen::Vector3f& v)
	//{
	//	return ml::vec3f(v[0], v[1], v[2]);
	//}
	//
	//static ml::mat3f EigToMat(const Eigen::Matrix3f& mat)
	//{
	//	return ml::mat3f(mat.data()).getTranspose();
	//}

	//static Eigen::Matrix3f MatToEig(const ml::mat3f& mat)
	//{
	//	return Eigen::Matrix3f(mat.getRawData()).transpose();
	//}

	//static Eigen::Matrix3f MatToEig(const matNxM<3, 3>& mat)
	//{
	//	return Eigen::Matrix3f(mat.getData()).transpose();
	//}
}
