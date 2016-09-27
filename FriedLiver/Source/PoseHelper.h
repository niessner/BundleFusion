#pragma once
#ifndef POSE_HELPER_H
#define POSE_HELPER_H
#include "GlobalDefines.h"

typedef ml::vec6f Pose;

namespace PoseHelper {

	static unsigned int countNumValidTransforms(const std::vector<mat4f>& trajectory)
	{
		unsigned int count = 0;
		for (unsigned int i = 0; i < trajectory.size(); i++) {
			if (trajectory[i][0] != -std::numeric_limits<float>::infinity()) count++;
		}
		return count;
	}

	static void composeTrajectory(unsigned int submapSize, const std::vector<mat4f>& keys, std::vector<mat4f>& all)
	{
		std::vector<mat4f> transforms;
		for (unsigned int i = 0; i < keys.size(); i++) {
			const mat4f& key = keys[i];
			transforms.push_back(key);

			const mat4f& offset = all[i*submapSize].getInverse();
			unsigned int num = std::min((int)submapSize, (int)all.size() - (int)(i * submapSize));
			for (unsigned int s = 1; s < num; s++) {
				transforms.push_back(key * offset * all[i*submapSize + s]);
			}
		}
		all = transforms;
	}

	static std::pair<float, unsigned int> evaluateAteRmse(const std::vector<mat4f>& trajectory, const std::vector<mat4f>& referenceTrajectory, unsigned int numTransforms = (unsigned int)-1) {
		if (numTransforms == (unsigned int)-1) numTransforms = (unsigned int)math::min(trajectory.size(), referenceTrajectory.size());
		if (numTransforms < 3) {
			std::cout << "cannot evaluate with < 3 transforms" << std::endl;
			if (numTransforms == 2) {
				if (referenceTrajectory[0].getTranslation().length() > 0.0001f) {
					std::cout << "cannot evaluate 2 with reference[0] not identity" << std::endl;
					return std::make_pair(-std::numeric_limits<float>::infinity(), 2);
				}
				return std::make_pair(vec3f::dist(trajectory[1].getTranslation(), referenceTrajectory[1].getTranslation()), 2);
			}
			return std::make_pair(-std::numeric_limits<float>::infinity(), numTransforms);
		}
		std::vector<vec3f> pts, refPts; //vec3f ptsMean(0.0f), refPtsMean(0.0f);
		for (unsigned int i = 0; i < numTransforms; i++) {
			if (trajectory[i][0] != -std::numeric_limits<float>::infinity() && referenceTrajectory[i][0] != -std::numeric_limits<float>::infinity()) {
				pts.push_back(trajectory[i].getTranslation());
				refPts.push_back(referenceTrajectory[i].getTranslation());
				//ptsMean += pts.back();
				//refPtsMean += refPts.back();
			}
		}
		if (pts.size() == 0) {
			std::cout << "ERROR no points to evaluate" << std::endl;
			return std::make_pair(-std::numeric_limits<float>::infinity(), 0);
		}
		//ptsMean /= (float)pts.size();
		//refPtsMean /= (float)refPts.size();
		//for (unsigned int i = 0; i < pts.size(); i++) {
		//	pts[i] -= ptsMean;
		//	refPts[i] -= refPtsMean;
		//}
		vec3f evs;
		mat4f align = EigenWrapperf::kabsch(pts, refPts, evs);
		float err = 0.0f;
		for (unsigned int i = 0; i < pts.size(); i++) {
			vec3f p0 = align * pts[i];
			vec3f p1 = refPts[i];
			float dist2 = vec3f::distSq(p0, p1);
			err += dist2;
		}
		float rmse = std::sqrt(err / pts.size());
		unsigned int num = (unsigned int)pts.size();
		return std::make_pair(rmse, num);
	}

	static mat4f getAlignmentBetweenTrajectories(const std::vector<mat4f>& trajectory, const std::vector<mat4f>& referenceTrajectory, unsigned int numTransforms = (unsigned int)-1) {
		mat4f ret; ret.setZero(-std::numeric_limits<float>::infinity());
		if (numTransforms == (unsigned int)-1) numTransforms = (unsigned int)math::min(trajectory.size(), referenceTrajectory.size());
		if (numTransforms < 3) {
			std::cout << "cannot evaluate with < 3 transforms" << std::endl;
			if (numTransforms == 2) {
				if (referenceTrajectory[0].getTranslation().length() > 0.0001f) {
					std::cout << "cannot evaluate 2 with reference[0] not identity" << std::endl;
					return ret;
				}
				return ret;
			}
			return ret;
		}
		std::vector<vec3f> pts, refPts;
		for (unsigned int i = 0; i < numTransforms; i++) {
			if (trajectory[i][0] != -std::numeric_limits<float>::infinity() && referenceTrajectory[i][0] != -std::numeric_limits<float>::infinity()) {
				pts.push_back(trajectory[i].getTranslation());
				refPts.push_back(referenceTrajectory[i].getTranslation());
			}
		}
		if (pts.size() == 0) {
			std::cout << "ERROR no points to evaluate" << std::endl;
			return ret;
		}
		vec3f evs;
		return EigenWrapperf::kabsch(pts, refPts, evs);
	}

	static std::vector<std::pair<unsigned int, float>> evaluateErr2PerImage(const std::vector<mat4f>& trajectory, const std::vector<mat4f>& referenceTrajectory) {
		std::vector<std::pair<unsigned int, float>> errors;

		size_t numTransforms = math::min(trajectory.size(), referenceTrajectory.size());
		if (numTransforms < 3) {
			std::cout << "cannot evaluate with < 3 transforms" << std::endl;
			if (numTransforms == 2) {
				if (referenceTrajectory[0].getTranslation().length() > 0.0001f) {
					std::cout << "cannot evaluate 2 with reference[0] not identity" << std::endl;
					return errors;
				}
				errors.push_back(std::make_pair(0u, vec3f::dist(trajectory[1].getTranslation(), referenceTrajectory[1].getTranslation())));
				return errors;
			}
			return errors;
		}
		std::vector<vec3f> pts, refPts;
		std::vector<unsigned int> imageIndices;
		for (unsigned int i = 0; i < numTransforms; i++) {
			if (trajectory[i][0] != -std::numeric_limits<float>::infinity() && referenceTrajectory[i][0] != -std::numeric_limits<float>::infinity()) {
				pts.push_back(trajectory[i].getTranslation());
				refPts.push_back(referenceTrajectory[i].getTranslation());
				imageIndices.push_back(i);
			}
		}
		vec3f evs;
		mat4f align = EigenWrapperf::kabsch(pts, refPts, evs);

		errors.resize(numTransforms);
		for (unsigned int i = 0; i < pts.size(); i++) {
			vec3f p0 = align * pts[i];
			vec3f p1 = refPts[i];
			float dist2 = vec3f::distSq(p0, p1);
			errors[i] = std::make_pair(imageIndices[i], dist2);
		}
		return errors;
	}

	static void saveToPoseFile(const std::string filename, const std::vector<mat4f>& trajectory) {
		std::ofstream s(filename);
		for (unsigned int i = 0; i < trajectory.size(); i++) {
			if (trajectory[i](0, 0) != -std::numeric_limits<float>::infinity()) {
				mat4f transform = trajectory[i];
				vec3f translation = transform.getTranslation();
				mat3f rotation = transform.getRotation();
				quatf quaternion(rotation);
				vec3f imag = quaternion.imag();
				float real = quaternion.real();
				s << i << " "; // time
				s << translation.x << " " << translation.y << " " << translation.z << " "; // translation
				s << imag.x << " " << imag.y << " " << imag.z << " " << real << std::endl; // rotation
			}
		}
		s.close();
	}

#ifndef USE_LIE_SPACE
	//! assumes z-y-x rotation composition (euler angles)
	static Pose MatrixToPose(const ml::mat4f& Rt) {
		ml::mat3f R = Rt.getRotation();
		ml::vec3f tr = Rt.getTranslation();

		float eps = 0.0001f;

		float psi, theta, phi; // x,y,z axis angles
		if (R(2, 0) > -1+eps && R(2, 0) < 1-eps) { // R(2, 0) != +/- 1
		//if (abs(R(2, 0) - 1) > eps && abs(R(2, 0) + 1) > eps) { // R(2, 0) != +/- 1
			theta = -asin(R(2, 0)); // \pi - theta
			float costheta = cos(theta);
			psi = atan2(R(2, 1) / costheta, R(2, 2) / costheta);
			phi = atan2(R(1, 0) / costheta, R(0, 0) / costheta);

			if (isnan(theta)) { std::cout << "ERROR MatrixToPose: NaN theta = -asin(" << R(2,0) << ")" << std::endl; getchar(); }
		}
		else {
			phi = 0;
			if (R(2, 0) <= -1 + eps) {
			//if (abs(R(2, 0) + 1) < eps) { // R(2, 0) == - 1
			//if (abs(R(2, 0) + 1) > eps) {
				theta = ml::math::PIf / 2.0f;
				psi = phi + atan2(R(0, 1), R(0, 2));
			}
			else {
				theta = -ml::math::PIf / 2.0f;
				psi = -phi + atan2(-R(0, 1), -R(0, 2));
			}
		}

		return Pose(psi, theta, phi, tr.x, tr.y, tr.z);
	}

	//! assumes z-y-x rotation composition (euler angles)
	static ml::mat4f PoseToMatrix(const Pose& ksi) {
		ml::mat4f res; res.setIdentity();
		ml::vec3f degrees;
		for (unsigned int i = 0; i < 3; i++)
			degrees[i] = ml::math::radiansToDegrees(ksi[i]);
		res.setRotationMatrix(ml::mat3f::rotationZ(degrees[2])*ml::mat3f::rotationY(degrees[1])*ml::mat3f::rotationX(degrees[0]));
		res.setTranslationVector(ml::vec3f(ksi[3], ksi[4], ksi[5]));
		return res;
	}

#else
	//https://github.com/MRPT/mrpt/blob/cc2f0d0e08b2659595c0821748be2d537a3c24b4/libs/base/src/poses/CPose3D.cpp

	static void rodrigues_so3_exp(const vec3f& w, const float A, const float B, mat3f& R)
	{
		{
			const float wx2 = w[0] * w[0];
			const float wy2 = w[1] * w[1];
			const float wz2 = w[2] * w[2];
			R(0, 0) = 1.0f - B*(wy2 + wz2);
			R(1, 1) = 1.0f - B*(wx2 + wz2);
			R(2, 2) = 1.0f - B*(wx2 + wy2);
		}
		{
			const float a = A*w[2];
			const float b = B*(w[0] * w[1]);
			R(0, 1) = b - a;
			R(1, 0) = b + a;
		}
		{
			const float a = A*w[1];
			const float b = B*(w[0] * w[2]);
			R(0, 2) = b + a;
			R(2, 0) = b - a;
		}
		{
			const float a = A*w[0];
			const float b = B*(w[1] * w[2]);
			R(1, 2) = b - a;
			R(2, 1) = b + a;
		}
	}

	static mat3f exp_rotation(const vec3f& w)
	{
		static const float one_6th = 1.0f / 6.0f;
		static const float one_20th = 1.0f / 20.0f;

		const float theta_sq = w.lengthSq(); //w*w;
		const float theta = std::sqrt(theta_sq);
		float A, B;
		//Use a Taylor series expansion near zero. This is required for
		//accuracy, since sin t / t and (1-cos t)/t^2 are both 0/0.
		if (theta_sq < 1e-8) {
			A = 1.0f - one_6th * theta_sq;
			B = 0.5f;
		}
		else {
			if (theta_sq < 1e-6) {
				B = 0.5f - 0.25f * one_6th * theta_sq;
				A = 1.0f - theta_sq * one_6th*(1.0f - one_20th * theta_sq);
			}
			else {
				const float inv_theta = 1.0f / theta;
				A = sin(theta) * inv_theta;
				B = (1 - cos(theta)) * (inv_theta * inv_theta);
			}
		}

		mat3f result;
		rodrigues_so3_exp(w, A, B, result);
		return result;
	}
	static vec3f ln_rotation(const mat3f& rotation)
	{
		vec3f result; // skew symm matrix = (R - R^T) * angle / (2 * sin(angle))

		const float cos_angle = (rotation.trace() - 1.0f) * 0.5f;
		//(R - R^T) / 2
		result[0] = (rotation(2, 1) - rotation(1, 2))*0.5f;
		result[1] = (rotation(0, 2) - rotation(2, 0))*0.5f;
		result[2] = (rotation(1, 0) - rotation(0, 1))*0.5f;

		float sin_angle_abs = result.length(); //sqrt(result*result);
		if (cos_angle > (float)0.70710678118654752440)
		{            // [0 - Pi/4[ use asin
			if (sin_angle_abs > 0){
				result *= asin(sin_angle_abs) / sin_angle_abs;
			}
		}
		else if (cos_angle > -(float)0.70710678118654752440)
		{    // [Pi/4 - 3Pi/4[ use acos, but antisymmetric part
			float angle = acos(cos_angle);
			result *= angle / sin_angle_abs;
		}
		else
		{  // rest use symmetric part
			// antisymmetric part vanishes, but still large rotation, need information from symmetric part
			const float angle = math::PIf - asin(sin_angle_abs);
			const float
				d0 = rotation(0, 0) - cos_angle,
				d1 = rotation(1, 1) - cos_angle,
				d2 = rotation(2, 2) - cos_angle;
			vec3f r2;
			if (fabs(d0) > fabs(d1) && fabs(d0) > fabs(d2))
			{ // first is largest, fill with first column
				r2[0] = d0;
				r2[1] = (rotation(1, 0) + rotation(0, 1))*0.5f;
				r2[2] = (rotation(0, 2) + rotation(2, 0))*0.5f;
			}
			else if (fabs(d1) > fabs(d2))
			{ 			    // second is largest, fill with second column
				r2[0] = (rotation(1, 0) + rotation(0, 1))*0.5f;
				r2[1] = d1;
				r2[2] = (rotation(2, 1) + rotation(1, 2))*0.5f;
			}
			else
			{							    // third is largest, fill with third column
				r2[0] = (rotation(0, 2) + rotation(2, 0))*0.5f;
				r2[1] = (rotation(2, 1) + rotation(1, 2))*0.5f;
				r2[2] = d2;
			}
			// flip, if we point in the wrong direction!
			if ((r2 | result) < 0)
				r2 *= -1;
			result = r2;
			result *= (angle / r2.length());
		}
		return result;
	}
	static Pose MatrixToPose(const mat4f& transform)
	{
		Pose result;
		const mat3f R = transform.getRotation();
		const vec3f t = transform.getTranslation();
		vec3f rot = ln_rotation(R);
		const float theta = rot.length(); //sqrt(rot*rot);

		float shtot = 0.5f;
		if (theta > 0.00001f)
			shtot = sin(theta*0.5f) / theta;

		// now do the rotation
		vec3f rot_half = rot;
		rot_half *= -0.5f;
		const mat3f halfrotator = exp_rotation(rot_half);

		vec3f rottrans = halfrotator * t;

		if (theta > 0.001f)
			rottrans -= rot * ((t | rot) * (1 - 2 * shtot) / rot.lengthSq()); //(rot*rot));
		else
			rottrans -= rot * ((t | rot) / 24);
		rottrans *= 1.0f / (2 * shtot);

		//for (int i = 0; i < 3; i++) result[i] = rot[i];
		//for (int i = 0; i < 3; i++) result[3 + i] = rottrans[i];
		for (int i = 0; i < 3; i++) result[i] = rottrans[i];
		for (int i = 0; i < 3; i++) result[3 + i] = rot[i];
		return result;
	}

	//pseudo-exponential exponentiates the rot part but just copies the trans
	static mat4f PoseToMatrix(const Pose& mu)//, bool pseudo_exponential)
	{
		vec3f translation;
		mat3f rotation;

		static const float one_6th = 1.0f / 6.0f;
		static const float one_20th = 1.0f / 20.0f;

		//vec3f mu_xyz = vec3f(mu[3], mu[4], mu[5]);
		//vec3f w = mu.getVec3();
		vec3f mu_xyz = mu.getVec3();
		vec3f w = vec3f(mu[3], mu[4], mu[5]);

		const float theta_sq = w.lengthSq(); // w*w;
		const float theta = std::sqrt(theta_sq);
		float A, B;

		vec3f cross = w ^ mu_xyz;

		if (theta_sq < 1e-8)
		{
			A = 1.0f - one_6th * theta_sq;
			B = 0.5f;
			//if (!pseudo_exponential) {
			translation[0] = mu_xyz[0] + 0.5f * cross[0];
			translation[1] = mu_xyz[1] + 0.5f * cross[1];
			translation[2] = mu_xyz[2] + 0.5f * cross[2];
			//}
		}
		else
		{
			float C;
			if (theta_sq < 1e-6) {
				C = one_6th*(1.0f - one_20th * theta_sq);
				A = 1.0f - theta_sq * C;
				B = 0.5f - 0.25f * one_6th * theta_sq;
			}
			else {
				const float inv_theta = 1.0f / theta;
				A = sin(theta) * inv_theta;
				B = (1 - cos(theta)) * (inv_theta * inv_theta);
				C = (1 - A) * (inv_theta * inv_theta);
			}

			vec3f w_cross = w ^ cross;
			//if (!pseudo_exponential) { //result.get_translation() = mu_xyz + B * cross + C * (w ^ cross);
				translation[0] = mu_xyz[0] + B * cross[0] + C * w_cross[0];
				translation[1] = mu_xyz[1] + B * cross[1] + C * w_cross[1];
				translation[2] = mu_xyz[2] + B * cross[2] + C * w_cross[2];
			//}
		}

		// 3x3 rotation part:
		rodrigues_so3_exp(w, A, B, rotation);

		//if (pseudo_exponential) out_pose.m_coords = mu_xyz;
		mat4f result = mat4f::identity();
		result.setRotationMatrix(rotation);
		result.setTranslationVector(translation);
		return result;
	}


	//static mat3f VectorToSkewSymmetricMatrix(const vec3f& v) {
	//	mat3f res = mat3f::zero();
	//	res(1, 0) = v[2];
	//	res(2, 0) = -v[1];
	//	res(2, 1) = v[0];
	//	res(0, 1) = -v[2];
	//	res(0, 2) = v[1];
	//	res(1, 2) = -v[0];
	//	return res;
	//}
	////! exponential map: so(3) -> SO(3) / w -> R3x3
	//static mat3f LieAlgebraToLieGroupSO3(const vec3f& w) {
	//	float norm = w.length();
	//	if (norm == 0.0f)
	//		return mat3f::identity();
	//	mat3f wHat = VectorToSkewSymmetricMatrix(w);

	//	mat3f res = mat3f::identity();
	//	res += wHat * (sin(norm) / norm);
	//	res += (wHat * wHat) * ((1.0f - cos(norm)) / (norm*norm));
	//	return res;
	//}

	//// LieAlgebraToLieGroupSE3
	//static mat4f PoseToMatrix(const vec6f& ksi) {
	//	vec3f w;
	//	vec3f t;

	//	for (int i = 0; i < 3; ++i) {
	//		w[i] = ksi[i];
	//		t[i] = ksi[i + 3];
	//	}
	//	float norm = w.length();
	//	vec3f trans;
	//	mat3f rot = LieAlgebraToLieGroupSO3(w);
	//	if (norm == 0.0f) {
	//		trans = vec3f::origin;
	//	}
	//	else {
	//		mat3f skewSymmetricW = VectorToSkewSymmetricMatrix(w);
	//		mat3f skewSymmetricW2 = skewSymmetricW * skewSymmetricW;
	//		mat3f V = mat3f::identity();
	//		V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
	//		V += skewSymmetricW2 * ((norm - sin(norm)) / (norm * norm * norm));
	//		trans = V * t;
	//	}
	//	mat4f res = mat4f::identity();
	//	res.setRotationMatrix(rot);
	//	res.setTranslationVector(trans);
	//	return res;
	//}

	//static vec3f SkewSymmetricMatrixToVector(const mat3f& m) {
	//	vec3f res;
	//	res[0] = m(2, 1);
	//	res[1] = m(0, 2);
	//	res[2] = m(1, 0);
	//	return res;
	//}

	////! logarithm map: SO(3) -> so(3) / R3x3 -> w
	//static vec3f LieGroupToLieAlgebraSO3(const mat3f& R) {
	//	float tmp = (R.trace() - 1.0f) / 2.0f;

	//	float angleOfRotation = acos(clamp(tmp, -1.0f, 1.0f));
	//	if (angleOfRotation == 0.0f)
	//		return vec3f::origin;
	//	mat3f lnR = (R - R.getTranspose()) * (angleOfRotation / (2.0f * sin(angleOfRotation)));
	//	return SkewSymmetricMatrixToVector(lnR);
	//}

	//// LieGroupToLieAlgebraSE3
	//static vec6f MatrixToPose(const mat4f& Rt) {
	//	mat3f R = Rt.getRotation();
	//	vec3f tr = Rt.getTranslation();

	//	vec3f w = LieGroupToLieAlgebraSO3(R);

	//	float	norm = w.length();
	//	mat3f skewSymmetricW = VectorToSkewSymmetricMatrix(w);
	//	mat3f skewSymmetricW2 = skewSymmetricW * skewSymmetricW;
	//	mat3f V = mat3f::identity();
	//	if (norm > 0.0f)	{
	//		V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
	//		V += skewSymmetricW2 * ((norm - sin(norm)) / (norm * norm * norm));
	//	}
	//	//else {
	//	//	V += skewSymmetricW * (1.0f - cos(norm));  // 1 - cos(0) = 0
	//	//	V += skewSymmetricW2 * (norm - sin(norm)); // 0 - sin(0) = 0
	//	//}
	//	vec3f t = V.getInverse() * tr;

	//	vec6f ksi(w.x, w.y, w.z, t.x, t.y, t.z);
	//	return ksi;
	//}
#endif

	static std::vector<Pose> convertToPoses(const std::vector<mat4f>& matrices) {
		std::vector<Pose> poses(matrices.size());

		for (unsigned int i = 0; i < matrices.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(matrices[i]);

		return poses;
	}
	static std::vector<mat4f> convertToMatrices(const std::vector<Pose>& poses) {
		std::vector<mat4f> matrices(poses.size());

		for (unsigned int i = 0; i < poses.size(); i++)
			matrices[i] = PoseHelper::PoseToMatrix(poses[i]);

		return matrices;
	}
}

#endif