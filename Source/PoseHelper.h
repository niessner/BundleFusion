#pragma once
#ifndef POSE_HELPER_H
#define POSE_HELPER_H

typedef ml::vec6f Pose;

namespace PoseHelper {

	static float evaluateAteRmse(const std::vector<mat4f>& trajectory, const std::vector<mat4f>& referenceTrajectory) {
		size_t numTransforms = math::min(trajectory.size(), referenceTrajectory.size());
		if (numTransforms < 3) {
			std::cout << "cannot evaluate with < 3 transforms" << std::endl;
			return -std::numeric_limits<float>::infinity();
		}
		std::vector<vec3f> pts, refPts; vec3f ptsMean(0.0f), refPtsMean(0.0f);
		for (unsigned int i = 0; i < numTransforms; i++) {
			if (trajectory[i][0] != -std::numeric_limits<float>::infinity()) {
				pts.push_back(trajectory[i].getTranslation());
				refPts.push_back(referenceTrajectory[i].getTranslation());
				ptsMean += pts.back();
				refPtsMean += refPts.back();
			}
		}
		ptsMean /= (float)pts.size();
		refPtsMean /= (float)refPts.size();
		for (unsigned int i = 0; i < pts.size(); i++) {
			pts[i] -= ptsMean;
			refPts[i] -= refPtsMean;
		}
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
		return rmse;
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
			theta = -asin(R(2, 0)); // \pi - theta
			float costheta = cos(theta);
			psi = atan2(R(2, 1) / costheta, R(2, 2) / costheta);
			phi = atan2(R(1, 0) / costheta, R(0, 0) / costheta);

			if (isnan(theta)) { std::cout << "ERROR MatrixToPose: NaN theta = -asin(" << R(2,0) << ")" << std::endl; getchar(); }
		}
		else {
			phi = 0;
			if (abs(R(2, 0) + 1) < eps) { // R(2, 0) == - 1
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
		res.setRotation(ml::mat3f::rotationZ(degrees[2])*ml::mat3f::rotationY(degrees[1])*ml::mat3f::rotationX(degrees[0]));
		res.setTranslationVector(ml::vec3f(ksi[3], ksi[4], ksi[5]));
		return res;
	}

	static std::vector<ml::mat4f> convertToMatrices(const std::vector<Pose>& poses) {
		std::vector<ml::mat4f> matrices(poses.size());

		for (unsigned int i = 0; i < poses.size(); i++)
			matrices[i] = PoseHelper::PoseToMatrix(poses[i]);

		return matrices;
	}
#else
	static mat3f VectorToSkewSymmetricMatrix(const vec3f& v) {
		mat3f res = mat3f::zero();
		res(1, 0) = v[2];
		res(2, 0) = -v[1];
		res(2, 1) = v[0];
		res(0, 1) = -v[2];
		res(0, 2) = v[1];
		res(1, 2) = -v[0];
		return res;
	}

	//! exponential map: so(3) -> SO(3) / w -> R3x3
	static mat3f LieAlgebraToLieGroupSO3(const vec3f& w) {
		float norm = w.length();
		if (norm == 0.0f)
			return mat3f::identity();
		mat3f wHat = VectorToSkewSymmetricMatrix(w);

		mat3f res = mat3f::identity();
		res += wHat * (sin(norm) / norm);
		res += (wHat * wHat) * ((1.0f - cos(norm)) / (norm*norm));
		return res;
	}

	// LieAlgebraToLieGroupSE3
	static mat4f PoseToMatrix(const vec6f& ksi) {
		vec3f w;
		vec3f t;

		for (int i = 0; i < 3; ++i) {
			w[i] = ksi[i];
			t[i] = ksi[i + 3];
		}
		float norm = w.length();
		vec3f trans;
		mat3f rot = LieAlgebraToLieGroupSO3(w);
		if (norm == 0.0f) {
			trans = vec3f::origin;
		}
		else {
			mat3f skewSymmetricW = VectorToSkewSymmetricMatrix(w);
			mat3f skewSymmetricW2 = skewSymmetricW * skewSymmetricW;
			mat3f V = mat3f::identity();
			V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
			V += skewSymmetricW2 * ((norm - sin(norm)) / (norm * norm * norm));
			trans = V * t;
		}
		mat4f res = mat4f::identity();
		res.setRotation(rot);
		res.setTranslationVector(trans);
		return res;
	}

	static vec3f SkewSymmetricMatrixToVector(const mat3f& m) {
		vec3f res;
		res[0] = m(2, 1);
		res[1] = m(0, 2);
		res[2] = m(1, 0);
		return res;
	}

	//! logarithm map: SO(3) -> so(3) / R3x3 -> w
	static vec3f LieGroupToLieAlgebraSO3(const mat3f& R) {
		float tmp = (R.trace() - 1.0f) / 2.0f;

		if (tmp < -1.0f)
			tmp = -1.0f;
		if (tmp > 1.0f)
			tmp = 1.0f;
		float angleOfRotation = acos(tmp);
		if (angleOfRotation == 0.0f)
			return vec3f::origin;
		mat3f lnR = (R - R.getTranspose()) * (angleOfRotation / (2.0f * sin(angleOfRotation)));
		return SkewSymmetricMatrixToVector(lnR);
	}

	// LieGroupToLieAlgebraSE3
	static vec6f MatrixToPose(const mat4f& Rt) {
		mat3f R = Rt.getRotation();
		vec3f tr = Rt.getTranslation();

		vec3f w = LieGroupToLieAlgebraSO3(R);

		float	norm = w.length();
		mat3f skewSymmetricW = VectorToSkewSymmetricMatrix(w);
		mat3f skewSymmetricW2 = skewSymmetricW * skewSymmetricW;
		mat3f V = mat3f::identity();
		if (norm > 0.0f)	{
			V += skewSymmetricW * ((1.0f - cos(norm)) / (norm * norm));
			V += skewSymmetricW2 * ((norm - sin(norm)) / (norm * norm * norm));
		}
		//else {
		//	V += skewSymmetricW * (1.0f - cos(norm));  // 1 - cos(0) = 0
		//	V += skewSymmetricW2 * (norm - sin(norm)); // 0 - sin(0) = 0
		//}
		vec3f t = V.getInverse() * tr;

		vec6f ksi(w.x, w.y, w.z, t.x, t.y, t.z);
		return ksi;
	}
#endif

	static std::vector<Pose> convertToPoses(const std::vector<ml::mat4f>& matrices) {
		std::vector<Pose> poses(matrices.size());

		for (unsigned int i = 0; i < matrices.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(matrices[i]);

		return poses;
	}
}

#endif