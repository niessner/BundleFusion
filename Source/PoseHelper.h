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

	//! assumes z-y-x rotation composition (euler angles)
	static Pose MatrixToPose(const ml::mat4f& Rt) {
		ml::mat3f R = Rt.getRotation();
		ml::vec3f tr = Rt.getTranslation();

		float eps = 0.0001f;

		float psi, theta, phi; // x,y,z axis angles
		if (abs(R(2, 0) - 1) > -1+eps && abs(R(2, 0) + 1) > 1-eps) { // R(2, 0) != +/- 1
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
	static std::vector<Pose> convertToPoses(const std::vector<ml::mat4f>& matrices) {
		std::vector<Pose> poses(matrices.size());

		for (unsigned int i = 0; i < matrices.size(); i++)
			poses[i] = PoseHelper::MatrixToPose(matrices[i]);

		return poses;
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
}

#endif