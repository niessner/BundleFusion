#pragma once

#include "cudaUtil.h"
#include "mLib.h"

namespace Util
{
	static void writeToImage(const float* d_buffer, unsigned int width, unsigned int height, const std::string& filename) {
		//bool minfInvalid = false;

		float* h_buffer = new float[width * height];
		cudaMemcpy(h_buffer, d_buffer, sizeof(float)*width*height, cudaMemcpyDeviceToHost);

		//for (unsigned int i = 0; i < width*height; i++) {
		//	if (h_buffer[i] == -std::numeric_limits<float>::infinity()) {
		//		minfInvalid = true;
		//		break;
		//	}
		//}
		//if (!minfInvalid) std::cout << "invalid valid != MINF" << std::endl;
		//else std::cout << "MINF invalid value" << std::endl;

		DepthImage32 dimage(width, height, h_buffer);
		dimage.setInvalidValue(-std::numeric_limits<float>::infinity());
		ColorImageR32G32B32 cImage(dimage);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
	}

	static void writeToImage(const float4* d_buffer, float min, float max, unsigned int width, unsigned int height, const std::string& filename) {
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		vec3f* data = new vec3f[size];
		for (unsigned int i = 0; i < size; i++) {
			data[i] = vec3f(h_buffer[i*4], h_buffer[i*4+1], h_buffer[i*4+2]);
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				data[i] = (data[i] - min) / (max - min);
			}
		}
		std::cout << "range: [" << min << ", " << max << "]" << std::endl;
		ColorImageR32G32B32 cImage(width, height, data);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
		SAFE_DELETE_ARRAY(data);
	}

	static void writeToImage(const float4* d_buffer, unsigned int width, unsigned int height, const std::string& filename) {
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		vec3f* data = new vec3f[size];
		float min = std::numeric_limits<float>::infinity(), max = -std::numeric_limits<float>::infinity();
		for (unsigned int i = 0; i < size; i++) {
			data[i] = vec3f(h_buffer[i*4], h_buffer[i*4+1], h_buffer[i*4+2]);
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				if (data[i].x < min) min = data[i].x;
				if (data[i].y < min) min = data[i].y;
				if (data[i].z < min) min = data[i].z;

				if (data[i].x > max) max = data[i].x;
				if (data[i].y > max) max = data[i].y;
				if (data[i].z > max) max = data[i].z;
			}
		}
		for (unsigned int i = 0; i < size; i++) {
			if (data[i].x != -std::numeric_limits<float>::infinity()) {
				data[i] = (data[i] - min) / (max - min);
			}
		}
		std::cout << "range: [" << min << ", " << max << "]" << std::endl;
		ColorImageR32G32B32 cImage(width, height, data);
		FreeImageWrapper::saveImage(filename, cImage);

		SAFE_DELETE_ARRAY(h_buffer);
		SAFE_DELETE_ARRAY(data);
	}

	static float* loadFloat4FromBinary(const std::string& filename, unsigned int& width, unsigned int& height, unsigned int& numChannels)
	{
		BinaryDataStreamFile s(filename, false);
		s >> width;
		s >> height;
		s >> numChannels;
		float * result = new float[numChannels*width*height];
		s.readData((BYTE*)result, width*height*sizeof(float)*numChannels);
		s.close();
		return result;
	}

	static float* loadFloatFromBinary(const std::string& filename, unsigned int& size)
	{
		BinaryDataStreamFile s(filename, false);
		s >> size;
		float * result = new float[size];
		s.readData((BYTE*)result, size*sizeof(float));
		s.close();
		return result;
	}

	static void saveFloat4ToBinary(float4* d_buffer, unsigned int width, unsigned int height, const std::string& filename) 
	{
		unsigned int size = width * height;
		float* h_buffer = new float[4 * size];
		cudaMemcpy(h_buffer, d_buffer, 4*sizeof(float)*size, cudaMemcpyDeviceToHost);

		BinaryDataStreamFile s(filename, true);
		s << width << height << 4;
		s.writeData((BYTE*)h_buffer, size*sizeof(float)*4);
		s.close();

		SAFE_DELETE_ARRAY(h_buffer);
	}

	static void getRange(const float* d_buffer, unsigned int size, float& min, float& max)
	{
		min = std::numeric_limits<float>::infinity();
		max = -std::numeric_limits<float>::infinity();

		float* h_buffer = new float[size];
		cudaMemcpy(h_buffer, d_buffer, sizeof(float)*size, cudaMemcpyDeviceToHost);

		for (unsigned int i = 0; i < size; i++) {
			if (h_buffer[i] != -std::numeric_limits<float>::infinity()) {
				if (h_buffer[i] < min) min = h_buffer[i];
				if (h_buffer[i] > max) max = h_buffer[i];
			}
		}

		SAFE_DELETE_ARRAY(h_buffer);
	}

	static vec3f depthToSkeleton(unsigned int ux, unsigned int uy, float* depthImage, unsigned int width, const CalibrationData& depthCalibration)
	{
		float depth = depthImage[uy*width+ux];
		return (depthCalibration.m_IntrinsicInverse * vec4f((float)ux*depth, (float)uy*depth, depth, 1.0f)).getVec3();
	}

	static vec3f getNormal(unsigned int x, unsigned int y, float* depth, unsigned int width, unsigned int height, const CalibrationData& depthCalibration)
	{
		vec3f ret(-std::numeric_limits<float>::infinity());
		if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
			vec3f cc = depthToSkeleton(x,y, depth, width, depthCalibration);
			vec3f pc = depthToSkeleton(x+1,y+0, depth, width, depthCalibration);
			vec3f cp = depthToSkeleton(x+0,y+1, depth, width, depthCalibration);
			vec3f mc = depthToSkeleton(x-1,y+0, depth, width, depthCalibration);
			vec3f cm = depthToSkeleton(x+0,y-1, depth, width, depthCalibration);

			if (cc.x != -std::numeric_limits<float>::infinity() && pc.x != -std::numeric_limits<float>::infinity() && cp.x != -std::numeric_limits<float>::infinity() && mc.x != -std::numeric_limits<float>::infinity() && cm.x != -std::numeric_limits<float>::infinity())
			{
				vec3f n = (pc - mc) ^ (cp - cm);
				float l = n.length();
				if (l > 0.0f) {
					ret = n/l;
				}
			}
		}
		return ret;
	}

	//! host data
	static void computePointCloud(float* h_depth, vec4uc* h_color, const CalibrationData& depthCalibration, const CalibrationData& colorCalibration, const vec4ui& dimensions, const mat4f& transform, PointCloudf& pc) {
		unsigned int depthWidth = dimensions.x;
		unsigned int depthHeight = dimensions.y;
		unsigned int colorWidth = dimensions.z;
		unsigned int colorHeight = dimensions.w;

		for (unsigned int i = 0; i < depthWidth*depthHeight; i++) {
			float depth = h_depth[i];
			if (depth != -std::numeric_limits<float>::infinity() && depth != -FLT_MAX && depth != 0.0f) {

				unsigned int ux = i % depthWidth;
				unsigned int uy = i / depthWidth;
				vec3f p = depthToSkeleton(ux, uy, h_depth, depthWidth, depthCalibration);

				vec3f n = getNormal(ux, uy, h_depth, depthWidth, depthHeight, depthCalibration);
				if (n.x != -std::numeric_limits<float>::infinity()) {

					ml::vec3f cp = colorCalibration.m_Intrinsic * p;
					vec2i colorCoords = math::round(vec2f(cp.x / cp.z, cp.y / cp.z));
					if (colorCoords.x >= 0 && colorCoords.y >= 0 && colorCoords.x < (int)colorWidth && colorCoords.y < (int)colorHeight) {
						vec4uc c = h_color[colorCoords.y * colorWidth + colorCoords.x];
						pc.m_colors.push_back(vec4f(c) / 255.0f);
						pc.m_points.push_back(p);
					} // valid color
				} // valid normal
			} // valid depth
		} // i
		for (auto& p : pc.m_points) {
			p = transform * p;
		}
		mat4f invTranspose = transform.getInverse().getTranspose();
		for (auto& n : pc.m_normals) {
			n = invTranspose * n;
			n.normalize();
		}
	}

	static void computePointCloud(float* h_depth, vec4f* h_color, const CalibrationData& depthCalibration, const CalibrationData& colorCalibration, const vec4ui& dimensions, const mat4f& transform, PointCloudf& pc) {
		vec4uc* color = new vec4uc[dimensions.z * dimensions.w];
		for (unsigned int i = 0; i < dimensions.z * dimensions.w; i++) {
			color[i] = vec4uc(h_color[i] * 255.0f);
		}
		computePointCloud(h_depth, color, depthCalibration, colorCalibration, dimensions, transform, pc);
		SAFE_DELETE_ARRAY(color);
	}

	//! device data
	static void computePointCloud(float* d_depth, float4* d_color, const CalibrationData& depthCalibration, const CalibrationData& colorCalibration, const vec4ui& dimensions, const mat4f& transform, PointCloudf& pc) {
		float* depth = new float[dimensions.x * dimensions.y];
		vec4f* color = new vec4f[dimensions.z * dimensions.w];
		cudaMemcpy(depth, d_depth, sizeof(float) * dimensions.x * dimensions.y, cudaMemcpyDeviceToHost);
		cudaMemcpy(color, d_color, sizeof(vec4f) * dimensions.z * dimensions.w, cudaMemcpyDeviceToHost);
		computePointCloud(depth, color, depthCalibration, colorCalibration, dimensions, transform, pc);
		SAFE_DELETE_ARRAY(depth);
		SAFE_DELETE_ARRAY(color);
	}

	static void savePointCloud(const std::string& filename, const std::list<PointCloudf>& recordedPoints) {
		std::cout << "recorded " << recordedPoints.size() << " frames" << std::endl;
		PointCloudf pc;
		for (const auto& p : recordedPoints ) {
			pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
			pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
			pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
		}
		PointCloudIOf::saveToFile(filename, pc);
	}

}
