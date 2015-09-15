
#include "stdafx.h"

#include "RGBDSensor.h"
#include <limits>

RGBDSensor::RGBDSensor()
{
	m_depthWidth  = 0;
	m_depthHeight = 0;

	m_colorWidth  = 0;
	m_colorHeight = 0;

	m_colorRGBX = NULL;

	m_currentRingBufIdx = 0;
}

void RGBDSensor::init(unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight, unsigned int depthRingBufferSize)
{
	std::cout << "sensor dimensions depth ( " << depthWidth << " / " << depthHeight <<" )" << std::endl;
	std::cout << "sensor dimensions color ( " << colorWidth << " / " << colorHeight <<" )" << std::endl;
	m_depthWidth  = static_cast<LONG>(depthWidth);
	m_depthHeight = static_cast<LONG>(depthHeight);

	m_colorWidth  = static_cast<LONG>(colorWidth);
	m_colorHeight = static_cast<LONG>(colorHeight);

	for (size_t i = 0; i < m_depthFloat.size(); i++) {
		SAFE_DELETE_ARRAY(m_depthFloat[i]);
	}
	m_depthFloat.resize(depthRingBufferSize);
	for (unsigned int i = 0; i<depthRingBufferSize; i++) {
		m_depthFloat[i] = new float[m_depthWidth*m_depthHeight];
	}

	SAFE_DELETE_ARRAY(m_colorRGBX);
	m_colorRGBX = new vec4uc[m_colorWidth*m_colorHeight];

}

RGBDSensor::~RGBDSensor()
{
	// done with pixel data
	SAFE_DELETE_ARRAY(m_colorRGBX);

	for (size_t i = 0; i < m_depthFloat.size(); i++) {
		SAFE_DELETE_ARRAY(m_depthFloat[i]);
	}
	m_depthFloat.clear();

	reset();
}


//! Get the intrinsic camera matrix of the depth sensor
const mat4f& RGBDSensor::getDepthIntrinsics() const
{
	return m_depthIntrinsics;
}

const mat4f& RGBDSensor::getDepthIntrinsicsInv() const
{
	return m_depthIntrinsicsInv;
}

//! Get the intrinsic camera matrix of the color sensor
const mat4f& RGBDSensor::getColorIntrinsics() const
{
	return m_colorIntrinsics;
}

const mat4f& RGBDSensor::getColorIntrinsicsInv() const
{
	return m_colorIntrinsicsInv;
}

const mat4f& RGBDSensor::getDepthExtrinsics() const
{
	return m_depthExtrinsics;
}

const mat4f& RGBDSensor::getDepthExtrinsicsInv() const
{
	return m_depthExtrinsicsInv;
}

const mat4f& RGBDSensor::getColorExtrinsics() const
{
	return m_colorExtrinsics;
}

const mat4f& RGBDSensor::getColorExtrinsicsInv() const
{
	return m_colorExtrinsicsInv;
}


void RGBDSensor::initializeDepthExtrinsics(const mat4f& m) {
	m_depthExtrinsics = m;
	m_depthExtrinsicsInv = m.getInverse();
}
void RGBDSensor::initializeColorExtrinsics(const mat4f& m) {
	m_colorExtrinsics = m;
	m_colorExtrinsicsInv = m.getInverse();
}

void RGBDSensor::initializeDepthIntrinsics(float fovX, float fovY, float centerX, float centerY)
{
	m_depthIntrinsics = mat4f(	fovX, 0.0f, centerX, 0.0f,
		0.0f, fovY, centerY, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();
	std::cout << m_depthIntrinsics << std::endl;
}

void RGBDSensor::initializeColorIntrinsics(float fovX, float fovY, float centerX, float centerY)
{
	m_colorIntrinsics = mat4f(	fovX, 0.0f, centerX, 0.0f,
		0.0f, fovY, centerY, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();
}

float* RGBDSensor::getDepthFloat() {
	return m_depthFloat[m_currentRingBufIdx];
}

const float* RGBDSensor::getDepthFloat() const {
	return m_depthFloat[m_currentRingBufIdx];
}


void RGBDSensor::incrementRingbufIdx()
{
	m_currentRingBufIdx = (m_currentRingBufIdx+1)%m_depthFloat.size();
}

//! gets the pointer to color array
vec4uc* RGBDSensor::getColorRGBX() {  
	return m_colorRGBX;
}

const vec4uc* RGBDSensor::getColorRGBX() const {
	return m_colorRGBX;
}


unsigned int RGBDSensor::getColorWidth()  const {
	return m_colorWidth;
}

unsigned int RGBDSensor::getColorHeight() const {
	return m_colorHeight;
}

unsigned int RGBDSensor::getDepthWidth()  const {
	return m_depthWidth;
}

unsigned int RGBDSensor::getDepthHeight() const {
	return m_depthHeight;
}

void RGBDSensor::reset()
{
	if (m_recordedDepthData.size()) {
		for (auto& d : m_recordedDepthData) {
			delete[] d;
		}
		m_recordedDepthData.clear();
	}
	if (m_recordedColorData.size()) {
		for (auto& c : m_recordedColorData) {
			delete[] c;
		}
		m_recordedColorData.clear();
	}
	m_recordedTrajectory.clear();
	m_recordedPoints.clear();
}

void RGBDSensor::savePointCloud( const std::string& filename, const mat4f& transform /*= mat4f::identity()*/ ) const
{
	//DepthImage d(getDepthHeight(), getDepthWidth(), getDepthFloat());
	//ColorImageRGB c(d);
	//FreeImageWrapper::saveImage("test.png", c, true);

	PointCloudf pc;
	for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
		unsigned int x = i % getDepthWidth();
		unsigned int y = i / getDepthWidth();

		float d = getDepthFloat()[i];
		if (d != 0.0f && d != -std::numeric_limits<float>::infinity()) {
			vec3f p = getDepthIntrinsicsInv()*vec3f((float)x*d, (float)y*d, d);

			//TODO check why our R and B is flipped
			vec4f c = vec4f(getColorRGBX()[i].z, getColorRGBX()[i].y, getColorRGBX()[i].x, getColorRGBX()[i].w);
			c /= 255.0f;

			pc.m_points.push_back(p);
			pc.m_colors.push_back(c);
		}
	}

	PointCloudIOf::saveToFile(filename, pc);
}

void RGBDSensor::recordFrame()
{
	m_recordedDepthData.push_back(m_depthFloat[m_currentRingBufIdx]);
	m_recordedColorData.push_back(m_colorRGBX);

	m_depthFloat[m_currentRingBufIdx] = new float[getDepthWidth()*getDepthHeight()];
	m_colorRGBX = new vec4uc[getColorWidth()*getColorWidth()];
}

void RGBDSensor::recordTrajectory(const mat4f& transform)
{
	m_recordedTrajectory.push_back(transform);
}

void RGBDSensor::saveRecordedFramesToFile( const std::string& filename )
{
	if (m_recordedDepthData.size() == 0 || m_recordedColorData.size() == 0) return;

	CalibratedSensorData cs;
	cs.m_DepthImageWidth = getDepthWidth();
	cs.m_DepthImageHeight = getDepthHeight();
	cs.m_ColorImageWidth = getColorWidth();
	cs.m_ColorImageHeight = getColorHeight();
	cs.m_DepthNumFrames = (unsigned int)m_recordedDepthData.size();
	cs.m_ColorNumFrames = (unsigned int)m_recordedColorData.size();

	cs.m_CalibrationDepth.m_Intrinsic = getDepthIntrinsics();
	cs.m_CalibrationDepth.m_Extrinsic = getDepthExtrinsics();
	cs.m_CalibrationDepth.m_IntrinsicInverse = cs.m_CalibrationDepth.m_Intrinsic.getInverse();
	cs.m_CalibrationDepth.m_ExtrinsicInverse = cs.m_CalibrationDepth.m_Extrinsic.getInverse();

	cs.m_CalibrationColor.m_Intrinsic = getColorIntrinsics();
	cs.m_CalibrationColor.m_Extrinsic = getColorExtrinsics();
	cs.m_CalibrationColor.m_IntrinsicInverse = cs.m_CalibrationColor.m_Intrinsic.getInverse();
	cs.m_CalibrationColor.m_ExtrinsicInverse = cs.m_CalibrationColor.m_Extrinsic.getInverse();

	cs.m_DepthImages.resize(cs.m_DepthNumFrames);
	cs.m_ColorImages.resize(cs.m_ColorNumFrames);
	unsigned int dFrame = 0;
	for (auto& a : m_recordedDepthData) {
		cs.m_DepthImages[dFrame] = a;
		dFrame++;
	}
	unsigned int cFrame = 0;
	for (auto& a : m_recordedColorData) {
		cs.m_ColorImages[cFrame] = a;
		cFrame++;
	}

	cs.m_trajectory = m_recordedTrajectory;

	std::cout << cs << std::endl;
	std::cout << "dumping recorded frames... ";

	std::string actualFilename = filename;
	while (util::fileExists(actualFilename)) {
		std::string path = util::directoryFromPath(actualFilename);
		std::string curr = util::fileNameFromPath(actualFilename);
		std::string ext = util::getFileExtension(curr);
		curr = util::removeExtensions(curr);
		std::string base = util::getBaseBeforeNumericSuffix(curr);
		unsigned int num = util::getNumericSuffix(curr);
		if (num == (unsigned int)-1) {
			num = 0;
		}
		actualFilename = path + base + std::to_string(num+1) + "." + ext;
	}
	BinaryDataStreamFile outStream(actualFilename, true);
	//BinaryDataStreamZLibFile outStream(filename, true);
	outStream << cs;
	std::cout << "done" << std::endl;

	m_recordedDepthData.clear();
	m_recordedColorData.clear();	//destructor of cs frees all allocated data
	//m_recordedTrajectory.clear();
}



ml::vec3f RGBDSensor::depthToSkeleton(unsigned int ux, unsigned int uy) const
{
	return depthToSkeleton(ux, uy, m_depthFloat[m_currentRingBufIdx][uy*getDepthWidth()+ux]);
}

ml::vec3f RGBDSensor::depthToSkeleton(unsigned int ux, unsigned int uy, float depth) const
{
	if (depth == -std::numeric_limits<float>::infinity()) return vec3f(depth);

	float x = ((float)ux-m_depthIntrinsics(0,2)) / m_depthIntrinsics(0,0);
	float y = ((float)uy-m_depthIntrinsics(1,2)) / m_depthIntrinsics(1,1);

	return vec3f(depth*x, depth*y, depth);
}

ml::vec3f RGBDSensor::getNormal(unsigned int x, unsigned int y) const
{
	vec3f ret(-std::numeric_limits<float>::infinity());
	if (x > 0 && y > 0 && x < getDepthWidth() - 1 && y < getDepthHeight() - 1) {
		vec3f cc = depthToSkeleton(x,y);
		vec3f pc = depthToSkeleton(x+1,y+0);
		vec3f cp = depthToSkeleton(x+0,y+1);
		vec3f mc = depthToSkeleton(x-1,y+0);
		vec3f cm = depthToSkeleton(x+0,y-1);

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

void RGBDSensor::computePointCurrentPointCloud(PointCloudf& pc, const mat4f& transform /*= mat4f::identity()*/) const
{
	//if (!(getColorWidth() == getDepthWidth() && getColorHeight() == getDepthHeight()))	throw MLIB_EXCEPTION("invalid dimensions");
	const float scaleWidth = (float)getColorWidth() / (float)getDepthWidth();
	const float scaleHeight = (float)getColorHeight() / (float)getDepthHeight();

	for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
		unsigned int x = i % getDepthWidth();
		unsigned int y = i / getDepthWidth();
		vec3f p = depthToSkeleton(x,y);
		if (p.x != -std::numeric_limits<float>::infinity() && p.x != 0.0f)	{

			//vec3f n = getNormal(x,y);
			//if (n.x != -FLT_MAX) {
				pc.m_points.push_back(p);
				//pc.m_normals.push_back(-n);

				unsigned int cx = math::round(scaleWidth * x);
				unsigned int cy = math::round(scaleHeight * y);
				vec4uc c = m_colorRGBX[cy * getColorWidth() + cx];
				pc.m_colors.push_back(vec4f(c.x/255.0f, c.y/255.0f, c.z/255.0f, 1.0f));	//there's a swap... dunno why really
			//}
		}
	}
	for (auto& p : pc.m_points) {
		p = transform * p;
	}
	mat4f invTranspose = transform.getInverse().getTranspose();
	for (auto& n : pc.m_normals) {
		n = invTranspose * n;
		n.normalize();
	}
}

void RGBDSensor::recordPointCloud(const mat4f& transform /*= mat4f::identity()*/)
{
	m_recordedPoints.push_back(PointCloudf());
	computePointCurrentPointCloud(m_recordedPoints.back(), transform);
}

void RGBDSensor::saveRecordedPointCloud(const std::string& filename)
{
	std::cout << "recorded " << m_recordedPoints.size() << " frames" << std::endl;
	PointCloudf pc;
	for (const auto& p : m_recordedPoints ) {
		pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
		pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
		pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
	}
	PointCloudIOf::saveToFile(filename, pc);
	m_recordedPoints.clear();
}

void RGBDSensor::saveRecordedPointCloud(const std::string& filename, const std::vector<int>& validImages, const std::vector<mat4f>& trajectory)
{
	MLIB_ASSERT(m_recordedPoints.size() == validImages.size() &&
		m_recordedPoints.size() <= trajectory.size());
	std::cout << "recorded " << m_recordedPoints.size() << " frames" << std::endl;
	// apply transforms
	PointCloudf pc;
	for (unsigned int i = 0; i < m_recordedPoints.size(); i++) {
		PointCloudf& p = m_recordedPoints[i];
		mat4f invTranspose = trajectory[i].getInverse().getTranspose();
		for (auto& pt : p.m_points)
			pt = trajectory[i] * pt;
		for (auto& n : p.m_normals) {
			n = invTranspose * n;
			n.normalize();
		}

		pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
		pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
		pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
	}

	PointCloudIOf::saveToFile(filename, pc);
	m_recordedPoints.clear();
}
