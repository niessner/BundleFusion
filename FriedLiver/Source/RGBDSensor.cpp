
#include "stdafx.h"

#include "RGBDSensor.h"

#include "GlobalAppState.h"

//namespace stb {
//#define STB_IMAGE_IMPLEMENTATION
//#include "SensorData/stb_image.h"
//#undef STB_IMAGE_IMPLEMENTATION
//
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include "SensorData/stb_image_write.h"
//#undef STB_IMAGE_WRITE_IMPLEMENTATION
//}

#include <limits>

RGBDSensor::RGBDSensor()
{
	m_depthWidth  = 0;
	m_depthHeight = 0;

	m_colorWidth  = 0;
	m_colorHeight = 0;

	m_colorRGBX = NULL;

	m_currentRingBufIdx = 0;

	m_bIsReceivingFrames = true;

	m_recordDataWidth = 0;
	m_recordDataHeight = 0;

	m_recordedData = NULL;
	m_recordedDataCache = NULL;
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

	m_recordDataWidth = GlobalAppState::get().s_recordDataWidth;
	m_recordDataHeight = GlobalAppState::get().s_recordDataHeight;

	m_bUseModernSensFilesForRecording = GlobalAppState::get().s_recordCompression;

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
	if (m == mat4f::identity()) {
		GlobalAppState::get().s_bUseCameraCalibration = false;
		m_depthExtrinsics = mat4f::identity();
		m_depthExtrinsicsInv = mat4f::identity();
	}
	else {
		m_depthExtrinsics = m;
		m_depthExtrinsicsInv = m.getInverse();
	}
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
	std::cout << "depth intrinsics" << std::endl;
	std::cout << m_depthIntrinsics << std::endl;

	m_recordIntrinsics = m_depthIntrinsics;
	m_recordIntrinsics._m00 *= (float)m_recordDataWidth / (float)getDepthWidth();
	m_recordIntrinsics._m11 *= (float)m_recordDataHeight / (float)getDepthHeight();
	m_recordIntrinsics._m02 *= (float)(m_recordDataWidth -1)/ (float)(getDepthWidth()-1);
	m_recordIntrinsics._m12 *= (float)(m_recordDataHeight-1) / (float)(getDepthHeight()-1);
	m_recordIntrinsicsInv = m_recordIntrinsics.getInverse();
}

void RGBDSensor::initializeColorIntrinsics(float fovX, float fovY, float centerX, float centerY)
{
	m_colorIntrinsics = mat4f(	fovX, 0.0f, centerX, 0.0f,
		0.0f, fovY, centerY, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();

	std::cout << "color intrinsics" << std::endl;
	std::cout << m_colorIntrinsics << std::endl;
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
	m_recordedPoints.clear();

	SAFE_DELETE(m_recordedDataCache);
	if (m_recordedData) {
		m_recordedData->free();
		SAFE_DELETE(m_recordedData);
	}
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
	if (m_bUseModernSensFilesForRecording) {
		if (!m_recordedData) {
			m_recordedData = new ml::SensorData;
			m_recordedData->initDefault(
				getColorWidth(),
				getColorHeight(),
				getDepthWidth(),
				getDepthHeight(),
				ml::SensorData::CalibrationData(getColorIntrinsics(), getColorExtrinsics()),
				ml::SensorData::CalibrationData(getDepthIntrinsics(), getDepthExtrinsics()),
				ml::SensorData::TYPE_JPEG,
				ml::SensorData::TYPE_ZLIB_USHORT,
				1000.0f,
				getSensorName()
				);
			//m_recordedData->m_sensorName = getSensorName();
			//m_recordedData->m_calibrationColor.m_intrinsic = getColorIntrinsics();
			//m_recordedData->m_calibrationColor.m_extrinsic = getColorExtrinsics();
			//m_recordedData->m_calibrationDepth.m_intrinsic = getDepthIntrinsics();
			//m_recordedData->m_calibrationDepth.m_extrinsic = getDepthExtrinsics();
			//m_recordedData->m_colorWidth = getColorWidth();
			//m_recordedData->m_colorHeight = getColorHeight();
			//m_recordedData->m_depthWidth = getDepthWidth();
			//m_recordedData->m_depthHeight = getDepthHeight();
			//m_recordedData->m_depthShift = 1000.0f;
		}
		if (!m_recordedDataCache) {
			const unsigned int cacheSize = 1000;
			m_recordedDataCache = new ml::SensorData::RGBDFrameCacheWrite(m_recordedData, cacheSize);
		}

		vec3uc* color = (vec3uc*)std::malloc(sizeof(vec3uc)*getColorWidth()*getColorHeight());
		unsigned short* depth = (unsigned short*)std::malloc(sizeof(unsigned short)*getDepthWidth()*getDepthHeight());
		for (unsigned int i = 0; i < getColorWidth()*getColorHeight(); i++) {
			const auto* c = getColorRGBX();
			color[i] = vec3uc(c[i].x, c[i].y, c[i].z);
		}
		for (unsigned int i = 0; i < getDepthWidth()*getDepthHeight(); i++) {
			const auto* d = getDepthFloat();
			depth[i] = (unsigned short)ml::math::round((m_recordedData->m_depthShift * d[i]));
		}

		//m_recordedData->m_frames.push_back(SensorData::RGBDFrame(color, getColorWidth(), getColorHeight(), depth, getDepthWidth(), getDepthHeight()));
		//std::free(color); 
		//std::free(depth);
		m_recordedDataCache->writeNextAndFree(color, depth);
	}
	else {
		// resample if m_recordDataWidt/height != 0
		if ((m_recordDataWidth == 0 && m_recordDataHeight == 0) || (getDepthWidth() == m_recordDataWidth && getDepthHeight() == m_recordDataHeight)) {
			m_recordedDepthData.push_back(m_depthFloat[m_currentRingBufIdx]);
			m_depthFloat[m_currentRingBufIdx] = new float[getDepthWidth()*getDepthHeight()];
		}
		else {
			m_recordedDepthData.push_back(new float[m_recordDataWidth*m_recordDataHeight]);
			float* depth = m_recordedDepthData.back();
			float scaleX = (float)(getDepthWidth() - 1) / (m_recordDataWidth - 1);
			float scaleY = (float)(getDepthHeight() - 1) / (m_recordDataHeight - 1);
			for (unsigned int y = 0; y < m_recordDataHeight; y++) {
				for (unsigned int x = 0; x < m_recordDataWidth; x++) {
					unsigned int _x = math::round(scaleX*x);	_x = math::clamp(_x, 0u, getDepthWidth());
					unsigned int _y = math::round(scaleY*y);	_y = math::clamp(_y, 0u, getDepthHeight());
					depth[y*m_recordDataWidth + x] = m_depthFloat[m_currentRingBufIdx][_y*getDepthWidth() + _x];
				}
			}
		}

		if ((m_recordDataWidth == 0 && m_recordDataHeight == 0) || (getColorWidth() == m_recordDataWidth && getColorHeight() == m_recordDataHeight)) {
			m_recordedColorData.push_back(m_colorRGBX);
			m_colorRGBX = new vec4uc[getColorWidth()*getColorHeight()];
		}
		else {
			m_recordedColorData.push_back(new vec4uc[m_recordDataWidth*m_recordDataHeight]);
			vec4uc* color = m_recordedColorData.back();
			float scaleX = (float)(getColorWidth() - 1) / (m_recordDataWidth - 1);
			float scaleY = (float)(getColorHeight() - 1) / (m_recordDataHeight - 1);
			for (unsigned int y = 0; y < m_recordDataHeight; y++) {
				for (unsigned int x = 0; x < m_recordDataWidth; x++) {
					unsigned int _x = math::round(scaleX*x);	_x = math::clamp(_x, 0u, getColorWidth());
					unsigned int _y = math::round(scaleY*y);	_y = math::clamp(_y, 0u, getColorHeight());
					color[y*m_recordDataWidth + x] = m_colorRGBX[_y*getColorWidth() + _x];
				}
			}
		}
	}
}

void RGBDSensor::saveRecordedFramesToFile(const std::string& filename, const std::vector<mat4f>& trajectory, bool overwriteExistingFile /*= false*/)
{
	if (m_bUseModernSensFilesForRecording) {

		if (!m_recordedDataCache || !m_recordedData) return;
		if (trajectory.size() == 0) return;

		SAFE_DELETE(m_recordedDataCache);		//forces cache to finish!
		
		unsigned int numFrames = (unsigned int)trajectory.size();
		numFrames = std::min(numFrames, (unsigned int)m_recordedData->m_frames.size());

		if (trajectory.size() > numFrames) throw MLIB_EXCEPTION("something went wrong; found more transforms than frames");

		//setting the trajectory first
		for (size_t i = 0; i < numFrames; i++) {
			m_recordedData->m_frames[i].setCameraToWorld(trajectory[i]);
		}
		//free frames without a trajectory
		if (numFrames < m_recordedData->m_frames.size()) {
			for (size_t i = numFrames; i < m_recordedData->m_frames.size(); i++) {
				m_recordedData->m_frames[i].free();
			}
			m_recordedData->m_frames.resize(numFrames);
		}

		std::cout << *m_recordedData << std::endl;
		std::cout << "dumping recorded frames ... ";

		std::string actualFilename = filename;
		if (!overwriteExistingFile) {
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
				actualFilename = path + base + std::to_string(num + 1) + "." + ext;
			}
		}

		m_recordedData->saveToFile(actualFilename);
		std::cout << "DONE!" << std::endl;


		m_recordedData->free();
		SAFE_DELETE(m_recordedData);

	}
	else {
		unsigned int numFrames = (unsigned int)trajectory.size();
		numFrames = std::min(numFrames, (unsigned int)m_recordedDepthData.size());
		numFrames = std::min(numFrames, (unsigned int)m_recordedColorData.size());

		if (numFrames == 0) return;

		CalibratedSensorData cs;
		cs.m_DepthImageWidth = m_recordDataWidth;
		cs.m_DepthImageHeight = m_recordDataHeight;
		cs.m_ColorImageWidth = m_recordDataWidth;
		cs.m_ColorImageHeight = m_recordDataHeight;
		cs.m_DepthNumFrames = numFrames;
		cs.m_ColorNumFrames = numFrames;

		cs.m_CalibrationDepth.m_Intrinsic = m_recordIntrinsics;
		cs.m_CalibrationDepth.m_Extrinsic = getDepthExtrinsics();
		cs.m_CalibrationDepth.m_IntrinsicInverse = cs.m_CalibrationDepth.m_Intrinsic.getInverse();
		cs.m_CalibrationDepth.m_ExtrinsicInverse = cs.m_CalibrationDepth.m_Extrinsic.getInverse();

		cs.m_CalibrationColor.m_Intrinsic = m_recordIntrinsics;
		cs.m_CalibrationColor.m_Extrinsic = getColorExtrinsics();
		cs.m_CalibrationColor.m_IntrinsicInverse = cs.m_CalibrationColor.m_Intrinsic.getInverse();
		cs.m_CalibrationColor.m_ExtrinsicInverse = cs.m_CalibrationColor.m_Extrinsic.getInverse();

		cs.m_DepthImages.resize(numFrames);
		cs.m_ColorImages.resize(numFrames);
		cs.m_trajectory.resize(numFrames);

		unsigned int dFrame = 0;
		for (auto& a : m_recordedDepthData) {
			if (dFrame >= numFrames) break;
			cs.m_DepthImages[dFrame] = a;
			dFrame++;
		}
		unsigned int cFrame = 0;
		for (auto& a : m_recordedColorData) {
			if (cFrame >= numFrames) break;
			cs.m_ColorImages[cFrame] = a;
			cFrame++;
		}

		for (unsigned int i = 0; i < numFrames; i++) {
			cs.m_trajectory[i] = trajectory[i];
		}

		std::cout << cs << std::endl;

		std::string folder = util::directoryFromPath(filename);
		if (!util::directoryExists(folder)) {
			util::makeDirectory(folder);
		}

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
			actualFilename = path + base + std::to_string(num + 1) + "." + ext;
		}



		std::cout << "dumping recorded frames to " << actualFilename << " ...";


		BinaryDataStreamFile outStream(actualFilename, true);
		//BinaryDataStreamZLibFile outStream(filename, true);
		outStream << cs;
		std::cout << "done" << std::endl;

		m_recordedDepthData.clear();
		m_recordedColorData.clear();	//destructor of cs frees all allocated data
		//m_recordedTrajectory.clear();
	}
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
	const float scaleWidth = (float)(getColorWidth() - 1) / (float)(getDepthWidth() - 1);
	const float scaleHeight = (float)(getColorHeight() - 1) / (float)(getDepthHeight() - 1);

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
	MLIB_ASSERT(m_recordedPoints.size() <= validImages.size() &&
		m_recordedPoints.size() <= trajectory.size());
	//!!!
	std::ofstream s("invalidImages.txt");
	for (unsigned int i = 0; i < m_recordedPoints.size(); i++) {
		if (validImages[i] != 1)
			s << "\timage " << i << " = " << validImages[i] << std::endl;
	}
	s.close();
	//!!!

	// apply transforms
	PointCloudf pc;
	for (unsigned int i = 0; i < m_recordedPoints.size(); i++) {
		if (validImages[i] != 0) {
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
	}

	PointCloudIOf::saveToFile(filename, pc);
	std::cout << "recorded " << m_recordedPoints.size() << " frames" << std::endl;
	m_recordedPoints.clear();
}

void RGBDSensor::saveRecordedPointCloudDEBUG(const std::string& filename, const std::vector<int>& validImages, const std::vector<mat4f>& trajectory, unsigned int submapSize)
{
	MLIB_ASSERT(m_recordedPoints.size() <= validImages.size() &&
		m_recordedPoints.size() <= trajectory.size());
	//!!!
	std::ofstream s("invalidImages.txt");
	for (unsigned int i = 0; i < m_recordedPoints.size(); i++) {
		if (validImages[i] != 1)
			s << "\timage " << i << " = " << validImages[i] << std::endl;
	}
	s.close();
	//!!!

	// apply transforms
	PointCloudf pc;
	for (unsigned int i = 0; i < m_recordedPoints.size(); i++) {
		if (validImages[i] != 0) {
			if (trajectory[i*submapSize][0] == -std::numeric_limits<float>::infinity()) {
				std::cout << "ERROR complete trajectory and valid images do not match! (" << i*submapSize << ")" << std::endl;
				getchar();
			}

			PointCloudf& p = m_recordedPoints[i];
			mat4f invTranspose = trajectory[i*submapSize].getInverse().getTranspose();
			for (auto& pt : p.m_points)
				pt = trajectory[i*submapSize] * pt;
			for (auto& n : p.m_normals) {
				n = invTranspose * n;
				n.normalize();
			}

			pc.m_points.insert(pc.m_points.end(), p.m_points.begin(), p.m_points.end());
			pc.m_colors.insert(pc.m_colors.end(), p.m_colors.begin(), p.m_colors.end());
			pc.m_normals.insert(pc.m_normals.end(), p.m_normals.begin(), p.m_normals.end());
		}
	}

	PointCloudIOf::saveToFile(filename, pc);
	std::cout << "recorded " << m_recordedPoints.size() << " frames" << std::endl;
	m_recordedPoints.clear();
}