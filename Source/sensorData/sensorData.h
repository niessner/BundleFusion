

#ifndef _SENSOR_FILE_H_
#define _SENSOR_FILE_H_

///////////////////////////////////////////////////////////////////////////////////
////////////////// ADJUST THESE DEFINES TO YOUR NEEDS /////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

#define _HAS_MLIB
#define _USE_UPLINK_COMPRESSION

///////////////////////////////////////////////////////////////////////////////////
/////////////////// DON'T TOUCH THE FILE BLOW THIS LINE ///////////////////////////
///////////////////////////////////////////////////////////////////////////////////

namespace stb {
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//#undef STB_IMAGE_IMPLEMENTATION

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
//#undef STB_IMAGE_WRITE_IMPLEMENTATION
}

#ifdef _USE_UPLINK_COMPRESSION
	#if _WIN32
	#pragma comment(lib, "gdiplus.lib")
	#pragma comment(lib, "Shlwapi.lib")
	#include <winsock2.h>
	#include <Ws2tcpip.h>
	#include "uplinksimple.h"
	#else 
	#undef _USE_UPLINK_COMPRESSION
	#endif 
#endif
#include <vector>
#include <string>
#include <exception>
#include <fstream>
#include <cassert>
#include <iostream>
#include <atomic>


namespace ml {

#ifndef _HAS_MLIB

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

	class MLibException : public std::exception {
	public:
		MLibException(const std::string& what) : std::exception() {
			m_msg = what;
		}
		MLibException(const char* what) : std::exception() {
			m_msg = std::string(what);
		}
		const char* what() const NOEXCEPT{
			return m_msg.c_str();
		}
	private:
		std::string m_msg;
	};



#ifndef MLIB_EXCEPTION
#define MLIB_EXCEPTION(s) ml::MLibException(std::string(__FUNCTION__).append(":").append(std::to_string(__LINE__)).append(": ").append(s).c_str())
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)       { if (p) { delete (p);     (p)=nullptr; } }
#endif

#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p);   (p)=nullptr; } }
#endif

#ifndef UINT64
#ifdef WIN32
	typedef unsigned __int64 UINT64;
#else
	typedef uint64_t UINT64;
#endif
#endif

	class vec3uc {
		union
		{
			struct
			{
				unsigned char x, y, z; // standard names for components
			};
			unsigned char array[3];     // array access
		};
	};

	class vec4uc {
		union
		{
			struct
			{
				unsigned char x, y, z, w; // standard names for components
			};
			unsigned char array[4];     // array access
		};
	};

	class mat4f {
	public:
		void setIdentity() {
			matrix[0] = 1.0;	matrix[1] = 0.0f;	matrix[2] = 0.0f; matrix[3] = 0.0f;
			matrix[4] = 0.0f;	matrix[5] = 1.0;	matrix[6] = 0.0f; matrix[7] = 0.0f;
			matrix[8] = 0.0f;	matrix[9] = 0.0f;	matrix[10] = 1.0; matrix[11] = 0.0f;
			matrix[12] = 0.0f;	matrix[13] = 0.0f;	matrix[14] = 0.0f; matrix[15] = 1.0;
		}
		union {
			//! access matrix using a single array
			float matrix[16];
			//! access matrix using a two-dimensional array
			float matrix2[4][4];
			//! access matrix using single elements
			struct {
				float
					_m00, _m01, _m02, _m03,
					_m10, _m11, _m12, _m13,
					_m20, _m21, _m22, _m23,
					_m30, _m31, _m32, _m33;
			};
		};
	};
#endif //_NO_MLIB_


	class SensorData {
	public:
		class CalibrationData {
		public:
			CalibrationData() {
				setIdentity();
			}

			void setIdentity() {
				m_intrinsic.setIdentity();
				m_extrinsic.setIdentity();
			}

			void writeToFile(std::ofstream& out) const {
				out.write((const char*)&m_intrinsic, sizeof(mat4f));
				out.write((const char*)&m_extrinsic, sizeof(mat4f));
			}

			void readFromFile(std::ifstream& in) {
				in.read((char*)&m_intrinsic, sizeof(mat4f));
				in.read((char*)&m_extrinsic, sizeof(mat4f));
			}

			bool operator==(const CalibrationData& other) const {
				for (unsigned int i = 0; i < 16; i++) {
					if (m_intrinsic[i] != other.m_intrinsic[i]) return false;
					if (m_extrinsic[i] != other.m_extrinsic[i]) return false;
				}
				return true;
			}

			bool operator!=(const CalibrationData& other) const {
				return !((*this) == other);
			}

			//! Camera-to-Proj matrix
			mat4f m_intrinsic;

			//! World-to-Camera matrix (accumulated R|t mapping back to the first frame))
			mat4f m_extrinsic;
		};



		class RGBDFrame {
		public:
			enum COMPRESSION_TYPE_COLOR {
				TYPE_RAW = 0,
				TYPE_PNG = 1,
				TYPE_JPEG = 2
			};
			enum COMPRESSION_TYPE_DEPTH {
				TYPE_RAW_USHORT = 0,
				TYPE_ZLIB_USHORT = 1,
				TYPE_OCCI_USHORT = 2
			};

			RGBDFrame() {
				m_colorCompressed = NULL;
				m_depthCompressed = NULL;
				m_colorSizeBytes = 0;
				m_depthSizeBytes = 0;
				m_frameToWorld.setIdentity();
			}
			RGBDFrame(
				const vec3uc* color, unsigned int colorWidth, unsigned int colorHeight,
				const unsigned short*  depth, unsigned int depthWidth, unsigned int depthHeight,
				COMPRESSION_TYPE_COLOR colorType = TYPE_JPEG,
				COMPRESSION_TYPE_DEPTH depthType = TYPE_ZLIB_USHORT)
			{
				m_colorCompressed = NULL;
				m_depthCompressed = NULL;
				m_colorSizeBytes = 0;
				m_depthSizeBytes = 0;
				m_frameToWorld.setIdentity();

				if (color) {
					//Timer t;
					compressColor(color, colorWidth, colorHeight, colorType);
					//std::cout << "compressColor " << t.getElapsedTimeMS() << " [ms] " << std::endl;
				}
				if (depth) {
					//Timer t;
					compressDepth(depth, depthWidth, depthHeight, depthType);
					//std::cout << "compressDepth " << t.getElapsedTimeMS() << " [ms] " << std::endl;
				}
			}

			void free() {
				freeColor();
				freeDepth();
				m_frameToWorld.setIdentity();
			}


			void freeColor() {
				if (m_colorCompressed) std::free(m_colorCompressed);
				m_colorCompressed = NULL;
				m_colorSizeBytes = 0;
				m_colorWidth = m_colorHeight = 0;
			}
			void freeDepth() {
				if (m_depthCompressed) std::free(m_depthCompressed);
				m_depthCompressed = NULL;
				m_depthSizeBytes = 0;
				m_depthWidth = m_depthHeight = 0;
			}


			void compressColor(const vec3uc* color, unsigned int width, unsigned int height, COMPRESSION_TYPE_COLOR type = TYPE_JPEG) {
				//if (m_colorCompressed)	stbi_image_free(m_colorCompressed);
				//int channels = 3;
				//int stride_bytes = width * channels;
				//int len = 0;
				//m_colorCompressed = stbi_write_png_to_mem((unsigned char*)color, stride_bytes, width, height, channels, &len);
				//m_colorSizeBytes = len;
				//if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("compression error");

				//if (m_colorCompressed) std::free(m_colorCompressed);
				//LodePNGColorType colorType = LodePNGColorType::LCT_RGB;
				//unsigned int bitDepth = 8;
				//unsigned error = lodepng_encode_memory(&m_colorCompressed, &m_colorSizeBytes, (const unsigned char*)color, width, height, colorType, bitDepth);
				//if (error != 0 || m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("compression error");

				m_colorCompressionType = type;

				if (type == TYPE_RAW) {
					if (m_colorSizeBytes != width*height) {_USE_UPLINK_COMPRESSION
						freeColor();
						m_colorSizeBytes = width*height*sizeof(vec3uc);
						m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
					}
					std::memcpy(m_colorCompressed, color, m_colorSizeBytes);
				}
				else if (type == TYPE_PNG || type == TYPE_JPEG) {
					freeColor();

#ifdef _USE_UPLINK_COMPRESSION
					uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
					uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
					if (type == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

					uplinksimple::MemoryBlock block;
					float quality = uplinksimple::defaultQuality;
					//float quality = 1.0f;
					uplinksimple::encode_image(codec, (const uint8_t*)color, width*height, format, width, height, block, quality);
					m_colorCompressed = block.Data;
					m_colorSizeBytes = block.Size;
					block.relinquishOwnership();
#else
					throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif

					//jpge::params p = jpge::params();
					//p.m_quality = 95;
					//int channels = 3;
					//int buffSize = std::max(1024u, width*height*channels);
					//void* buff = std::malloc(buffSize);
					//bool r = jpge::compress_image_to_jpeg_file_in_memory(buff, buffSize, width, height, channels, (const jpge::uint8*)color, p);
					//if (r == false)	 std::cout << "error - len: " << buffSize << std::endl;
					//m_colorSizeBytes = buffSize;
					//m_colorCompressed = (unsigned char*)std::malloc(buffSize);
					//std::memcpy(m_colorCompressed, buff, buffSize);
					//std::free(buff);
				}
				else {
					throw MLIB_EXCEPTION("unknown compression type");
				}

				m_colorWidth = width;
				m_colorHeight = height;
			}

			vec3uc* decompressColorAlloc() const {
				if (m_colorCompressionType == TYPE_RAW)	return decompressColorAlloc_raw();
#ifdef _USE_UPLINK_COMPRESSION
				else return decompressColorAlloc_occ();	//this handles all image formats;
#else
				else return decompressColorAlloc_stb();	// this handles all image formats
#endif
				//if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
				//LodePNGColorType colorType = LodePNGColorType::LCT_RGB;
				//unsigned int bitDepth = 8;
				//vec3uc* res = NULL;
				//unsigned int width, height;
				//unsigned error = lodepng_decode_memory((unsigned char**)&res, &width, &height, m_colorCompressed, m_colorSizeBytes, colorType, bitDepth);
				//if (error != 0 || res == NULL) throw MLIB_EXCEPTION("decompression error");
				//return res;
			}

			vec3uc* decompressColorAlloc_stb() const {	//can handle PNG, JPEG etc.
				if (m_colorCompressionType != TYPE_JPEG && m_colorCompressionType != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
				int channels = 3;
				int width, height;
				unsigned char* raw = stb::stbi_load_from_memory(m_colorCompressed, (int)m_colorSizeBytes, &width, &height, NULL, channels);
				return (vec3uc*)raw;
			}

			vec3uc* decompressColorAlloc_occ() const {
#ifdef _USE_UPLINK_COMPRESSION
				if (m_colorCompressionType != TYPE_JPEG && m_colorCompressionType != TYPE_PNG) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("decompression error");
				uplinksimple::graphics_PixelFormat format = uplinksimple::graphics_PixelFormat_RGB;
				uplinksimple::graphics_ImageCodec codec = uplinksimple::graphics_ImageCodec_JPEG;
				if (m_colorCompressionType == TYPE_PNG) codec = uplinksimple::graphics_ImageCodec_PNG;

				uplinksimple::MemoryBlock block;
				float quality = uplinksimple::defaultQuality;
				size_t width = 0;
				size_t height = 0;
				uplinksimple::decode_image(codec, (const uint8_t*)m_colorCompressed, m_colorSizeBytes, format, width, height, block);
				vec3uc* res = (vec3uc*)block.Data;
				block.relinquishOwnership();
				return res;
#else
				throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
				return NULL;
#endif
			}

			vec3uc* decompressColorAlloc_raw() const {
				if (m_colorCompressionType != TYPE_RAW) throw MLIB_EXCEPTION("invliad type");
				if (m_colorCompressed == NULL || m_colorSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
				vec3uc* res = (vec3uc*)std::malloc(m_colorSizeBytes);
				memcpy(res, m_colorCompressed, m_colorSizeBytes);
				return res;
			}

			void compressDepth(const unsigned short* depth, unsigned int width, unsigned int height, COMPRESSION_TYPE_DEPTH type = TYPE_ZLIB_USHORT) {
				freeDepth();

				m_depthCompressionType = type;

				if (type == TYPE_RAW_USHORT) {
					if (m_depthSizeBytes != width*height) {
						freeDepth();
						m_depthSizeBytes = width*height*sizeof(unsigned short);
						m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
					}
					std::memcpy(m_depthCompressed, depth, m_depthSizeBytes);
				}
				else if (type == TYPE_ZLIB_USHORT) {
					freeDepth();

					int out_len = 0;
					int quality = 8;
					int n = 2;
					unsigned char* tmpBuff = (unsigned char *)std::malloc((width*n + 1) * height);
					std::memcpy(tmpBuff, depth, width*height*sizeof(unsigned short));
					m_depthCompressed = stb::stbi_zlib_compress(tmpBuff, width*height*sizeof(unsigned short), &out_len, quality);
					std::free(tmpBuff);
					m_depthSizeBytes = out_len;
				} 
				else if (type == TYPE_OCCI_USHORT) {
					freeDepth();
#ifdef _USE_UPLINK_COMPRESSION
					int out_len = 0;
					int n = 2;
					unsigned int tmpBuffSize = (width*n + 1) * height;
					unsigned char* tmpBuff = (unsigned char *)std::malloc(tmpBuffSize);
					out_len = uplinksimple::encode(depth, width*height, tmpBuff, tmpBuffSize);
					m_depthSizeBytes = out_len;
					m_depthCompressed = (unsigned char*)std::malloc(out_len);
					std::memcpy(m_depthCompressed, tmpBuff, out_len);
					std::free(tmpBuff);
#else
					throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
#endif
				}
				else {
					throw MLIB_EXCEPTION("unknown compression type");
				}

				m_depthWidth = width;
				m_depthHeight = height;
			}

			unsigned short* decompressDepthAlloc() const {
				if (m_depthCompressionType == TYPE_RAW_USHORT)	return decompressDepthAlloc_raw();
				else if (m_depthCompressionType == TYPE_ZLIB_USHORT) return decompressDepthAlloc_stb();
				else if (m_depthCompressionType == TYPE_OCCI_USHORT) return decompressDepthAlloc_occ();
				else {
					throw MLIB_EXCEPTION("invalid type");
					return NULL;
				}
			}

			unsigned short* decompressDepthAlloc_stb() const {
				if (m_depthCompressionType != TYPE_ZLIB_USHORT) throw MLIB_EXCEPTION("invliad type");
				unsigned short* res;
				int len;
				res = (unsigned short*)stb::stbi_zlib_decode_malloc((const char*)m_depthCompressed, (int)m_depthSizeBytes, &len);
				return res;
			}

			unsigned int short* decompressDepthAlloc_occ() const {
#ifdef _USE_UPLINK_COMPRESSION
				if (m_depthCompressionType != TYPE_OCCI_USHORT) throw MLIB_EXCEPTION("invliad type");
				unsigned short* res = (unsigned short*)std::malloc(m_depthWidth*m_depthHeight * 2);
				uplinksimple::decode(m_depthCompressed, (unsigned int)m_depthSizeBytes, m_depthWidth*m_depthHeight, res);
				//uplinksimple::shift2depth(res, m_depthWidth*m_depthHeight); //this is a stupid idea i think
				return res;
#else
				throw MLIB_EXCEPTION("need UPLINK_COMPRESSION");
				return NULL;
#endif
			}

			unsigned short* decompressDepthAlloc_raw() const {
				if (m_depthCompressed == NULL || m_depthSizeBytes == 0) throw MLIB_EXCEPTION("invalid data");
				unsigned short* res = (unsigned short*)std::malloc(m_depthSizeBytes);
				memcpy(res, m_depthCompressed, m_depthSizeBytes);
				return res;
			}

			unsigned char* getColorCompressed() const {
				return m_colorCompressed;
			}
			unsigned char* getDepthCompressed() const {
				return m_depthCompressed;
			}
			size_t getColorSizeBytes() const {
				return m_colorSizeBytes;
			}
			size_t getDepthSizeBytes() const {
				return m_depthSizeBytes;
			}

			void writeToFile(std::ofstream& out) const {
				out.write((const char*)&m_colorCompressionType, sizeof(COMPRESSION_TYPE_COLOR));
				out.write((const char*)&m_depthCompressionType, sizeof(COMPRESSION_TYPE_DEPTH));
				out.write((const char*)&m_frameToWorld, sizeof(mat4f));
				out.write((const char*)&m_colorWidth, sizeof(unsigned int));
				out.write((const char*)&m_colorHeight, sizeof(unsigned int));
				out.write((const char*)&m_depthWidth, sizeof(unsigned int));
				out.write((const char*)&m_depthHeight, sizeof(unsigned int));
				out.write((const char*)&m_colorSizeBytes, sizeof(UINT64));
				out.write((const char*)&m_depthSizeBytes, sizeof(UINT64));
				out.write((const char*)m_colorCompressed, m_colorSizeBytes);
				out.write((const char*)m_depthCompressed, m_depthSizeBytes);
			}

			void readFromFile(std::ifstream& in) {
				free();
				in.read((char*)&m_colorCompressionType, sizeof(COMPRESSION_TYPE_COLOR));
				in.read((char*)&m_depthCompressionType, sizeof(COMPRESSION_TYPE_DEPTH));
				in.read((char*)&m_frameToWorld, sizeof(mat4f));
				in.read((char*)&m_colorWidth, sizeof(unsigned int));
				in.read((char*)&m_colorHeight, sizeof(unsigned int));
				in.read((char*)&m_depthWidth, sizeof(unsigned int));
				in.read((char*)&m_depthHeight, sizeof(unsigned int));
				in.read((char*)&m_colorSizeBytes, sizeof(UINT64));
				in.read((char*)&m_depthSizeBytes, sizeof(UINT64));
				m_colorCompressed = (unsigned char*)std::malloc(m_colorSizeBytes);
				in.read((char*)m_colorCompressed, m_colorSizeBytes);
				m_depthCompressed = (unsigned char*)std::malloc(m_depthSizeBytes);
				in.read((char*)m_depthCompressed, m_depthSizeBytes);
			}

			bool operator==(const RGBDFrame& other) const {
				if (m_colorCompressionType != other.m_colorCompressionType)	return false;
				if (m_depthCompressionType != other.m_depthCompressionType) return false;
				if (m_colorSizeBytes != other.m_colorSizeBytes) return false;
				if (m_depthSizeBytes != other.m_depthSizeBytes) return false;
				if (m_colorWidth != other.m_colorWidth) return false;
				if (m_colorHeight != other.m_colorHeight) return false;
				if (m_depthWidth != other.m_depthWidth) return false;
				if (m_depthHeight != other.m_depthHeight) return false;
				for (unsigned int i = 0; i < 16; i++) {
					if (m_frameToWorld[i] != other.m_frameToWorld[i]) return false;
				}
				for (UINT64 i = 0; i < m_colorSizeBytes; i++) {
					if (m_colorCompressed[i] != other.m_colorCompressed[i]) return false;
				}
				for (UINT64 i = 0; i < m_depthSizeBytes; i++) {
					if (m_depthCompressed[i] != other.m_depthCompressed[i]) return false;
				}
				return true;
			}

			bool operator!=(const RGBDFrame& other) const {
				return !((*this) == other);
			}

			COMPRESSION_TYPE_COLOR m_colorCompressionType;
			COMPRESSION_TYPE_DEPTH m_depthCompressionType;
			UINT64 m_colorSizeBytes;	//compressed byte size
			UINT64 m_depthSizeBytes;	//compressed byte size
			unsigned char* m_colorCompressed;
			unsigned char* m_depthCompressed;
			unsigned int m_colorWidth, m_colorHeight;
			unsigned int m_depthWidth, m_depthHeight;

			//! camera trajectory (from base to current frame)
			mat4f m_frameToWorld;
		};


		//////////////////////////////////
		// SensorData Class starts here //
		//////////////////////////////////

		SensorData() {
			//the first 3 versions [0,1,2] are reserved for the old .sensor files
#define M_SENSOR_DATA_VERSION 3
			m_versionNumber = M_CALIBRATED_SENSOR_DATA_VERSION;
			m_sensorName = "Unknown";
		}

		SensorData(const std::string& filename) {
			//the first 3 versions [0,1,2] are reserved for the old .sensor files
#define M_SENSOR_DATA_VERSION 3
			m_versionNumber = M_CALIBRATED_SENSOR_DATA_VERSION;
			m_sensorName = "Unknown";
			readFromFile(filename);
		}

		~SensorData() {
			free();
		}

		void free() {
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].free();
			}
			m_frames.clear();
			m_calibrationColor.setIdentity();
			m_calibrationDepth.setIdentity();
			m_colorWidth = 0;
			m_colorHeight = 0;
			m_depthWidth = 0;
			m_depthHeight = 0;
		}

		void assertVersionNumber() const {
			if (m_versionNumber != M_CALIBRATED_SENSOR_DATA_VERSION)	throw MLIB_EXCEPTION("Invalid file version");
		}

		void writeToFile(const std::string& filename) const {
			std::ofstream out(filename, std::ios::binary);

			out.write((const char*)&m_versionNumber, sizeof(unsigned int));
			UINT64 strLen = m_sensorName.size();
			out.write((const char*)&strLen, sizeof(UINT64));
			out.write((const char*)&m_sensorName[0], strLen*sizeof(char));

			m_calibrationColor.writeToFile(out);
			m_calibrationDepth.writeToFile(out);

			out.write((const char*)&m_colorWidth, sizeof(unsigned int));
			out.write((const char*)&m_colorHeight, sizeof(unsigned int));
			out.write((const char*)&m_depthWidth, sizeof(unsigned int));
			out.write((const char*)&m_depthHeight, sizeof(unsigned int));
			out.write((const char*)&m_depthShift, sizeof(float));

			UINT64 numFrames = m_frames.size();
			out.write((const char*)&numFrames, sizeof(UINT64));
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].writeToFile(out);
			}
		}


		void readFromFile(const std::string& filename) {
			std::ifstream in(filename, std::ios::binary);

			if (!in.is_open()) {
				throw MLIB_EXCEPTION("could not open file " + filename);
			}

			in.read((char*)&m_versionNumber, sizeof(unsigned int));
			assertVersionNumber();
			UINT64 strLen = 0;
			in.read((char*)&strLen, sizeof(UINT64));
			m_sensorName.resize(strLen);
			in.read((char*)&m_sensorName[0], strLen*sizeof(char));

			m_calibrationColor.readFromFile(in);
			m_calibrationDepth.readFromFile(in);

			in.read((char*)&m_colorWidth, sizeof(unsigned int));
			in.read((char*)&m_colorHeight, sizeof(unsigned int));
			in.read((char*)&m_depthWidth, sizeof(unsigned int));
			in.read((char*)&m_depthHeight, sizeof(unsigned int));
			in.read((char*)&m_depthShift, sizeof(unsigned int));

			UINT64 numFrames = 0;
			in.read((char*)&numFrames, sizeof(UINT64));
			m_frames.resize(numFrames);
			for (size_t i = 0; i < m_frames.size(); i++) {
				m_frames[i].readFromFile(in);
			}
		}



		class StringCounter {
		public:
			StringCounter(const std::string& base, const std::string fileEnding, unsigned int numCountDigits = 0, unsigned int initValue = 0) {
				m_Base = base;
				if (fileEnding[0] != '.') {
					m_FileEnding = ".";
					m_FileEnding.append(fileEnding);
				}
				else {
					m_FileEnding = fileEnding;
				}
				m_NumCountDigits = numCountDigits;
				m_InitValue = initValue;
				resetCounter();
			}

			~StringCounter() {
			}

			std::string getNext() {
				std::string curr = getCurrent();
				m_Current++;
				return curr;
			}

			std::string getCurrent() {
				std::stringstream ss;
				ss << m_Base;
				for (unsigned int i = std::max(1u, (unsigned int)std::ceilf(std::log10f((float)m_Current + 1))); i < m_NumCountDigits; i++) ss << "0";
				ss << m_Current;
				ss << m_FileEnding;
				return ss.str();
			}

			void resetCounter() {
				m_Current = m_InitValue;
			}
		private:
			std::string		m_Base;
			std::string		m_FileEnding;
			unsigned int	m_NumCountDigits;
			unsigned int	m_Current;
			unsigned int	m_InitValue;
		};


//#ifdef 	_HAS_MLIB	//needs free image to write out data
//		//! 7-scenes format
//		void writeToImages(const std::string& outputFolder, const std::string& basename = "frame-") const {
//			if (!ml::util::directoryExists(outputFolder)) ml::util::makeDirectory(outputFolder);
//
//			{
//				//write meta information
//				const std::string& metaData = "info.txt";
//				std::ofstream outMeta(outputFolder + "/" + metaData);
//
//				outMeta << "m_versionNumber" << " = " << m_versionNumber << '\n';
//				outMeta << "m_sensorName" << " = " << m_sensorName << '\n';
//				outMeta << "m_colorWidth" << " = " << m_colorWidth << '\n';
//				outMeta << "m_colorHeight" << " = " << m_colorHeight << '\n';
//				outMeta << "m_depthWidth" << " = " << m_depthWidth << '\n';
//				outMeta << "m_depthHeight" << " = " << m_depthHeight << '\n';
//				outMeta << "m_depthShift" << " = " << m_depthShift << '\n';
//				outMeta << "m_calibrationColorIntrinsic" << " = ";
//				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationColor.m_intrinsic[i] << " ";	outMeta << "\n";
//				outMeta << "m_calibrationColorExtrinsic" << " = ";
//				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationColor.m_extrinsic[i] << " ";	outMeta << "\n";
//				outMeta << "m_calibrationDepthIntrinsic" << " = ";
//				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationDepth.m_intrinsic[i] << " ";	outMeta << "\n";
//				outMeta << "m_calibrationDepthExtrinsic" << " = ";
//				for (unsigned int i = 0; i < 16; i++) outMeta << m_calibrationDepth.m_extrinsic[i] << " ";	outMeta << "\n";
//				UINT64 numFrames = m_frames.size();
//				outMeta << "m_frames.size" << " = " << numFrames << "\n";
//			}
//
//			if (m_frames.size() == 0) return;	//nothing to do
//			const RGBDFrame& baseFrame = m_frames[0];	//assumes all frames have the same format
//			std::string colorFormatEnding = "png";	//default is png			
//			if (baseFrame.m_colorCompressionType == RGBDFrame::TYPE_PNG) colorFormatEnding = "png";
//			else if (baseFrame.m_colorCompressionType == RGBDFrame::TYPE_JPEG) colorFormatEnding = "jpg";
//
//
//			StringCounter scColor(outputFolder + "/" + basename, "color." + colorFormatEnding, 6);
//			StringCounter scDepth(outputFolder + "/" + basename, "depth.png", 6);
//			StringCounter scPose(outputFolder + "/" + basename, ".pose.txt", 6);
//
//			for (size_t i = 0; i < m_frames.size(); i++) {
//				std::string colorFile = scColor.getNext();
//				std::string depthFile = scDepth.getNext();
//				std::string poseFile = scPose.getNext();
//
//				std::cout << "current Frame: " << i << std::endl;
//
//				const RGBDFrame& f = m_frames[i];
//
//				//color data
//				if (baseFrame.m_colorCompressionType == RGBDFrame::TYPE_RAW)
//				{
//					RGBDFrame frameCompressed((vec3uc*)f.getColorCompressed(), m_colorWidth, m_colorHeight, NULL, 0, 0, RGBDFrame::TYPE_PNG);
//					FILE* file = fopen(colorFile.c_str(), "wb");
//					if (!file) throw MLIB_EXCEPTION("cannot open file " + colorFile);
//					fwrite(frameCompressed.getColorCompressed(), 1, frameCompressed.getColorSizeBytes(), file);
//					fclose(file);
//					frameCompressed.free();
//				}
//				else if (baseFrame.m_colorCompressionType == RGBDFrame::TYPE_PNG || baseFrame.m_colorCompressionType == RGBDFrame::TYPE_JPEG)
//				{
//					FILE* file = fopen(colorFile.c_str(), "wb");
//					if (!file) throw MLIB_EXCEPTION("cannot open file " + colorFile);
//					fwrite(f.getColorCompressed(), 1, f.getColorSizeBytes(), file);
//					fclose(file);
//				}
//				else {
//					throw MLIB_EXCEPTION("unknown format");
//				}
//
//				//depth data
//				const bool writeDepthData = true;
//				if (writeDepthData)
//				{
//					unsigned short* depth = f.decompressDepthAlloc();
//					DepthImage16 image(m_depthWidth, m_depthHeight, depth);
//					FreeImageWrapper::saveImage(depthFile, image);					
//					std::free(depth);
//				}
//
//				savePoseFile(poseFile, f.m_frameToWorld);
//			}
//		}
//
//		//! 7-scenes format
//		void readFromImages(const std::string& sourceFolder, const std::string& basename = "frame-", const std::string& colorEnding = "png") 
//		{
//			if (colorEnding != "png" && colorEnding != "jpg") throw MLIB_EXCEPTION("invalid color format " + colorEnding);
//			
//
//			{
//				//write meta information
//				const std::string& metaData = "info.txt";
//				std::ifstream inMeta(sourceFolder + "/" + metaData);
//
//				std::string varName; std::string seperator;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_versionNumber;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_sensorName;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_colorWidth;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_colorHeight;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthWidth;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthHeight;
//				inMeta >> varName; inMeta >> seperator; inMeta >> m_depthShift;
//
//				inMeta >> varName; inMeta >> seperator;
//				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationColor.m_intrinsic[i];
//				inMeta >> varName; inMeta >> seperator;
//				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationColor.m_extrinsic[i];
//				inMeta >> varName; inMeta >> seperator;
//				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationDepth.m_intrinsic[i];
//				inMeta >> varName; inMeta >> seperator;
//				for (unsigned int i = 0; i < 16; i++) inMeta >> m_calibrationDepth.m_extrinsic[i];
//				UINT64 numFrames;
//				inMeta >> varName; inMeta >> numFrames;
//			}
//
//
//			StringCounter scColor(sourceFolder + "/" + basename, "color." + colorEnding, 6);
//			StringCounter scDepth(sourceFolder + "/" + basename, "depth.png", 6);
//			StringCounter scPose(sourceFolder + "/" + basename, ".pose.txt", 6);
//
//			for (unsigned int i = 0;; i++) {
//				std::string colorFile = scColor.getNext();
//				std::string depthFile = scDepth.getNext();
//				std::string poseFile = scPose.getNext();
//
//				if (!ml::util::fileExists(colorFile) || !ml::util::fileExists(depthFile) || !ml::util::fileExists(poseFile)) {
//					std::cout << "DONE" << std::endl;
//					break;
//				}
//
//				std::cout << "current Frame: " << i << std::endl;
//
//				//ColorImageR8G8B8 colorImage;	ml::FreeImageWrapper::loadImage(colorFile, colorImage, false);
//				//vec3uc* colorData = new vec3uc[m_colorWidth*m_colorHeight];
//				//memcpy(colorData, colorImage.getPointer(), sizeof(vec3uc)*m_colorWidth*m_colorHeight);
//				std::ifstream colorStream(colorFile, std::ios::binary | std::ios::ate);
//				size_t colorSizeBytes = colorStream.tellg();
//				colorStream.seekg(0, std::ios::beg);
//				unsigned char* colorData = (unsigned char*)std::malloc(colorSizeBytes);
//				colorStream.read((char*)colorData, colorSizeBytes);
//				colorStream.close();
//
//
//				DepthImage16 depthImage;			ml::FreeImageWrapper::loadImage(depthFile, depthImage, false);
//				unsigned short*	depthData = new unsigned short[m_depthWidth*m_depthHeight];
//				memcpy(depthData, depthImage.getPointer(), sizeof(unsigned short)*m_depthWidth*m_depthHeight);
//				
//
//				ml::SensorData::RGBDFrame::COMPRESSION_TYPE_COLOR compressionColor = ml::SensorData::RGBDFrame::COMPRESSION_TYPE_COLOR::TYPE_PNG;
//				if (colorEnding == "png")		compressionColor = ml::SensorData::RGBDFrame::COMPRESSION_TYPE_COLOR::TYPE_PNG;
//				else if (colorEnding == "jpg")	compressionColor = ml::SensorData::RGBDFrame::COMPRESSION_TYPE_COLOR::TYPE_JPEG;
//				else throw MLIB_EXCEPTION("invalid color format " + compressionColor);
//
//				//by default use TYPE_OCCI_USHORT (it's just the best)
//				ml::SensorData::RGBDFrame::COMPRESSION_TYPE_DEPTH compressionDepth = ml::SensorData::RGBDFrame::COMPRESSION_TYPE_DEPTH::TYPE_OCCI_USHORT;
//
//				//m_frames.push_back(RGBDFrame(colorData, m_colorWidth, m_colorHeight, depthData, m_depthWidth, m_depthHeight, compressionColor, compressionDepth));
//				m_frames.push_back(RGBDFrame(NULL, 0, 0, depthData, m_depthWidth, m_depthHeight, compressionColor, compressionDepth));
//				m_frames.back().m_colorCompressionType = compressionColor;
//				m_frames.back().m_colorWidth = m_colorWidth;
//				m_frames.back().m_colorHeight = m_colorHeight;
//				m_frames.back().m_colorSizeBytes = colorSizeBytes;
//				m_frames.back().m_colorCompressed = colorData;
//
//				//! camera trajectory (from base to current frame)
//				mat4f m_frameToWorld;
//
//				mat4f pose = loadPoseFile(poseFile);
//				m_frames.back().m_frameToWorld = pose;
//
//				////debug
//				//{
//				//	unsigned short* depth = m_frames.back().decompressDepth();
//				//	ml::DepthImage16 depth16(m_depthWidth, m_depthHeight, depth);
//				//	ml::FreeImageWrapper::saveImage(sourceFolder + "/" + "depth" + std::to_string(i) + ".png", ml::ColorImageR32G32B32A32(depth16));
//				//}
//
//				SAFE_DELETE_ARRAY(depthData);
//			}
//		}
//#endif //_HAS_MLIB

		bool operator==(const SensorData& other) const {
			if (m_versionNumber != other.m_versionNumber) return false;
			if (m_sensorName != other.m_sensorName) return false;
			if (m_calibrationColor != other.m_calibrationColor) return false;
			if (m_calibrationDepth != other.m_calibrationDepth) return false;
			if (m_colorWidth != other.m_colorWidth) return false;
			if (m_colorHeight != other.m_colorHeight) return false;
			if (m_depthWidth != other.m_depthWidth) return false;
			if (m_depthHeight != other.m_depthHeight) return false;
			if (m_depthShift != other.m_depthShift) return false;
			if (m_frames.size() != other.m_frames.size()) return false;
			for (size_t i = 0; i < m_frames.size(); i++) {
				if (m_frames[i] != other.m_frames[i])	return false;
			}
			return true;
		}

		bool operator!=(const SensorData& other) const {
			return !((*this) == other);
		}

		///////////////////////////////
		//MEMBER VARIABLES START HERE//
		///////////////////////////////

		unsigned int	m_versionNumber;
		std::string		m_sensorName;

		CalibrationData m_calibrationColor;
		CalibrationData m_calibrationDepth;

		unsigned int m_colorWidth;
		unsigned int m_colorHeight;
		unsigned int m_depthWidth;
		unsigned int m_depthHeight;
		float m_depthShift;	//conversion from float[m] to ushort

		std::vector<RGBDFrame> m_frames;

		/////////////////////////////
		//MEMBER VARIABLES END HERE//
		/////////////////////////////


		static mat4f loadPoseFile(const std::string& filename) {
			std::ifstream file(filename);
			mat4f m;
			file >>
				m._m00 >> m._m01 >> m._m02 >> m._m03 >>
				m._m10 >> m._m11 >> m._m12 >> m._m13 >>
				m._m20 >> m._m21 >> m._m22 >> m._m23 >>
				m._m30 >> m._m31 >> m._m32 >> m._m33;
			file.close();
			return m;
		}

		static void savePoseFile(const std::string& filename, const mat4f& m) {
			std::ofstream file(filename);
			file <<
				m._m00 << " " << m._m01 << " " << m._m02 << " " << m._m03 << "\n" <<
				m._m10 << " " << m._m11 << " " << m._m12 << " " << m._m13 << "\n" <<
				m._m20 << " " << m._m21 << " " << m._m22 << " " << m._m23 << "\n" <<
				m._m30 << " " << m._m31 << " " << m._m32 << " " << m._m33;
			file.close();
		}


	};


	class RGBDFrameCacheRead {
public:
		struct FrameState {
			FrameState() {
				m_bIsReady = false;
				m_colorFrame = NULL;
				m_depthFrame = NULL;
			}
			~FrameState() {
				//NEEDS MANUAL FREE
			}
			void free() {
				if (m_colorFrame) std::free(m_colorFrame);
				if (m_depthFrame) std::free(m_depthFrame);
			}
			bool			m_bIsReady;
			vec3uc*			m_colorFrame;
			unsigned short*	m_depthFrame;
		};
		RGBDFrameCacheRead(SensorData* sensorData, unsigned int cacheSize) : m_decompThread() {
			m_sensorData = sensorData;
			m_cacheSize = cacheSize;
			m_bTerminateThread = false;
			m_nextFromSensorCache = 0;
			m_nextFromSensorData = 0;
			startDecompBackgroundThread();
		}

		~RGBDFrameCacheRead() {
			m_bTerminateThread = true;
			if (m_decompThread.joinable()) {
				m_decompThread.join();
			}

			for (auto& fs : m_data) {
				fs.free();
			}
		}

		FrameState getNext() {
			while (1) {
				if (m_nextFromSensorCache >= m_sensorData->m_frames.size()) {
					m_bTerminateThread = true;	// should be already true anyway
					break; //we're done
				}
				if (m_data.size() > 0 && m_data.front().m_bIsReady) {
					m_mutexList.lock();
					FrameState fs = m_data.front();
					m_data.pop_front();
					m_mutexList.unlock();
					m_nextFromSensorCache++;
					return fs;
				}
				else {
					Sleep(0);
				}
			}
			return FrameState();
		}

	private:
		void startDecompBackgroundThread() {
			m_decompThread = std::thread(decompFunc, this);
		}

		static void decompFunc(RGBDFrameCacheRead* cache) {
			while (1) {
				if (cache->m_bTerminateThread) break;
				if (cache->m_nextFromSensorData >= cache->m_sensorData->m_frames.size()) break;	//we're done

				if (cache->m_data.size() < cache->m_cacheSize) {	//need to fill the cache
					cache->m_mutexList.lock();
					cache->m_data.push_back(FrameState());
					cache->m_mutexList.unlock();

					SensorData::RGBDFrame& frame = cache->m_sensorData->m_frames[cache->m_nextFromSensorData];
					FrameState& fs = cache->m_data.back();

					//std::cout << "decompressing frame " << cache->m_nextFromSensorData << std::endl;
					fs.m_colorFrame = frame.decompressColorAlloc();
					fs.m_depthFrame = frame.decompressDepthAlloc();
					fs.m_bIsReady = true;
					cache->m_nextFromSensorData++;
				}
			}

			std::cout << "done decomp" << std::endl;
			int a = 5;
		}

		SensorData* m_sensorData;
		unsigned int m_cacheSize;

		std::list<FrameState> m_data;
		std::thread m_decompThread;
		std::mutex m_mutexList;
		std::atomic<bool> m_bTerminateThread;

		unsigned int m_nextFromSensorData;
		unsigned int m_nextFromSensorCache;
	};

	class RGBDFrameCacheWrite {
	private:
		struct FrameState {
			FrameState() {
				m_bIsReady = false;
				m_colorFrame = NULL;
				m_depthFrame = NULL;
			}
			~FrameState() {
				//NEEDS MANUAL FREE
			}
			void free() {
				if (m_colorFrame) std::free(m_colorFrame);
				if (m_depthFrame) std::free(m_depthFrame);
			}
			bool			m_bIsReady;
			vec3uc*			m_colorFrame;
			unsigned short*	m_depthFrame;
		};
	public:
		RGBDFrameCacheWrite(SensorData* sensorData, unsigned int cacheSize) : m_compThread() {
			m_sensorData = sensorData;
			m_cacheSize = cacheSize;
			m_bTerminateThread = false;
			startCompBackgroundThread();
		}
 
		~RGBDFrameCacheWrite() {
			m_bTerminateThread = true;
			if (m_compThread.joinable()) {
				m_compThread.join();
			}

			for (auto& fs : m_data) {
				fs.free();
			}
		}

		//! appends the data to the cache for process AND frees the memory
		void writeNextAndFree(vec3uc* color, unsigned short* depth) {
			FrameState fs;
			fs.m_colorFrame = color;
			fs.m_depthFrame = depth;
			while (m_data.size() >= m_cacheSize) {
				Sleep(0);	//wait until we have space in our cache
			}
			m_mutexList.lock();
			m_data.push_back(fs);
			m_mutexList.unlock();
		}

	private:
		void startCompBackgroundThread() {
			m_compThread = std::thread(compFunc, this);
		}

		static void compFunc(RGBDFrameCacheWrite* cache) {
			while (1) {
				if (cache->m_bTerminateThread && cache->m_data.size() == 0) break;	//if terminated AND compression is done
				if (cache->m_data.size() > 0) {
					cache->m_mutexList.lock();
					FrameState fs = cache->m_data.front();
					cache->m_data.pop_front();
					cache->m_mutexList.unlock();

					cache->m_sensorData->m_frames.push_back(SensorData::RGBDFrame(
						fs.m_colorFrame,
						cache->m_sensorData->m_colorWidth,
						cache->m_sensorData->m_colorHeight,
						fs.m_depthFrame,
						cache->m_sensorData->m_depthWidth,
						cache->m_sensorData->m_depthHeight));
					fs.free();

				}
			}
		}

		SensorData* m_sensorData;
		unsigned int m_cacheSize;

		std::list<FrameState> m_data;
		std::thread m_compThread;
		std::mutex m_mutexList;
		std::atomic<bool> m_bTerminateThread;
	};

#ifndef VAR_STR_LINE
#define VAR_STR_LINE(x) '\t' << #x << '=' << x << '\n'
#endif
	inline std::ostream& operator<<(std::ostream& s, const SensorData& sensorData) {
		s << "CalibratedSensorData:\n";
		s << VAR_STR_LINE(sensorData.m_versionNumber);
		s << VAR_STR_LINE(sensorData.m_sensorName);
		s << VAR_STR_LINE(sensorData.m_colorWidth);
		s << VAR_STR_LINE(sensorData.m_colorHeight);
		s << VAR_STR_LINE(sensorData.m_depthWidth);
		s << VAR_STR_LINE(sensorData.m_depthHeight);
		//s << VAR_STR_LINE(sensorData.m_CalibrationDepth);
		//s << VAR_STR_LINE(sensorData.m_CalibrationColor);
		s << VAR_STR_LINE(sensorData.m_frames.size());
		return s;
	}

}	//namespace ml

#endif //_SENSOR_FILE_H_

