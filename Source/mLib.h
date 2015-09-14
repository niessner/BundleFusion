
//
// mLib config options
//

#define MLIB_ERROR_CHECK
#define MLIB_BOUNDS_CHECK
#define MLIB_SOCKETS

//
// mLib includes
//


#include "mLibCore.h"
#include "mLibDepthCamera.h"
#include "mLibFreeImage.h"
#include "mLibZlib.h"

using namespace ml;


#define MLIB_CUDA_SAFE_CALL(b) { if(b != cudaSuccess) throw MLIB_EXCEPTION(cudaGetErrorString(b)); }
#define MLIB_CUDA_SAFE_FREE(b) { if(!b) { MLIB_CUDA_SAFE_CALL(cudaFree(b)); b = NULL; } }