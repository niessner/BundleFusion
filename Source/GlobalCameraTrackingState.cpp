
#include "stdafx.h"

#include "GlobalCameraTrackingState.h"



// Five level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512, 256, 128, 64, 32};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024, 512, 256, 128, 64};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12, 3, 1, 1, 1};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {6, 3, 2, 2, 1};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1, 1, 1, 1, 1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.05f, 0.07f, 0.09f, 0.11f, 0.13f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.98f, 0.97f, 0.96f, 0.95f, 0.94f};*/

// Three level hierarchy
/*
unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512, 256, 128};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024, 512, 256};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12, 3, 1};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {15, 7, 3};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1, 1, 1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f, 0.15f, 0.2f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.98f, 0.97f, 0.96f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(300.0f/180.0f)*M_PI, (400.0f/180.0f)*M_PI, (500.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {300.3f, 300.5f,300.7f}; // meters
*/

// Two level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512, 256};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024, 512};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12, 3};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {20, 10};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1, 1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f, 0.15f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f, 0.96f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(3000.0f/180.0f)*M_PI, (400.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {3000.4f, 200.5f}; // meters
*/

/*
// One level hierarchy
unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {35};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Queens
// One level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {35};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Passage
// One level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {45};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Bookshop
// One level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {30};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Augustus
// One level hierarchy
/*unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {30};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Statues
// One level hierarchy
/*
unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {30};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.10f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {0.4f}; // meters
*/

// Timings
/*
unsigned int GlobalCameraTrackingState::s_blockSizeNormalize[GlobalCameraTrackingState::s_maxLevels] = {512};
unsigned int GlobalCameraTrackingState::s_numBucketsNormalize[GlobalCameraTrackingState::s_maxLevels] = {1024};
unsigned int GlobalCameraTrackingState::s_localWindowSize[GlobalCameraTrackingState::s_maxLevels] = {12};
unsigned int GlobalCameraTrackingState::s_maxOuterIter[GlobalCameraTrackingState::s_maxLevels] = {15};
unsigned int GlobalCameraTrackingState::s_maxInnerIter[GlobalCameraTrackingState::s_maxLevels] = {1};
float GlobalCameraTrackingState::s_distThres[GlobalCameraTrackingState::s_maxLevels] = {0.15f};
float GlobalCameraTrackingState::s_normalThres[GlobalCameraTrackingState::s_maxLevels] = {0.97f};

float GlobalCameraTrackingState::s_angleTransThres[GlobalCameraTrackingState::s_maxLevels] = {(30000.0f/180.0f)*M_PI};// radians
float GlobalCameraTrackingState::s_distTransThres[GlobalCameraTrackingState::s_maxLevels] = {3000.4f}; // meters
*/



