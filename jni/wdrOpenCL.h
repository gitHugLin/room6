#ifndef __WDROPENCL_H__
#define __WDROPENCL_H__

#include <CL/cl.h>
#include <fstream>
#include <iostream>
#include <log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

using namespace cv;
using namespace std;

typedef int INT32;
typedef unsigned int UINT32;
typedef short INT16;
typedef unsigned short UINT16;
typedef char INT8;
typedef unsigned char UINT8;
typedef void VOID;
typedef unsigned long long LONG;
typedef unsigned char UCHAR;

class wdrOpenCL {
public:
  wdrOpenCL();
  ~wdrOpenCL();

private:
  Mat mSrcImage;
  Mat mDstImage;
  INT32 mWidth;
  INT32 mHeight;
  UINT8 *mGrayBuffer;
  float *mToneMapLut;
  float mGainOffset;

public:
  void process();
  bool loadData(string path, bool pgm = true);

private:
  INT32 mGrayBufferSize;
  INT32 mTMBufferSize;
  cl_int mArraySize;

private:
  cl_context mContext;
  cl_command_queue mCommandQueue;
  cl_program mProgram;
  cl_device_id mDevice;
  cl_kernel mKernelTM;
  const INT32 mNumMemoryObjects;
  cl_mem mMemoryObjects[3];
  cl_int mErrorNumber;

private:
  bool initWdr();
  bool toneMapping();
  void printDeviceInfo(cl_device_id device);
  string errorNumberToString(cl_int errorNumber);
  inline bool checkSuccess(cl_int errorNumber);
  bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue,
                     cl_program program, cl_kernel kernel,
                     cl_mem *memoryObjects, int numberOfMemoryObjects);
  bool createContext(cl_context *context);
  bool createCommandQueue(cl_context context, cl_command_queue *commandQueue,
                          cl_device_id *device);
  bool createProgram(cl_context context, cl_device_id device, string filename,
                     cl_program *program);
};

#endif
