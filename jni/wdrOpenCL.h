#ifndef __WDROPENCL_H__
#define __WDROPENCL_H__

#include <iostream>
#include <sstream>
#include <fstream>
#include <log.h>
#include <CL/cl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

typedef int            INT32;
typedef unsigned int   UINT32;
typedef short          INT16;
typedef unsigned short UINT16;
typedef char           INT8;
typedef unsigned char  UINT8;
typedef void           VOID;
typedef unsigned long long  LONG;
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
    UINT8* mGrayBuffer;
    UINT32* mIntefralBuffer;
public:
    void integralImage();
    void process();
    void scale();
    bool loadData(string path);
private:
    INT32 mGrayBufferSize;
    INT32 mIntefralBufferSize;
    cl_int mArraySize;
private:
    cl_context mContext;
    cl_command_queue mCommandQueue;
    cl_program mProgram;
    cl_device_id mDevice;
    cl_kernel mKernel;
    cl_kernel mKernelPSH;
    cl_kernel mKernelPSV;
    cl_kernel mKernelTM;
    const INT32 mNumMemoryObjects;
    cl_mem mMemoryObjects[2];
    cl_int mErrorNumber;
    double mStart, mEnd;
private:
    bool initWdr();
    INT32 initOpenCL();
    void preSumHorizontal();
    void preSumVertical();
    string errorNumberToString(cl_int errorNumber);
    inline bool checkSuccess(cl_int errorNumber);
    bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program,
                cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects);
    bool createContext(cl_context* context);
    bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device);
    bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program);

private:
    bool RGBToRGBA(const unsigned char* const rgbData, unsigned char* const rgbaData, INT32 width, INT32 height);
    bool RGBAToRGB(const unsigned char* const rgbaData, unsigned char* const rgbData, INT32 width, INT32 height);
};


#endif
