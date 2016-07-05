#ifndef __WDROPENCL_H__
#define __WDROPENCL_H__

#include <sys/time.h>
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

class wdrOpenCL {
public:
    wdrOpenCL();
    ~wdrOpenCL();
private:
    Mat mSrcImage;
    Mat mDstImage;
    int mWidth;
    int mHeight;
    unsigned char* mGrayBuffer;
    unsigned int* mIntefralBuffer;
public:
    void process();
    bool loadData(string path);
private:
    cl_context mContext;
    cl_command_queue mCommandQueue;
    cl_program mProgram;
    cl_device_id mDevice;
    cl_kernel mKernel;
    const int mNumMemoryObjects;
    cl_mem mMemoryObjects[2];
    cl_int mErrorNumber;
    double mStart, mEnd;
private:
    int initOpenCL();
    string errorNumberToString(cl_int errorNumber);
    inline bool checkSuccess(cl_int errorNumber);
    bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program,
                cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects);
    bool createContext(cl_context* context);
    bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device);
    bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program);
    double getTimestamp();

private:
    bool RGBToRGBA(const unsigned char* const rgbData, unsigned char* const rgbaData, int width, int height);
    bool RGBAToRGB(const unsigned char* const rgbaData, unsigned char* const rgbData, int width, int height);
};


#endif
