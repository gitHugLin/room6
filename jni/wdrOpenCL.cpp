#include "wdrOpenCL.h"

static double gWork_begin = 0;
static double gWork_end = 0;
static double gTime = 0;
//开始计时
static void workEnd(char *tag = "TimeCounts");
static void workBegin();
static void workBegin()
{
    gWork_begin = getTickCount();
}
//结束计时
static void workEnd(char *tag)
{
    gWork_end = getTickCount() - gWork_begin;
    gTime = gWork_end /((double)getTickFrequency() )* 1000.0;
    printf("[TAG: %s ]:TIME = %lf ms \n",tag,gTime);
}

wdrOpenCL::wdrOpenCL():mNumMemoryObjects(2)
{
    mContext = 0;
    mCommandQueue = 0;
    mProgram = 0;
    mDevice = 0;
    mKernel = 0;
    mMemoryObjects[0] = 0;
    mMemoryObjects[1] = 0;
    mStart = 0;
    mEnd = 0;
    mGrayBuffer = NULL;
    mIntefralBuffer = NULL;
    mWidth = 0;
    mHeight = 0;
    mArraySize = 0;
    mGrayBufferSize = 0;
    mIntefralBufferSize= 0;
}

wdrOpenCL::~wdrOpenCL() {
    if (NULL != mIntefralBuffer) {
        delete []mIntefralBuffer;
        mIntefralBuffer = NULL;
    }
    mGrayBuffer = NULL;
}

string wdrOpenCL::errorNumberToString(cl_int errorNumber)
{
    switch (errorNumber)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        default:
            return "Unknown error";
    }
}

inline bool wdrOpenCL::checkSuccess(cl_int errorNumber)
{
    if (errorNumber != CL_SUCCESS)
    {
        cerr << "OpenCL error: " << errorNumberToString(errorNumber) << endl;
        return false;
    }
    return true;
}

bool wdrOpenCL::cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program,
            cl_kernel kernel, cl_mem* memoryObjects, int numberOfMemoryObjects)
{
    bool returnValue = true;
    if (context != 0) {
        if (!checkSuccess(clReleaseContext(context))) {
            cerr << "Releasing the OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (commandQueue != 0) {
        if (!checkSuccess(clReleaseCommandQueue(commandQueue))) {
            cerr << "Releasing the OpenCL command queue failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (kernel != 0) {
        if (!checkSuccess(clReleaseKernel(kernel))) {
            cerr << "Releasing the OpenCL kernel failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    if (program != 0) {
        if (!checkSuccess(clReleaseProgram(program))) {
            cerr << "Releasing the OpenCL program failed. " << __FILE__ << ":"<< __LINE__ << endl;
            returnValue = false;
        }
    }

    for (int index = 0; index < numberOfMemoryObjects; index++) {
        if (memoryObjects[index] != 0) {
            if (!checkSuccess(clReleaseMemObject(memoryObjects[index]))) {
                cerr << "Releasing the OpenCL memory object " << index << " failed. " << __FILE__ << ":"<< __LINE__ << endl;
                returnValue = false;
            }
        }
    }

    return returnValue;
}

bool wdrOpenCL::createContext(cl_context* context)
{
    cl_int errorNumber = 0;
    cl_uint numberOfPlatforms = 0;
    cl_platform_id firstPlatformID = 0;

    /* Retrieve a single platform ID. */
    if (!checkSuccess(clGetPlatformIDs(1, &firstPlatformID, &numberOfPlatforms))) {
        cerr << "Retrieving OpenCL platforms failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (numberOfPlatforms <= 0) {
        cerr << "No OpenCL platforms found. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Get a context with a GPU device from the platform found above. */
    cl_context_properties contextProperties [] = {CL_CONTEXT_PLATFORM, (cl_context_properties)firstPlatformID, 0};
    *context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber)) {
        cerr << "Creating an OpenCL context failed. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

bool wdrOpenCL::createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device)
{
    cl_int errorNumber = 0;
    cl_device_id* devices = NULL;
    size_t deviceBufferSize = -1;

    /* Retrieve the size of the buffer needed to contain information about the devices in this OpenCL context. */
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize))) {
        cerr << "Failed to get OpenCL context information. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if(deviceBufferSize == 0) {
        cerr << "No OpenCL devices found. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Retrieve the list of devices available in this context. */
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    if (!checkSuccess(clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL))) {
        cerr << "Failed to get the OpenCL context information. " << __FILE__ << ":"<< __LINE__ << endl;
        delete [] devices;
        return false;
    }

    /* Use the first available device in this context. */
    *device = devices[0];
    delete [] devices;

    /* Set up the command queue with the selected device. */
    *commandQueue = clCreateCommandQueue(context, *device, CL_QUEUE_PROFILING_ENABLE, &errorNumber);
    if (!checkSuccess(errorNumber)) {
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

bool wdrOpenCL::createProgram(cl_context context, cl_device_id device, string filename, cl_program* program)
{
    cl_int errorNumber = 0;
    ifstream kernelFile(filename.c_str(), ios::in);

    if(!kernelFile.is_open()) {
        cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /*
     * Read the kernel file into an output stream.
     * Convert this into a char array for passing to OpenCL.
     */
    ostringstream outputStringStream;
    outputStringStream << kernelFile.rdbuf();
    string srcStdStr = outputStringStream.str();
    const char* charSource = srcStdStr.c_str();

    *program = clCreateProgramWithSource(context, 1, &charSource, NULL, &errorNumber);
    if (!checkSuccess(errorNumber) || program == NULL) {
        cerr << "Failed to create OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Try to build the OpenCL program. */
    bool buildSuccess = checkSuccess(clBuildProgram(*program, 0, NULL, NULL, NULL, NULL));

    /* Get the size of the build log. */
    size_t logSize = 0;
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

    /*
     * If the build succeeds with no log, an empty string is returned (logSize = 1),
     * we only want to print the message if it has some content (logSize > 1).
     */
    if (logSize > 1) {
        char* log = new char[logSize];
        clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        string* stringChars = new string(log, logSize);
        cerr << "Build log:\n " << *stringChars << endl;

        delete[] log;
        delete stringChars;
    }

    if (!buildSuccess) {
        clReleaseProgram(*program);
        cerr << "Failed to build OpenCL program. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    return true;
}

bool wdrOpenCL::RGBToRGBA(const unsigned char* const rgbData, unsigned char* const rgbaData, INT32 width, INT32 height)
{
    if (rgbData == NULL) {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (rgbaData == NULL) {
        cerr << "rgbaData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = 0; n < width * height; n++) {
        /* Copy the RGB components directly. */
        rgbaData[4 * n + 0] = rgbData[3 * n + 0];
        rgbaData[4 * n + 1] = rgbData[3 * n + 1];
        rgbaData[4 * n + 2] = rgbData[3 * n + 2];

        /* Set the alpha channel to 255 (fully opaque). */
        rgbaData[4 * n + 3] = (unsigned char)255;
    }
    return true;
}

bool wdrOpenCL::RGBAToRGB(const unsigned char* const rgbaData, unsigned char* const rgbData, INT32 width, INT32 height)
{
    if (rgbaData == NULL) {
        cerr << "rgbaData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (rgbData == NULL) {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = 0; n < width * height; n++) {
        /* Copy the RGB components but throw away the alpha channel. */
        rgbData[3 * n + 0] = rgbaData[4 * n + 0];
        rgbData[3 * n + 1] = rgbaData[4 * n + 1];
        rgbData[3 * n + 2] = rgbaData[4 * n + 2];
    }
    return true;
}

INT32 wdrOpenCL::initOpenCL() {
    int ret = 1;
    if (!createContext(&mContext)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return -1;
    }

    if (!createCommandQueue(mContext, &mCommandQueue, &mDevice)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return -1;
    }

    if (!createProgram(mContext, mDevice, "image_scaling.cl", &mProgram)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return -1;
    }

    mKernel = clCreateKernel(mProgram, "image_scaling", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return -1;
    }
    return ret;
}

bool wdrOpenCL::initWdr()
{
    bool ret = true;
    if (!createContext(&mContext)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    if (!createCommandQueue(mContext, &mCommandQueue, &mDevice)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    if (!createProgram(mContext, mDevice, "wdr.cl", &mProgram)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    mKernelPSH = clCreateKernel(mProgram, "preSumHorizontal", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernelPSH, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    mKernelPSV = clCreateKernel(mProgram, "preSumVertical", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernelPSV, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
    mKernelTM = clCreateKernel(mProgram, "toneMapping", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernelTM, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }
}

bool wdrOpenCL::loadData(string path) {
    bool ret = true;
    mSrcImage = imread(path);
    if(mSrcImage.empty()) {
        printf("Error: can not loat picture\n" );
        ret = false;
    }
    mWidth = mSrcImage.cols;
    mHeight = mSrcImage.rows;
    mArraySize = mWidth*mHeight;
    mGrayBufferSize = mArraySize*sizeof(cl_uchar);
    mIntefralBufferSize = mArraySize*sizeof(cl_uint);
    mGrayBuffer = mSrcImage.data;
    mIntefralBuffer = new UINT32[mArraySize];
    return ret;
}

void wdrOpenCL::preSumHorizontal()
{
    workBegin();
    if (!createContext(&mContext)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    if (!createCommandQueue(mContext, &mCommandQueue, &mDevice)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    if (!createProgram(mContext, mDevice, "wdr.cl", &mProgram)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }
    mKernel = clCreateKernel(mProgram, "preSumHorizontal", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    workEnd("initOpenCL part1");
    /*
     * Ask the OpenCL implementation to allocate buffers for the data.
     * We ask the OpenCL implemenation to allocate memory rather than allocating
     * it on the CPU to avoid having to copy the data later.
     * The read/write flags relate to accesses to the memory from within the kernel.
     */
     workBegin();
    bool createMemoryObjectsSuccess = true;
    mMemoryObjects[0] = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, mGrayBufferSize, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);
    mMemoryObjects[1] = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, mIntefralBufferSize, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);

    if (!createMemoryObjectsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU. */
    bool mapMemoryObjectsSuccess = true;
    cl_uchar* inputGrayBuffer = (cl_uchar*)clEnqueueMapBuffer(mCommandQueue, mMemoryObjects[0],
                                    CL_TRUE, CL_MAP_WRITE, 0, mGrayBufferSize, 0, NULL, NULL, &mErrorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(mErrorNumber);
    if (!mapMemoryObjectsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /*
     * Unmap the memory objects as we have finished using them from the CPU side.
     * We unmap the memory because otherwise:
     * - reads and writes to that memory from inside a kernel on the OpenCL side are undefined.
     * - the OpenCL implementation cannot free the memory when it is finished.
     */
    if (!checkSuccess(clEnqueueUnmapMemObject(mCommandQueue, mMemoryObjects[0], inputGrayBuffer, 0, NULL, NULL))) {
       cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }
workEnd("initOpenCL part2");
workBegin();
    cl_int width = mWidth;
    cl_int height = mHeight;

    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 0, sizeof(cl_mem), &mMemoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 1, sizeof(cl_mem), &mMemoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 2, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 3, sizeof(cl_int), &height));
    if (!setKernelArgumentsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retrieve profiling information later. */
    cl_event event = 0;
    /* [Global work size] */
    /*
     * Each instance of our OpenCL kernel operates on a single element of each array so the number of
     * instances needed is the number of elements in the array.
     */
    workEnd("clSetKernelArg");
    workBegin();
    size_t globalWorksize[2] = {1, height};
    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(mCommandQueue, mKernel, 2, NULL, globalWorksize, NULL, 0, NULL, &event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* [Global work size] */
    /* Wait for kernel execution completion. */
    if (!checkSuccess(clFinish(mCommandQueue))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    workEnd("clEnqueueNDRangeKernel");
    workBegin();
    /* Print the profiling information for the event. */
    //printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* Get a pointer to the output data. */
    cl_uint* outputIntegralBuffer = (cl_uint*)clEnqueueMapBuffer(mCommandQueue, mMemoryObjects[1],
                                    CL_TRUE, CL_MAP_READ, 0, mIntefralBufferSize, 0, NULL, NULL, &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    workEnd("clEnqueueMapBuffer");
    /* Unmap the memory object as we are finished using them from the CPU side. */
    if (!checkSuccess(clEnqueueUnmapMemObject(mCommandQueue, mMemoryObjects[1], outputIntegralBuffer, 0, NULL, NULL))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* Release OpenCL objects. */
    cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
    printf("finished preSumHorizontal!\n");
}

void wdrOpenCL::preSumVertical()
{
    if (!createContext(&mContext)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    if (!createCommandQueue(mContext, &mCommandQueue, &mDevice)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    if (!createProgram(mContext, mDevice, "preSumVertical.cl", &mProgram)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
    }
    mKernel = clCreateKernel(mProgram, "preSumVertical", &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /*
     * Ask the OpenCL implementation to allocate buffers for the data.
     * We ask the OpenCL implemenation to allocate memory rather than allocating
     * it on the CPU to avoid having to copy the data later.
     * The read/write flags relate to accesses to the memory from within the kernel.
     */
    bool createMemoryObjectsSuccess = true;
    mMemoryObjects[0] = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, mGrayBufferSize, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);
    mMemoryObjects[1] = clCreateBuffer(mContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, mIntefralBufferSize, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);

    if (!createMemoryObjectsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* Map the memory buffers created by the OpenCL implementation to pointers so we can access them on the CPU. */
    bool mapMemoryObjectsSuccess = true;
    cl_uchar* inputGrayBuffer = (cl_uchar*)clEnqueueMapBuffer(mCommandQueue, mMemoryObjects[0],
                                    CL_TRUE, CL_MAP_WRITE, 0, mGrayBufferSize, 0, NULL, NULL, &mErrorNumber);
    mapMemoryObjectsSuccess &= checkSuccess(mErrorNumber);
    if (!mapMemoryObjectsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /*
     * Unmap the memory objects as we have finished using them from the CPU side.
     * We unmap the memory because otherwise:
     * - reads and writes to that memory from inside a kernel on the OpenCL side are undefined.
     * - the OpenCL implementation cannot free the memory when it is finished.
     */
    if (!checkSuccess(clEnqueueUnmapMemObject(mCommandQueue, mMemoryObjects[0], inputGrayBuffer, 0, NULL, NULL))) {
       cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
       cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }

    cl_int width = mWidth;
    cl_int height = mHeight;

    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 0, sizeof(cl_mem), &mMemoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 1, sizeof(cl_mem), &mMemoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 2, sizeof(cl_int), &width));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 3, sizeof(cl_int), &height));
    if (!setKernelArgumentsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* An event to associate with the Kernel. Allows us to retrieve profiling information later. */
    cl_event event = 0;
    /* [Global work size] */
    /*
     * Each instance of our OpenCL kernel operates on a single element of each array so the number of
     * instances needed is the number of elements in the array.
     */
     workBegin();
    size_t globalWorksize[2] = {1, height};
    /* Enqueue the kernel */
    if (!checkSuccess(clEnqueueNDRangeKernel(mCommandQueue, mKernel, 2, NULL, globalWorksize, NULL, 0, NULL, &event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* [Global work size] */
    /* Wait for kernel execution completion. */
    if (!checkSuccess(clFinish(mCommandQueue))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    workEnd("clEnqueueNDRangeKernel");
    /* Print the profiling information for the event. */
    //printProfilingInfo(event);
    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* Get a pointer to the output data. */
    cl_uint* outputIntegralBuffer = (cl_uint*)clEnqueueMapBuffer(mCommandQueue, mMemoryObjects[1],
                                    CL_TRUE, CL_MAP_READ, 0, mIntefralBufferSize, 0, NULL, NULL, &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed to map buffer. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* Unmap the memory object as we are finished using them from the CPU side. */
    if (!checkSuccess(clEnqueueUnmapMemObject(mCommandQueue, mMemoryObjects[1], outputIntegralBuffer, 0, NULL, NULL))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Unmapping memory objects failed " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* Release OpenCL objects. */
    cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
    printf("finished preSumVertical!\n");
}

void wdrOpenCL::integralImage()
{
    preSumHorizontal();
}

void wdrOpenCL::process()
{
    loadData("/data/local/input.jpg");
    //workBegin();
    integralImage();
    //workEnd("Time of Process:");
}

void wdrOpenCL::scale()
{
    loadData("/data/local/input.jpg");
    initOpenCL();
    const int scaleFactor = 8;
    INT32 newWidth = mWidth*scaleFactor;
    INT32 newHeight = mHeight*scaleFactor;

    cl_image_format format;
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGBA;
    bool createMemoryObjectsSuccess = true;

    mMemoryObjects[0] = clCreateImage2D(mContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, mWidth, mHeight, 0, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);

    mMemoryObjects[1] = clCreateImage2D(mContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, newWidth, newHeight, 0, NULL, &mErrorNumber);
    createMemoryObjectsSuccess &= checkSuccess(mErrorNumber);

    if (!createMemoryObjectsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {mWidth, mHeight, 1};
    size_t rowPitch;
    //workBegin();
    unsigned char* inputImageRGBA = (unsigned char*)clEnqueueMapImage(mCommandQueue,  mMemoryObjects[0],
                CL_TRUE, CL_MAP_WRITE, origin, region, &rowPitch, NULL, 0, NULL, NULL, &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed mapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    /* [Map image objects to host pointers] */
    /* Convert the input data from RGB to RGBA (moves it to the OpenCL allocated memory at the same time). */
    RGBToRGBA(mGrayBuffer, inputImageRGBA, mWidth, mHeight);
    if (!checkSuccess(clEnqueueUnmapMemObject(mCommandQueue, mMemoryObjects[0], inputImageRGBA, 0, NULL, NULL))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed unmapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    cl_float widthNormalizationFactor = 1.0f / newWidth;
    cl_float heightNormalizationFactor = 1.0f / newHeight;

    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 0, sizeof(cl_mem), &mMemoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 1, sizeof(cl_mem), &mMemoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 2, sizeof(cl_float), &widthNormalizationFactor));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(mKernel, 3, sizeof(cl_float), &heightNormalizationFactor));
    if (!setKernelArgumentsSuccess) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
    }
    const INT32 workDimensions = 2;
    size_t globalWorkSize[workDimensions] = {newWidth, newHeight};
    /* An event to associate with the kernel. Allows us to retrieve profiling information later. */
    cl_event event = 0;
    /* Enqueue the kernel. */
    if (!checkSuccess(clEnqueueNDRangeKernel(mCommandQueue, mKernel, workDimensions, NULL, globalWorkSize, NULL, 0, NULL, &event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* Wait for kernel execution completion. */
    if (!checkSuccess(clFinish(mCommandQueue))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event))) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    size_t newRegion[3] = {newWidth, newHeight, 1};

    unsigned char* outputImage = (unsigned char*)clEnqueueMapImage(mCommandQueue,  mMemoryObjects[1],
                CL_TRUE, CL_MAP_READ, origin, newRegion, &rowPitch, NULL, 0, NULL, NULL, &mErrorNumber);
    if (!checkSuccess(mErrorNumber)) {
        cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
        cerr << "Failed mapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
    }

    unsigned char* outputImageRGB = new unsigned char[newWidth * newHeight * 3];
    RGBAToRGB(outputImage, outputImageRGB, newWidth, newHeight);

    Mat oImage = Mat(newHeight, newWidth, CV_8UC3, outputImageRGB);
    imwrite("/data/local/output_cl.jpg", oImage);
    delete[] outputImageRGB;
    cleanUpOpenCL(mContext, mCommandQueue, mProgram, mKernel, mMemoryObjects, mNumMemoryObjects);
    printf("finished runing scaling image\n");
}
