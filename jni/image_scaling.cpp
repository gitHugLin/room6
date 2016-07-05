#include <sys/time.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <CL/cl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

string errorNumberToString(cl_int errorNumber)
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

inline bool checkSuccess(cl_int errorNumber)
{
    if (errorNumber != CL_SUCCESS)
    {
        cerr << "OpenCL error: " << errorNumberToString(errorNumber) << endl;
        return false;
    }
    return true;
}

bool cleanUpOpenCL(cl_context context, cl_command_queue commandQueue, cl_program program,
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

bool createContext(cl_context* context)
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

bool createCommandQueue(cl_context context, cl_command_queue* commandQueue, cl_device_id* device)
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

bool createProgram(cl_context context, cl_device_id device, string filename, cl_program* program)
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

double getTimestamp()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_usec + tv.tv_sec*1e6;
}

bool RGBToRGBA(const unsigned char* const rgbData, unsigned char* const rgbaData, int width, int height)
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

bool RGBAToRGB(const unsigned char* const rgbaData, unsigned char* const rgbData, int width, int height)
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

/**
 * \brief OpenCL image object sample code.
 * \details Demonstration of how to use OpenCL image objects to resize an image.
 * \return The exit code of the application, non-zero if a problem occurred.
 */
int main(void)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    const int numMemoryObjects = 2;
    cl_mem memoryObjects[numMemoryObjects] = {0, 0};
    cl_int errorNumber;
    double start, end;

    /* Set up OpenCL environment: create context, command queue, program and kernel. */
    if (!createContext(&context)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed to create an OpenCL context. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createCommandQueue(context, &commandQueue, &device)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed to create the OpenCL command queue. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    if (!createProgram(context, device, "image_scaling.cl", &program)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed to create OpenCL program." << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    kernel = clCreateKernel(program, "image_scaling", &errorNumber);
    if (!checkSuccess(errorNumber)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed to create OpenCL kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* The scaling factor to use when resizing the image. */
    const int scaleFactor = 8;

    /* Load the input image data. */
    unsigned char* inputImage = NULL;
    int width, height;
    Mat iImage;
    iImage = imread("/data/local/input.jpg");
    width = iImage.cols;
    height = iImage.rows;
    inputImage = iImage.data;

    /*
     * Calculate the width and height of the new image.
     * Used to allocate the correct amount of output memory and the number of kernels to use.
     */
    int newWidth = width * scaleFactor;
    int newHeight = height * scaleFactor;

    /* [Allocate image objects] */
    /*
     * Specify the format of the image.
     * The bitmap image we are using is RGB888, which is not a supported OpenCL image format.
     * We will use RGBA8888 and add an empty alpha channel.
     */
    cl_image_format format;
    format.image_channel_data_type = CL_UNORM_INT8;
    format.image_channel_order = CL_RGBA;

    /* Allocate memory for the input image that can be accessed by the CPU and GPU. */
    bool createMemoryObjectsSuccess = true;

    memoryObjects[0] = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, width, height, 0, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);

    memoryObjects[1] = clCreateImage2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, &format, newWidth, newHeight, 0, NULL, &errorNumber);
    createMemoryObjectsSuccess &= checkSuccess(errorNumber);

    if (!createMemoryObjectsSuccess) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed creating the image. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    /* [Allocate image objects] */

    /* [Map image objects to host pointers] */
    /*
     * Like with memory buffers, we now map the allocated memory to a host side pointer.
     * Unlike buffers, we must specify origin coordinates, width and height for the region of the image we wish to map.
     */
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {width, height, 1};

    /*
     * clEnqueueMapImage also returns the rowPitch; the width of the mapped region in bytes.
     * If the image format is not known, this is required information when accessing the image object as a normal array.
     * The number of bytes per pixel can vary with the image format being used,
     * this affects the offset into the array for a given coordinate.
     * In our case the image format is fixed as RGBA8888 so we don't need to worry about the rowPitch.
     */
    size_t rowPitch;
    start = getTimestamp();
    unsigned char* inputImageRGBA = (unsigned char*)clEnqueueMapImage(commandQueue,  memoryObjects[0],
                CL_TRUE, CL_MAP_WRITE, origin, region, &rowPitch, NULL, 0, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed mapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    /* [Map image objects to host pointers] */

    /* Convert the input data from RGB to RGBA (moves it to the OpenCL allocated memory at the same time). */
    RGBToRGBA(inputImage, inputImageRGBA, width, height);

    // Mat srcImage = Mat(height, width, CV_8UC4, inputImageRGBA);
    // Mat dstImage;
    // Size size(newWidth, newHeight);
    // start = getTimestamp();
    // resize(srcImage, dstImage, size);
    // end = getTimestamp();
    // float cpuTook = (end-start)*1e-3;
    // printf("OpenCV(CPU) scaling took %.2lfms\n", cpuTook);
    // imwrite("/data/local/output_cv.jpg", dstImage);

    /* Unmap the image from the host. */
    if (!checkSuccess(clEnqueueUnmapMemObject(commandQueue, memoryObjects[0], inputImageRGBA, 0, NULL, NULL))) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed unmapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /*
     * Calculate the normalization factor for the image coordinates.
     * By using normalized coordinates we don't have to manually map the destination coordinates to the source coordinates.
     */
    cl_float widthNormalizationFactor = 1.0f / newWidth;
    cl_float heightNormalizationFactor = 1.0f / newHeight;

    /* Setup the kernel arguments. */
    bool setKernelArgumentsSuccess = true;
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 0, sizeof(cl_mem), &memoryObjects[0]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 1, sizeof(cl_mem), &memoryObjects[1]));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 2, sizeof(cl_float), &widthNormalizationFactor));
    setKernelArgumentsSuccess &= checkSuccess(clSetKernelArg(kernel, 3, sizeof(cl_float), &heightNormalizationFactor));
    if (!setKernelArgumentsSuccess) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, 3);
        cerr << "Failed setting OpenCL kernel arguments. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /*
     * Set the kernel work size. Each kernel operates on one pixel of the ouput image.
     * Therefore, we need newWidth * newHeight kernel instances.
     * We are using two work dimensions because it maps nicely onto the coordinates of the image.
     * With one dimension we would have to derive the y coordinate from the x coordinate in the kernel.
     */
    const int workDimensions = 2;
    size_t globalWorkSize[workDimensions] = {newWidth, newHeight};

    /* An event to associate with the kernel. Allows us to retrieve profiling information later. */
    cl_event event = 0;

    //start = getTimestamp();
    /* Enqueue the kernel. */
    if (!checkSuccess(clEnqueueNDRangeKernel(commandQueue, kernel, workDimensions, NULL, globalWorkSize, NULL, 0, NULL, &event))) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed enqueuing the kernel. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    /* Wait for kernel execution completion. */
    if (!checkSuccess(clFinish(commandQueue))) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed waiting for kernel execution to finish. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }
    end = getTimestamp();
    float gpuTook = (end-start)*1e-3;
    printf("OpenCL(GPU) scaling took %.2lfms\n", gpuTook);

    /* Release the event object. */
    if (!checkSuccess(clReleaseEvent(event))) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed releasing the event object. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    size_t newRegion[3] = {newWidth, newHeight, 1};

    unsigned char* outputImage = (unsigned char*)clEnqueueMapImage(commandQueue,  memoryObjects[1], CL_TRUE, CL_MAP_READ, origin, newRegion, &rowPitch, NULL, 0, NULL, NULL, &errorNumber);
    if (!checkSuccess(errorNumber)) {
        cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);
        cerr << "Failed mapping the input image. " << __FILE__ << ":"<< __LINE__ << endl;
        return 1;
    }

    unsigned char* outputImageRGB = new unsigned char[newWidth * newHeight * 3];
    RGBAToRGB(outputImage, outputImageRGB, newWidth, newHeight);

    Mat oImage = Mat(newHeight, newWidth, CV_8UC3, outputImageRGB);
    imwrite("/data/local/output_cl.jpg", oImage);

    delete[] outputImageRGB;

    cleanUpOpenCL(context, commandQueue, program, kernel, memoryObjects, numMemoryObjects);

    printf("CPU/GPU scaling took factor 72.00ms/%.2lfms = %.2lf\n", gpuTook, 72/gpuTook);

    return 0;
}
