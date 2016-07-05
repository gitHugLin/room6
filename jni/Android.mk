LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on
include /Users/linqi/SDKDir/OpenCV-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_LDFLAGS := -Wl,--build-id -llog
LOCAL_LDLIBS := -lGLES_mali
LOCAL_CFLAGS :=  -DSK_SUPPORT_LEGACY_SETCONFIG -mfloat-abi=softfp -mfpu=neon
# -march=armv7-a -mtune=cortex-a8
#TARGET_CFLAGS += -O3

LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_C_INCLUDES += \
	    $(LOCAL_PATH)/../../opencv-sdk/native/jni/include

LOCAL_SRC_FILES := \
	main.cpp \
	wdrOpenCL.cpp

LOCAL_MODULE:= opencl-wdr
LOCAL_SHARED_LIBRARIES := libopencv_java3
include $(BUILD_EXECUTABLE)
