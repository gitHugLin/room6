#！/bin/bash
ndk-build
adb root
adb remount
adb push /Users/linqi/WorkDir/OpenCL_Pro/wdr/obj/local/armeabi-v7a/opencl-wdr /system/bin
adb push ./wdr.cl /system/bin
#adb push /Users/linqi/WorkDir/OpenCL_Pro/wdr/obj/local/armeabi-v7a/libopencv_java3.so /system/lib
adb shell chmod 777 /system/bin/opencl-wdr
#adb shell /system/bin/opencl-wdr
#sleep 1
#adb pull /data/local/output_cl.jpg ./pictures/
#adb pull /data/local/output_cv.jpg ./pictures/

#adb logcat *:E
#adb logcat -s MY_LOG_TAG
#adb logcat -c
