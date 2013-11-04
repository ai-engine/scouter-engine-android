LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

OPENCV_LIB_TYPE:=STATIC
include /Users/mchmasa/Desktop/android/OpenCV-2.4.6-android-sdk/sdk/native/jni/OpenCV.mk

LOCAL_MODULE    := scouterengine
LOCAL_SRC_FILES := scouterengine.cpp

include $(BUILD_SHARED_LIBRARY)
