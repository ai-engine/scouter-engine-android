APP_STL := gnustl_static
APP_CPPFLAGS := -frtti -fexceptions
APP_CPPFLAGS += -O3
APP_PLATFORM := android-14

APP_ABI := armeabi-v7a

#APP_OPTIM	:= debug
APP_OPTIM	:= release

LOCAL_ARM_MODE := arm # armかthumbを指定（デフォルトはthumb）armのほうが高速
LOCAL_ARM_NEON := true


LOCAL_DISABLE_NO_EXECUTE := true  #セキュリティリスクあり