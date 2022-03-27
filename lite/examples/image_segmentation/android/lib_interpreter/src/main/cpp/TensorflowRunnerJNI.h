#ifndef DEMO_TESTJNI_H
#define DEMO_TESTJNI_H

#include <jni.h>

#ifdef __cplusplus
    extern "C" {
#endif

    JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeInit(JNIEnv*, jobject, jbyteArray data);

    JNIEXPORT void JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeDestroy(JNIEnv*, jobject, jlong nativeInstance);

    JNIEXPORT int    JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeInputsNum(JNIEnv*, jobject, jlong nativeInstance);
    JNIEXPORT jintArray  JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeGetInputsDim(JNIEnv*, jobject, jlong nativeInstance, int index);

    JNIEXPORT int    JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeOutputsNum(JNIEnv*, jobject, jlong nativeInstance);
    JNIEXPORT jintArray JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeGetOutputsDim(JNIEnv*, jobject, jlong nativeInstance, int index);

    JNIEXPORT void   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeBindInput(JNIEnv*, jobject, jlong nativeInstance, int index, int ssboId);
    JNIEXPORT void   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeBindOutput(JNIEnv*, jobject, jlong nativeInstance, int index, int ssboId);

    JNIEXPORT jboolean   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeRun(JNIEnv*, jobject, jlong nativeInstance);

#ifdef __cplusplus
};
#endif

#endif //DEMO_TESTJNI_H
