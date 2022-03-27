#include <jni.h>
#include "TensorflowRunnerJNI.h"
#include "TensorflowRunner.h"
#include <GLES2/gl2.h>

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeInit(JNIEnv* env, jobject obj, jbyteArray data)
{
  jboolean isCopy = false;
  char * cData = (char *)env->GetByteArrayElements(data, &isCopy);
  jsize len    = env->GetArrayLength(data);

  auto* runner = new TensorflowRunner();

  bool res = runner->init(cData, len);
  assert(res);
  env->ReleaseByteArrayElements(data, (jbyte *)cData, JNI_ABORT);
  return (jlong)runner;
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeDestroy(JNIEnv*, jobject, jlong nativeInstance)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  if (runner) {
    runner->destroy();
    delete runner;
  }
}

JNIEXPORT int    JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeInputsNum(JNIEnv*, jobject, jlong nativeInstance)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  return runner ? runner->getInputsNum() : 0;
}

JNIEXPORT jintArray  JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeGetInputsDim(JNIEnv* env, jobject,int index, jlong nativeInstance)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  auto dims = runner ? runner->getInputsDim(index) : std::vector<int>();
  jintArray ret = env->NewIntArray(dims.size());
  env->SetIntArrayRegion (ret, 0, dims.size(), dims.data());
  return ret;
}

JNIEXPORT int    JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeOutputsNum(JNIEnv*, jobject, jlong nativeInstance)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  return runner ? runner->getOutputsNum() : 0;
}

JNIEXPORT jintArray JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeGetOutputsDim(JNIEnv* env, jobject, jlong nativeInstance, int index)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  auto dims = runner ? runner->getOutputsDim(index) : std::vector<int>();
  jintArray ret = env->NewIntArray(dims.size());
  env->SetIntArrayRegion (ret, 0, dims.size(), dims.data());
  return ret;
}

JNIEXPORT void   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeBindInput(JNIEnv*, jobject, jlong nativeInstance, int index, int ssboId)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  if (runner)
      runner->bindInput(index, ssboId);
}

JNIEXPORT void   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeBindOutput(JNIEnv*, jobject, jlong nativeInstance, int index, int ssboId)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  if (runner)
      runner->bindOutput(index, ssboId);
}

JNIEXPORT jboolean   JNICALL Java_org_tensorflow_lite_examples_imagesegmentation_tflite_TensorflowRunner_nativeRun(JNIEnv*, jobject, jlong nativeInstance)
{
  TensorflowRunner* runner = (TensorflowRunner*)nativeInstance;
  return runner ? runner->run() : false;
}
