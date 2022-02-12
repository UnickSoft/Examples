#include <cstdint>
#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "TensorflowRunner.h"

#include "tensorflow/lite/delegates/gpu/api.h"
// This code should be enabled as soon as TensorFlow version, which mediapipe
// uses, will include this module.
#ifdef __ANDROID__
#include "tensorflow/lite/delegates/gpu/gl/api.h"
#endif

#include <tflite_model_loader.h>

#include "tensorflow/lite/delegates/gpu/delegate.h"

#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/model.h"

#ifdef __ANDROID__
#include "tensorflow/lite/delegates/gpu/cl/api.h"
#endif
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

bool TensorflowRunner::init(const char * data, int length)
{
    auto modelLoaded = mediapipe::TfLiteModelLoader::LoadFromMemory(data, length);
    tflite::gpu::InferenceOptions options;
    bool allow_precision_loss_ = true;
    options.priority1 = allow_precision_loss_
                        ? tflite::gpu::InferencePriority::MIN_LATENCY
                        : tflite::gpu::InferencePriority::MAX_PRECISION;
    options.priority2 = tflite::gpu::InferencePriority::AUTO;
    options.priority3 = tflite::gpu::InferencePriority::AUTO;
    options.usage = tflite::gpu::InferenceUsage::SUSTAINED_SPEED;

    tflite_gpu_runner = std::make_unique<tflite::gpu::TFLiteGPURunner>(options);
    op_resolver = tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();

    const auto& model = *modelLoaded.get();
    auto res = tflite_gpu_runner->InitializeWithModel(model, op_resolver);
    res = tflite_gpu_runner->Build();
    return res;
}

void TensorflowRunner::destroy()
{
    tflite_gpu_runner.reset();
}

bool TensorflowRunner::run()
{
  return tflite_gpu_runner->Invoke();
}

int TensorflowRunner::getInputsNum() const
{
    return tflite_gpu_runner ? tflite_gpu_runner->GetInputShapes().size() : 0;
}

std::vector<int> TensorflowRunner::getInputsDim(int index) const
{
    if (!tflite_gpu_runner)
        return std::vector<int>();

    std::vector<int> res;
    auto shape = tflite_gpu_runner->GetInputShapes()[index];
    res.push_back(shape.w);
    res.push_back(shape.h);
    res.push_back(shape.c);
    return res;
}

int TensorflowRunner::getOutputsNum() const
{
    return tflite_gpu_runner ? tflite_gpu_runner->GetOutputShapes().size() : 0;
}

std::vector<int> TensorflowRunner::getOutputsDim(int index) const
{
    if (!tflite_gpu_runner)
        return std::vector<int>();

    std::vector<int> res;
    auto shape = tflite_gpu_runner->GetOutputShapes()[index];
    res.push_back(shape.w);
    res.push_back(shape.h);
    res.push_back(shape.c);
    return res;
}

void TensorflowRunner::bindInput(int index, int ssboId)
{
    if (tflite_gpu_runner)
        tflite_gpu_runner->BindSSBOToInputTensor(ssboId, index);
}

void TensorflowRunner::bindOutput(int index, int ssboId)
{
    if (tflite_gpu_runner)
        tflite_gpu_runner->BindSSBOToOutputTensor(ssboId, index);
}
