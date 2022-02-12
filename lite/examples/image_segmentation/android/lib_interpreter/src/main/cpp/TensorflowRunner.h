#ifndef DEMO_TEST_H
#define DEMO_TEST_H

#include <vector>
#include "tflite_gpu_runner.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"

class TensorflowRunner {
    public:
        // Init model from buffer
        bool init(const char * data, int length);
        void destroy();

        // Bind inout/outputs
        void bindInput(int index,  int ssboId);
        void bindOutput(int index, int ssboId);

        // Run network
        bool run();

        // Input/Output dimensions.
        int getInputsNum() const;
        std::vector<int> getInputsDim(int index) const;
        int getOutputsNum() const;
        std::vector<int> getOutputsDim(int index) const;

private:

    std::unique_ptr<tflite::gpu::TFLiteGPURunner> tflite_gpu_runner;
    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    TfLiteDelegate* delegate = nullptr;

};

#endif //DEMO_TEST_H
