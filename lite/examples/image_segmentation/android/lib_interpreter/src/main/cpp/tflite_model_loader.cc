// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tflite_model_loader.h"
#include <iostream>

namespace mediapipe {

std::string PathToResourceAsFile(const std::string& path) {
  return path;
}

TfLiteModelPtr TfLiteModelLoader::LoadFromPath(
    const std::string& path) {
  std::string model_path = path;

  model_path = mediapipe::PathToResourceAsFile(model_path);
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (model == nullptr) {
      std::cout << "Failed to load model from path " << model_path;
      return TfLiteModelPtr(nullptr);
  }
  else
      {
      return TfLiteModelPtr(
              model.release(), [](tflite::FlatBufferModel *model) { delete model; });
  }
}

TfLiteModelPtr TfLiteModelLoader::LoadFromMemory(
        const char* data,
        int length) {

    auto model = tflite::FlatBufferModel::BuildFromBuffer(data, length);
    if (model == nullptr) {
        std::cout << "Failed to from buffer ";
        return TfLiteModelPtr(nullptr);
    }
    else
    {
        return TfLiteModelPtr(
                model.release(), [](tflite::FlatBufferModel *model) { delete model; });
    }
}

}  // namespace mediapipe
