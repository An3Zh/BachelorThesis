#include <iostream>
#include <memory>
#include <stdexcept>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/register.h" 
#include "edgetpu.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " model_edgetpu.tflite" << std::endl;
        return 1;
    }

    std::cout << "Hi there, debugger!" << std::endl;
    const char* modelPath = argv[1];
    // Load TFLite model
    auto model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    if (!model) {
        std::cerr << "Failed to load model: " << modelPath << std::endl;
        return 2;
    }

    // Register Edge TPU custom op
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    // Build interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to build interpreter." << std::endl;
        return 3;
    }

    // Attach Edge TPU delegate
    auto tpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpuContext.get());

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors (maybe incompatible model?)." << std::endl;
        return 4;
    }

    std::cout << "Model loaded and tensors allocated successfully!" << std::endl;
    return 0;
}