#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <chrono>
#include <string>
#include <sstream>
#include <limits>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/register.h"
#include "edgetpu.h"

// Helper to get input/output quantization params from TFLite
struct QuantParams {
    float scale;
    int zeroPoint;
};

QuantParams getInputQuantParams(std::unique_ptr<tflite::Interpreter>& interpreter) {
    auto* tensor = interpreter->input_tensor(0);
    QuantParams qp;
    qp.scale = tensor->params.scale;
    qp.zeroPoint = tensor->params.zero_point;
    return qp;
}
QuantParams getOutputQuantParams(std::unique_ptr<tflite::Interpreter>& interpreter) {
    auto* tensor = interpreter->output_tensor(0);
    QuantParams qp;
    qp.scale = tensor->params.scale;
    qp.zeroPoint = tensor->params.zero_point;
    return qp;
}

int main() {
    // ---- Interactive Scene Selection ----
    std::string modelRoot = "/home/mendel/BA/dev/coral/shared";
    std::string scenesRoot = "/home/mendel/BA/dev/coral/shared/patches";
    std::cout << "Enter scene ID: ";
    std::string sceneId;
    std::cin >> sceneId;
    std::string inputFolder = scenesRoot + "/scene_" + sceneId + "_bin_patches";
    std::string outputFolder = scenesRoot + "/scene_" + sceneId + "_output";

    // Make sure output dir exists
    mkdir(outputFolder.c_str(), 0777);

    const int batch = 1, height = 192, width = 192, channels = 4;
    const int inputElements = batch * height * width * channels;

    // ---- Load model ----
    auto model = tflite::FlatBufferModel::BuildFromFile((modelRoot + "/quant_edgetpu.tflite").c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    auto tpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpuContext.get());
    interpreter->AllocateTensors();

    QuantParams inputQ = getInputQuantParams(interpreter);
    QuantParams outputQ = getOutputQuantParams(interpreter);

    DIR* dir = opendir(inputFolder.c_str());
    if (!dir) {
        std::cerr << "Failed to open input directory: " << inputFolder << std::endl;
        return 1;
    }

    struct dirent* entry;
    int patchCount = 0;
    double totalMs = 0;

    while ((entry = readdir(dir)) != NULL) {
        std::string fileName(entry->d_name);

        // Only process .bin files (skip . and ..)
        if (fileName.length() > 4 && fileName.substr(fileName.length() - 4) == ".bin") {
            std::string inPath = inputFolder + "/" + fileName;
            std::string outPath = outputFolder + "/" + fileName;

            // ---- Read float32 input ----
            std::vector<float> inputData(inputElements);
            std::ifstream fin(inPath, std::ios::binary);
            if (!fin) {
                std::cerr << "Failed to open input: " << inPath << std::endl;
                continue;
            }
            fin.read(reinterpret_cast<char*>(inputData.data()), inputElements * sizeof(float));

            // ---- Quantize to int8 ----
            std::vector<int8_t> quantInput(inputElements);
            for (int i = 0; i < inputElements; ++i) {
                float normVal = inputData[i];
                int32_t q = static_cast<int32_t>(std::round(normVal / inputQ.scale + inputQ.zeroPoint));
                q = std::min(std::max(q, -128), 127);
                quantInput[i] = static_cast<int8_t>(q);
            }

            // ---- Set input ----
            TfLiteTensor* inputTensor = interpreter->input_tensor(0);
            memcpy(inputTensor->data.int8, quantInput.data(), inputElements);

            // ---- Run inference ----
            auto start = std::chrono::high_resolution_clock::now();
            interpreter->Invoke();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inferenceMs = end - start;
            totalMs += inferenceMs.count();
            patchCount++;
            std::cout << "Patch No:" << patchCount << std::endl;

            // ---- Dequantize output to float32 ----
            TfLiteTensor* outputTensor = interpreter->output_tensor(0);
            int outputElements = outputTensor->bytes / sizeof(int8_t);
            int8_t* outRaw = outputTensor->data.int8;
            std::vector<float> outFloat(outputElements);
            for (int i = 0; i < outputElements; ++i) {
                outFloat[i] = outputQ.scale * (static_cast<int>(outRaw[i]) - outputQ.zeroPoint);
            }

            // ---- Save float32 output ----
            std::ofstream fout(outPath, std::ios::binary);
            fout.write(reinterpret_cast<char*>(outFloat.data()), outputElements * sizeof(float));

            std::cout << fileName << ": " << inferenceMs.count() << " ms" << std::endl;
        }
    }

    closedir(dir);

    std::cout << "Processed " << patchCount << " patches." << std::endl;
    if (patchCount > 0)
        std::cout << "Average inference time: " << (totalMs / patchCount) << " ms" << std::endl;

    return 0; // Avoid EdgeTPU segfault
}