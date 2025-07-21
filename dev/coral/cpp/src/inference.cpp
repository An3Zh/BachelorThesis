#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <dirent.h>
#include <sys/stat.h>
#include <cstring>
#include <chrono>
#include <string>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/register.h"
#include "edgetpu.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bin_folder> <output_bin_folder>" << std::endl;
        return 1;
    }

    std::string inputFolder = argv[1];
    std::string outputFolder = argv[2];

    // Make sure output dir exists
    mkdir(outputFolder.c_str(), 0777);

    const int batch = 1, height = 192, width = 192, channels = 4;
    const int inputElements = batch * height * width * channels;

    // Load model
    auto model = tflite::FlatBufferModel::BuildFromFile("/home/mendel/BA/dev/shared/quant_edgetpu.tflite");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    auto tpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpuContext.get());
    interpreter->AllocateTensors();

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

            // Read input
            std::vector<int8_t> inputData(inputElements);
            std::ifstream fin(inPath, std::ios::binary);
            if (!fin) {
                std::cerr << "Failed to open input: " << inPath << std::endl;
                continue;
            }
            fin.read(reinterpret_cast<char*>(inputData.data()), inputElements);

            // Set input
            TfLiteTensor* inputTensor = interpreter->input_tensor(0);
            memcpy(inputTensor->data.int8, inputData.data(), inputElements);

            // Run inference and measure time
            auto start = std::chrono::high_resolution_clock::now();
            interpreter->Invoke();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> inferenceMs = end - start;
            totalMs += inferenceMs.count();
            patchCount++;
            std::cout << "Patch No:" << patchCount << std::endl;

            // Save output
            TfLiteTensor* outputTensor = interpreter->output_tensor(0);
            int outputBytes = outputTensor->bytes;
            std::vector<int8_t> outputData(outputBytes / sizeof(int8_t));
            memcpy(outputData.data(), outputTensor->data.int8, outputBytes);

            std::ofstream fout(outPath, std::ios::binary);
            fout.write(reinterpret_cast<char*>(outputData.data()), outputBytes);

            std::cout << fileName << ": " << inferenceMs.count() << " ms" << std::endl;
        }
    }

    closedir(dir);

    std::cout << "Processed " << patchCount << " patches." << std::endl;
    if (patchCount > 0)
        std::cout << "Average inference time: " << (totalMs / patchCount) << " ms" << std::endl;

    return 0; // avoid EdgeTPU segfault
}