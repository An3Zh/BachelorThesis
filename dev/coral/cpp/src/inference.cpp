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
#include <regex>
#include <map>
#include <algorithm>
#include <iomanip>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/register.h"
#include "edgetpu.h"

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

// Map sceneId to grid size (cols, rows)
const std::map<int, std::pair<int, int>>& getSceneGridSizes() {
    static const std::map<int, std::pair<int, int>> sceneGridSizes = {
        {3052,  {20, 21}}, {18008, {24, 24}}, {29032, {21, 21}}, {29041, {21, 21}},
        {29044, {20, 21}}, {32030, {21, 21}}, {32035, {20, 21}}, {32037, {20, 21}},
        {34029, {21, 21}}, {34033, {21, 21}}, {34037, {21, 21}}, {35029, {21, 21}},
        {35035, {20, 21}}, {39035, {20, 21}}, {50024, {21, 21}}, {63012, {23, 23}},
        {63013, {23, 23}}, {64012, {23, 23}}, {64015, {22, 22}}, {66014, {22, 23}}
    };
    return sceneGridSizes;
}

// Utility: get folder name for scene
std::string sceneFolderName(int sceneId) {
    if (sceneId == 3052) {
        return "scene_03052_bin_patches";
    }
    std::ostringstream oss;
    oss << "scene_" << sceneId << "_bin_patches";
    return oss.str();
}

std::vector<float> readBinFile(const std::string& filePath, int numElements) {
    std::vector<float> data(numElements);
    std::ifstream fin(filePath, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open input: " << filePath << std::endl;
    }
    fin.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
    return data;
}

struct infResult {
    int code;
    int sceneId;
    std::string inputFolder;
    int patchCount;
};

infResult inference() {
    // ---- Interactive Scene Selection ----
    std::string modelRoot = "/home/mendel/BA/dev/coral/shared";
    std::string scenesRoot = "/mnt/sdcard/input";
    const auto& sceneGridSizes = getSceneGridSizes();

    // Output available scene IDs in a compact, multi-column format
    std::cout << "Available Scenes:\n";
    std::vector<int> sceneIds;
    int idx = 1;
    const int perLine = 4; // Change this for more/less columns
    for (const auto& kv : sceneGridSizes) {
        std::cout << std::setw(2) << idx << ". " << kv.first 
                  << " (" << kv.second.first << "x" << kv.second.second << ")   ";
        sceneIds.push_back(kv.first);
        if (idx % perLine == 0) std::cout << "\n";
        idx++;
    }
    if ((idx-1) % perLine != 0) std::cout << "\n";


    // Select index
    int pickIdx = 0;
    while (true) {
        std::cout << "Pick a scene by number: ";
        std::cin >> pickIdx;
        if (pickIdx > 0 && pickIdx <= static_cast<int>(sceneIds.size())) break;
        std::cout << "Invalid choice.\n";
    }
    int sceneId = sceneIds[pickIdx - 1];
    auto [cols, rows] = sceneGridSizes.at(sceneId);

    // Build input/output folder
    std::string inputFolder = scenesRoot + "/" + sceneFolderName(sceneId);
    std::string outputFolder = "/mnt/sdcard/output/patches/scene_" + std::to_string(sceneId) + "_output";
    if (sceneId == 3052) {
        outputFolder = "/mnt/sdcard/output/patches/scene_03052_output";
    }
    mkdir(outputFolder.c_str(), 0777);

    // Count .bin files in inputFolder
    DIR* dir = opendir(inputFolder.c_str());
    if (!dir) {
        std::cerr << "Failed to open input directory: " << inputFolder << std::endl;
        return {1, sceneId, inputFolder, 0};
    }
    int fileCount = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName(entry->d_name);
        if (fileName.length() > 4 && fileName.substr(fileName.length() - 4) == ".bin") {
            fileCount++;
        }
    }
    closedir(dir);

    // Must match grid size
    if (fileCount != cols * rows) {
        std::cerr << "Error: Found " << fileCount << " .bin files, but expected " << (cols * rows)
                  << " for scene " << sceneId << " (" << cols << "x" << rows << ")." << std::endl;
        return {2, sceneId, inputFolder, fileCount};
    }
    std::cout << "Scene " << sceneId << ": Found correct number of patches (" << fileCount << ")\n";

    // ---- Model inference ---- (unchanged)
    const int batch = 1, height = 192, width = 192, channels = 4;
    const int inputElements = batch * height * width * channels;
    auto model = tflite::FlatBufferModel::BuildFromFile((modelRoot + "/good.tflite").c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    auto tpuContext = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpuContext.get());
    interpreter->AllocateTensors();

    QuantParams inputQ = getInputQuantParams(interpreter);
    QuantParams outputQ = getOutputQuantParams(interpreter);

    // Re-open dir for iteration
    dir = opendir(inputFolder.c_str());
    if (!dir) {
        std::cerr << "Failed to open input directory (second open): " << inputFolder << std::endl;
        return {3, sceneId, inputFolder, 0};
    }

    int patchCount = 0;
    double totalMs = 0;

    while ((entry = readdir(dir)) != NULL) {
        std::string fileName(entry->d_name);
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

            std::cout << "Patch #" << patchCount << " : " << inferenceMs.count() << " ms" << std::endl;
        }
    }
    closedir(dir);

    std::cout << "Processed " << patchCount << " patches." << std::endl;
    if (patchCount > 0)
        std::cout << "Total inference time: " << (totalMs / 1000) << " s" << std::endl;

    return {0, sceneId, inputFolder, patchCount};
}

int main() {
    auto infRes = inference();
    if (infRes.code == 0) {
        return 0;
    } else {
        std::cerr << "Inference failed for scene: " << infRes.sceneId 
                  << " (folder: " << infRes.inputFolder 
                  << ", files: " << infRes.patchCount << ")" << std::endl;
        return infRes.code;
    }
}