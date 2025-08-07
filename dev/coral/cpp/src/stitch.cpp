//#include "stitch.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <map>
#include <iomanip>
#include <sstream>
#include <cstdlib>

// All scene info
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

std::string sceneFolderName(int sceneId) {
    if (sceneId == 3052) return "scene_03052_output";
    std::ostringstream oss;
    oss << "scene_" << sceneId << "_output";
    return oss.str();
}

const std::pair<int, int>& getSceneGridSize(const std::string& sceneIdStr) {
    const auto& sceneGridSizes = getSceneGridSizes();
    int sceneId = std::stoi(sceneIdStr);
    auto it = sceneGridSizes.find(sceneId);
    if (it != sceneGridSizes.end()) {
        return it->second;
    }
    throw std::runtime_error("Scene ID not found in grid size map!");
}

std::vector<float> readBinFile(const std::string& filePath, int numElements) {
    std::vector<float> data(numElements);
    std::ifstream fin(filePath, std::ios::binary);
    if (!fin) throw std::runtime_error("Failed to open file " + filePath);
    fin.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
    return data;
}

void writeBinFile(const std::string& filePath, const std::vector<float>& data) {
    std::ofstream fout(filePath, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

std::map<int, std::string> getPatchFiles(const std::string& folder) {
    std::map<int, std::string> patchFiles;
    DIR* dir = opendir(folder.c_str());
    if (!dir) throw std::runtime_error("Failed to open folder: " + folder);

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName(entry->d_name);
        if (fileName.length() > 4 && fileName.substr(fileName.length()-4) == ".bin") {
            size_t p = fileName.find("patch_");
            if (p != std::string::npos) {
                size_t p2 = fileName.find("_", p + 6);
                if (p2 != std::string::npos) {
                    int patchNum = std::stoi(fileName.substr(p + 6, p2 - (p + 6)));
                    patchFiles[patchNum] = fileName;
                }
            }
        }
    }
    closedir(dir);
    return patchFiles;
}

int stitchPatches(
    const std::string& folder,
    const std::string& sceneIdStr,
    int patchHeight, int patchWidth,
    const std::string& outputFile
) {
    try {
        auto [cols, rows] = getSceneGridSize(sceneIdStr);
        std::cout << "Expecting " << (cols*rows) << " patches. Grid: " << cols << " x " << rows << std::endl;

        int patchElems = patchHeight * patchWidth;
        int outHeight = rows * patchHeight;
        int outWidth  = cols * patchWidth;
        int outElems  = outHeight * outWidth;

        std::vector<float> stitched(outElems, 0.0f);

        auto patchFiles = getPatchFiles(folder);

        // Check patch count
        if (static_cast<int>(patchFiles.size()) != cols * rows) {
            std::cerr << "Error: Found " << patchFiles.size()
                      << " patch files, but expected " << (cols * rows) << "!\n";
            return 1;
        }

        for (int patchNum = 1; patchNum <= cols * rows; ++patchNum) {
            if (patchFiles.count(patchNum) == 0) {
                std::cerr << "Missing patch: " << patchNum << std::endl;
                continue;
            }
            std::string inPath = folder + "/" + patchFiles[patchNum];
            auto patchData = readBinFile(inPath, patchElems);

            int row = (patchNum - 1) / cols;
            int col = (patchNum - 1) % cols;

            for (int y = 0; y < patchHeight; ++y) {
                for (int x = 0; x < patchWidth; ++x) {
                    int patchIdx = y * patchWidth + x;
                    int imgY = row * patchHeight + y;
                    int imgX = col * patchWidth + x;
                    int stitchedIdx = imgY * outWidth + imgX;
                    stitched[stitchedIdx] = patchData[patchIdx];
                }
            }
        }

        writeBinFile(outputFile, stitched);
        std::cout << "Stitched image written to: " << outputFile << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR in stitchPatches: " << ex.what() << std::endl;
        return 1;
    }
}

int main() {
    // [interactive picking code as before]
    // ... same as previous version up to and including stitching ...
    const auto& sceneGridSizes = getSceneGridSizes();

    std::cout << "Available Scenes:\n";
    std::vector<int> sceneIds;
    int idx = 1;
    const int perLine = 4;
    for (const auto& kv : sceneGridSizes) {
        std::cout << std::setw(2) << idx << ". " << kv.first
                  << " (" << kv.second.first << "x" << kv.second.second << ")   ";
        sceneIds.push_back(kv.first);
        if (idx % perLine == 0) std::cout << "\n";
        idx++;
    }
    if ((idx-1) % perLine != 0) std::cout << "\n";

    int pickIdx = 0;
    while (true) {
        std::cout << "Pick a scene by number: ";
        std::cin >> pickIdx;
        if (pickIdx > 0 && pickIdx <= static_cast<int>(sceneIds.size())) break;
        std::cout << "Invalid choice.\n";
    }
    int sceneId = sceneIds[pickIdx - 1];
    auto [cols, rows] = sceneGridSizes.at(sceneId);

    std::string inputRoot  = "/mnt/sdcard/output/patches";
    std::string outputRoot = "/mnt/sdcard/output/stitched";
    std::string inputFolder = inputRoot + "/" + sceneFolderName(sceneId);

    int patchHeight = 192, patchWidth = 192;

    std::ostringstream oss;
    oss << outputRoot << "/stitched_scene_" << sceneId << ".bin";
    std::string outputFile = oss.str();

    std::cout << "Input folder: " << inputFolder << std::endl;
    std::cout << "Output file:  " << outputFile << std::endl;

    int ret = stitchPatches(inputFolder, std::to_string(sceneId), patchHeight, patchWidth, outputFile);
    if (ret != 0) {
        std::cerr << "Stitching failed." << std::endl;
        return 1;
    }

    // After successful stitching:
    std::ostringstream pyCmd;
    pyCmd << "python3 scripts/convBinScene.py " << sceneId;
    std::cout << "Converting bin to PNG with: " << pyCmd.str() << std::endl;
    int pyRet = std::system(pyCmd.str().c_str());
    if (pyRet != 0) {
        std::cerr << "Python conversion failed!" << std::endl;
        return 2;
    }
    std::cout << "Conversion complete for scene " << sceneId << std::endl;

    return 0;
}