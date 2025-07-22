#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <map>

// Your getSceneGridSize function
const std::pair<int, int>& getSceneGridSize(const std::string& sceneIdStr) {
    static const std::map<int, std::pair<int, int>> sceneGridSizes = {
        {3052,  {20, 21}}, {18008, {24, 24}}, {29032, {21, 21}}, {29041, {21, 21}},
        {29044, {20, 21}}, {32030, {21, 21}}, {32035, {20, 21}}, {32037, {20, 21}},
        {34029, {21, 21}}, {34033, {21, 21}}, {34037, {21, 21}}, {35029, {21, 21}},
        {35035, {20, 21}}, {39035, {20, 21}}, {50024, {21, 21}}, {63012, {23, 23}},
        {63013, {23, 23}}, {64012, {23, 23}}, {64015, {22, 22}}, {66014, {22, 23}}
    };
    int sceneId = std::stoi(sceneIdStr);
    auto it = sceneGridSizes.find(sceneId);
    if (it != sceneGridSizes.end()) {
        return it->second;
    }
    throw std::runtime_error("Scene ID not found in grid size map!");
}

// Read binary file as floats
std::vector<float> readBinFile(const std::string& filePath, int numElements) {
    std::vector<float> data(numElements);
    std::ifstream fin(filePath, std::ios::binary);
    if (!fin) throw std::runtime_error("Failed to open file " + filePath);
    fin.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
    return data;
}

// Write stitched image to binary file
void writeBinFile(const std::string& filePath, const std::vector<float>& data) {
    std::ofstream fout(filePath, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

// Find all bin files and index by patch number
std::map<int, std::string> getPatchFiles(const std::string& folder) {
    std::map<int, std::string> patchFiles; // patchNum -> filename
    DIR* dir = opendir(folder.c_str());
    if (!dir) throw std::runtime_error("Failed to open folder: " + folder);

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName(entry->d_name);
        // Look for .bin files with pattern patch_{num}_
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

// Main stitching function
void stitchPatches(
    const std::string& folder, 
    const std::string& sceneIdStr,
    int patchHeight, int patchWidth,
    const std::string& outputFile
) {
    auto [cols, rows] = getSceneGridSize(sceneIdStr);
    std::cout << "Expecting " << (cols*rows) << " patches. Grid: " << cols << " x " << rows << std::endl;

    int patchElems = patchHeight * patchWidth;
    int outHeight = rows * patchHeight;
    int outWidth  = cols * patchWidth;
    int outElems  = outHeight * outWidth;

    // Output image (row-major)
    std::vector<float> stitched(outElems, 0.0f);

    auto patchFiles = getPatchFiles(folder);

    for (int patchNum = 1; patchNum <= cols * rows; ++patchNum) {
        if (patchFiles.count(patchNum) == 0) {
            std::cerr << "Missing patch: " << patchNum << std::endl;
            continue; // Or throw, depending on strictness
        }
        // Read patch data
        std::string inPath = folder + "/" + patchFiles[patchNum];
        auto patchData = readBinFile(inPath, patchElems);

        // Calculate grid position
        int row = (patchNum - 1) / cols;
        int col = (patchNum - 1) % cols;

        // Place patch into stitched image
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
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <folder> <sceneId> <patchHeight> <patchWidth> <outputFile>" << std::endl;
        return 1;
    }
    try {
        stitchPatches(argv[1], argv[2], std::stoi(argv[3]), std::stoi(argv[4]), argv[5]);
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}