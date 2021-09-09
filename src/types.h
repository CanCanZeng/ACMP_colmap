#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

#define MAX_IMAGES 256

struct Camera {
    float K[9];
    float R[9];  // Rcw
    float t[3];  // tcw
    int height;
    int width;
    float depth_min;
    float depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};


#include "colmap_interface/endian.h"
#include <glog/logging.h>

template <typename T>
void ReadMap(const std::string& path, cv::Mat_<T> &data) {
    std::fstream text_file(path, std::ios::in | std::ios::binary);
    CHECK(text_file.is_open()) << path;

    char unused_char;
    size_t width, height, depth;
    text_file >> width >> unused_char >> height >> unused_char >> depth >>
        unused_char;
    std::streampos pos = text_file.tellg();
    text_file.close();

    CHECK_GT(width, 0);
    CHECK_GT(height, 0);
    CHECK_EQ(depth, data.channels());
    std::vector<uchar> raw_data(width * height * sizeof (T));

    std::fstream binary_file(path, std::ios::in | std::ios::binary);
    CHECK(binary_file.is_open()) << path;
    binary_file.seekg(pos);
    colmap::ReadBinaryLittleEndian<uchar>(&binary_file, &raw_data);
    binary_file.close();

    size_t one_channel_bytes = width * height * sizeof (T) / depth;
    std::vector<cv::Mat> channels(depth);
    for(int i = 0; i < depth; i++) {
        channels[i] = cv::Mat::zeros(height, width, CV_MAKETYPE(data.depth(), 1));
        memcpy(channels[i].data, raw_data.data() + i * one_channel_bytes, one_channel_bytes);
    }
    cv::merge(channels, data);
}

template <typename T>
void WriteMap(const std::string& path, cv::Mat_<T> &data) {
    size_t width = data.cols, height = data.rows, depth = data.channels();

    std::vector<cv::Mat> channels;
    cv::split(data, channels);

    size_t one_channel_bytes = width * height * sizeof (T) / depth;
    std::vector<uchar> raw_data(width * height * sizeof (T));
    for(size_t i = 0; i < depth; i++) {
        uchar* p_raw_data = &(raw_data[i * one_channel_bytes]);
        memcpy(p_raw_data, channels[i].data, one_channel_bytes);
    }

    std::fstream text_file(path, std::ios::out);
    CHECK(text_file.is_open()) << path;
    text_file << width << "&" << height << "&" << depth << "&";
    text_file.close();

    std::fstream binary_file(path, std::ios::out | std::ios::binary | std::ios::app);
    CHECK(binary_file.is_open()) << path;
    colmap::WriteBinaryLittleEndian<uchar>(&binary_file, raw_data);
    binary_file.close();
}
