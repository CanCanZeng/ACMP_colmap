#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <sys/stat.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <glog/logging.h>

#include "./endian.h"

#include "types.h"

namespace Eigen {
typedef Eigen::Matrix<uint8_t, 3, 1> Vector3ub;
}

namespace colmap {

// See colmap/src/util/types.h
typedef uint32_t camera_t;
typedef uint32_t image_t;
typedef uint64_t image_pair_t;
typedef uint32_t point2D_t;
typedef uint64_t point3D_t;

struct Camera {
    camera_t camera_id;
    int model_id = 1;
    int width, height;
    std::vector<double> params;
};

struct Image {
    std::string name;
    image_t image_id;
    camera_t camera_id;
    Eigen::Vector4d qcw;
    Eigen::Vector3d tcw;
    std::vector<std::pair<Eigen::Vector2d, point3D_t>> keypoints;

    Eigen::Matrix3d Rcw;
    Eigen::Vector3d Project(const Eigen::Vector3d& point3D) const {
        return Rcw * point3D + tcw;
    }

    Eigen::Vector3d GetCenter() const{
        auto twc = -Rcw.transpose() * tcw;
        return twc;
    }
};

struct Point3D {
    Eigen::Vector3d xyz;
    Eigen::Vector3ub color;
    double error;
    std::vector<std::pair<image_t, point2D_t>> track;
};

void ReadCamerasBinary(std::unordered_map<camera_t, Camera>& cameras, const std::string& path);
void ReadImagesBinary(std::unordered_map<image_t, Image>& images, const std::string& path);
void ReadPoints3DBinary(std::unordered_map<point3D_t, Point3D>& points3D, const std::string& path);

void ReadCamerasTxt(std::unordered_map<camera_t, Camera>& cameras, const std::string& path);
void ReadImagesTxt(std::unordered_map<image_t, Image>& images, const std::string& path);
void ReadPoints3DTxt(std::unordered_map<point3D_t, Point3D>& points3D, const std::string& path);
} // namespace colmap

class Model {
public:
    std::unordered_map<int, std::string> image_id_to_image_name;
    std::unordered_map<int, Camera> image_id_to_camera;
    std::unordered_map<int, std::vector<int>> covis_vec;

    std::string root_folder;
    const std::string model_folder = "sparse";
    const std::string depth_folder = "stereo/depth_maps";
    const std::string normal_folder = "stereo/normal_maps";
    const std::string image_folder = "images";

    Model(const std::string& _root_folder, const std::string& _model_folder, const std::string& _depth_folder, const std::string& _normal_folder, const std::string& _image_folder)
        :root_folder(_root_folder), model_folder(_model_folder), depth_folder(_depth_folder), normal_folder(_normal_folder), image_folder(_image_folder)
    {}

    // For each image, determine the maximally overlapping images, sorted based on
    // the number of shared points subject to a minimum robust average
    // triangulation angle of the points.
    std::unordered_map<int, std::vector<int>> GetMaxOverlappingImages(const size_t num_images, const double min_triangulation_angle) const;

    // Compute the number of shared points between all overlapping images.
    std::unordered_map<int, std::map<int, int>> ComputeSharedPoints() const;

    // Compute the median triangulation angles between all overlapping images.
    std::unordered_map<int, std::map<int, double>> ComputeTriangulationAngles(const double percentile = 50) const;

    // Compute the robust minimum and maximum depths from the sparse point cloud.
    std::unordered_map<int, std::pair<double, double>> ComputeDepthRanges() const;

    bool Read();

    // release model contents
    void ReduceMemory();

private:
    std::unordered_map<colmap::camera_t, colmap::Camera> cameras;
    std::unordered_map<colmap::image_t, colmap::Image> images;
    std::unordered_map<colmap::point3D_t, colmap::Point3D> points;
    std::unordered_map<int, std::pair<double, double>> depth_ranges;
};


// ***************************************************
// ************   util functions *********************
// ***************************************************
// ***************************************************

bool ExistsFile(const std::string& path);
std::string GetFolder(std::string name);
void CreateFolder(std::string abspath);   // recursively

template <typename T>
T Percentile(const std::vector<T>& elems, const double p) {
    CHECK(!elems.empty());
    CHECK_GE(p, 0);
    CHECK_LE(p, 100);

    const int idx = static_cast<int>(std::round(p / 100 * (elems.size() - 1)));
    const size_t percentile_idx =
            std::max(0, std::min(static_cast<int>(elems.size() - 1), idx));

    std::vector<T> ordered_elems = elems;
    std::nth_element(ordered_elems.begin(),
                     ordered_elems.begin() + percentile_idx, ordered_elems.end());

    return ordered_elems.at(percentile_idx);
}
