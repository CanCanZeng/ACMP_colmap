#include "./colmap_interface.h"

namespace colmap {

bool IsNotWhiteSpace(const int character) {
    return character != ' ' && character != '\n' && character != '\r' &&
            character != '\t';
}

void StringLeftTrim(std::string* str) {
    str->erase(str->begin(),
               std::find_if(str->begin(), str->end(), IsNotWhiteSpace));
}

void StringRightTrim(std::string* str) {
    str->erase(std::find_if(str->rbegin(), str->rend(), IsNotWhiteSpace).base(),
               str->end());
}

void StringTrim(std::string* str) {
    StringLeftTrim(str);
    StringRightTrim(str);
}

void ReadCamerasBinary(std::unordered_map<camera_t, Camera>& cameras, const std::string& path){
    cameras.clear();

    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    cameras.reserve(num_cameras);

    Camera camera;
    for (size_t i = 0; i < num_cameras; ++i) {
        camera.camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        camera.model_id = ReadBinaryLittleEndian<int>(&file);
        camera.width = ReadBinaryLittleEndian<uint64_t>(&file);
        camera.height = ReadBinaryLittleEndian<uint64_t>(&file);
        CHECK(camera.model_id == 1);
        camera.params.resize(4);
        ReadBinaryLittleEndian<double>(&file, &camera.params);
        cameras.emplace(camera.camera_id, camera);
    }
}

void ReadImagesBinary(std::unordered_map<image_t, Image>& images, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
    images.reserve(num_reg_images);

    Image image;
    for (size_t i = 0; i < num_reg_images; ++i) {
        image.image_id = ReadBinaryLittleEndian<image_t>(&file);

        image.qcw[0] = ReadBinaryLittleEndian<double>(&file);
        image.qcw[1] = ReadBinaryLittleEndian<double>(&file);
        image.qcw[2] = ReadBinaryLittleEndian<double>(&file);
        image.qcw[3] = ReadBinaryLittleEndian<double>(&file);
        image.qcw.normalize();
        image.Rcw = Eigen::Quaterniond(image.qcw[0], image.qcw[1], image.qcw[2], image.qcw[3]).matrix();

        image.tcw[0] = ReadBinaryLittleEndian<double>(&file);
        image.tcw[1] = ReadBinaryLittleEndian<double>(&file);
        image.tcw[2] = ReadBinaryLittleEndian<double>(&file);

        image.camera_id = ReadBinaryLittleEndian<camera_t>(&file);

        image.name.clear();
        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                image.name += name_char;
            }
        } while (name_char != '\0');

        const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);
        image.keypoints.resize(num_points2D);
        for(size_t j = 0; j < num_points2D; ++j) {
            const double x = ReadBinaryLittleEndian<double>(&file);
            const double y = ReadBinaryLittleEndian<double>(&file);
            point3D_t point3D_idx = ReadBinaryLittleEndian<point3D_t>(&file);
            image.keypoints[j] = std::pair<Eigen::Vector2d, point3D_t>(Eigen::Vector2d(x, y), point3D_idx);
        }

        images.emplace(image.image_id, image);
    }
}

void ReadPoints3DBinary(std::unordered_map<point3D_t, Point3D>& points3D, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;

    const size_t num_points3D = ReadBinaryLittleEndian<uint64_t>(&file);
    points3D.reserve(num_points3D);

    Point3D point3D;
    for (size_t i = 0; i < num_points3D; ++i) {
        const point3D_t point3D_id = ReadBinaryLittleEndian<point3D_t>(&file);

        point3D.xyz[0] = ReadBinaryLittleEndian<double>(&file);
        point3D.xyz[1] = ReadBinaryLittleEndian<double>(&file);
        point3D.xyz[2] = ReadBinaryLittleEndian<double>(&file);
        point3D.color[0] = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.color[1] = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.color[2] = ReadBinaryLittleEndian<uint8_t>(&file);
        point3D.error = ReadBinaryLittleEndian<double>(&file);

        const size_t track_length = ReadBinaryLittleEndian<uint64_t>(&file);
        point3D.track.resize(track_length);
        for (size_t j = 0; j < track_length; ++j) {
            const image_t image_id = ReadBinaryLittleEndian<image_t>(&file);
            const point2D_t point2D_idx = ReadBinaryLittleEndian<point2D_t>(&file);
            point3D.track[j] = std::pair<image_t, point2D_t>(image_id, point2D_idx);
        }
        point3D.track.shrink_to_fit();

        points3D.emplace(point3D_id, point3D);
    }
}

void ReadCamerasTxt(std::unordered_map<camera_t, Camera>& cameras, const std::string& path) {
    cameras.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        Camera camera;
        camera.params.resize(4);
        std::string model_type;
        line_stream >> camera.camera_id >> model_type;
        CHECK(model_type == "PINHOLE");

        line_stream >> camera.width >> camera.height >> camera.params[0] >> camera.params[1] >> camera.params[2] >> camera.params[3];

        cameras.emplace(camera.camera_id, camera);
    }
}

void ReadImagesTxt(std::unordered_map<image_t, Image>& images, const std::string& path) {
    images.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    constexpr point3D_t kInvalidPoint3DId = std::numeric_limits<point3D_t>::max();
    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream1(line);

        // ID
        std::getline(line_stream1, item, ' ');
        const image_t image_id = std::stoul(item);

        Image image;
        image.image_id = image_id;

        // QVEC (qw, qx, qy, qz)
        std::getline(line_stream1, item, ' ');
        image.qcw[0] = std::stod(item);

        std::getline(line_stream1, item, ' ');
        image.qcw[1] = std::stod(item);

        std::getline(line_stream1, item, ' ');
        image.qcw[2] = std::stod(item);

        std::getline(line_stream1, item, ' ');
        image.qcw[3] = std::stod(item);

        image.qcw.normalize();
        image.Rcw = Eigen::Quaterniond(image.qcw[0], image.qcw[1], image.qcw[2], image.qcw[3]).matrix();

        // TVEC
        std::getline(line_stream1, item, ' ');
        image.tcw[0] = std::stod(item);

        std::getline(line_stream1, item, ' ');
        image.tcw[1] = std::stod(item);

        std::getline(line_stream1, item, ' ');
        image.tcw[2] = std::stod(item);

        // CAMERA_ID
        std::getline(line_stream1, item, ' ');
        image.camera_id = std::stoul(item);

        // NAME
        std::getline(line_stream1, item, ' ');
        image.name = item;

        // POINTS2D
        if (!std::getline(file, line)) {
            break;
        }

        StringTrim(&line);
        std::stringstream line_stream2(line);

        if (!line.empty()) {
            while (!line_stream2.eof()) {
                Eigen::Vector2d point;

                std::getline(line_stream2, item, ' ');
                point.x() = std::stod(item);

                std::getline(line_stream2, item, ' ');
                point.y() = std::stod(item);

                std::getline(line_stream2, item, ' ');
                if (item == "-1") {
                    image.keypoints.emplace_back(point, kInvalidPoint3DId);
                } else {
                    image.keypoints.emplace_back(point, std::stoul(item));
                }
            }
        }

        images.emplace(image.image_id, image);
    }
}

void ReadPoints3DTxt(std::unordered_map<point3D_t, Point3D>& points3D, const std::string& path) {
    points3D.clear();

    std::ifstream file(path);
    CHECK(file.is_open()) << path;

    std::string line;
    std::string item;

    Point3D point3D;
    while (std::getline(file, line)) {
        StringTrim(&line);

        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream line_stream(line);

        // ID
        std::getline(line_stream, item, ' ');
        const point3D_t point3D_id = std::stol(item);

        // XYZ
        std::getline(line_stream, item, ' ');
        point3D.xyz[0] = std::stod(item);

        std::getline(line_stream, item, ' ');
        point3D.xyz[1] = std::stod(item);

        std::getline(line_stream, item, ' ');
        point3D.xyz[2] = std::stod(item);

        // Color
        std::getline(line_stream, item, ' ');
        point3D.color[0] = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.color[1] = static_cast<uint8_t>(std::stoi(item));

        std::getline(line_stream, item, ' ');
        point3D.color[2] = static_cast<uint8_t>(std::stoi(item));

        // ERROR
        std::getline(line_stream, item, ' ');
        point3D.error = std::stod(item);

        // TRACK
        point3D.track.clear();
        while (!line_stream.eof()) {
            std::getline(line_stream, item, ' ');
            StringTrim(&item);
            if (item.empty()) {
                break;
            }
            image_t image_id = std::stoul(item);

            std::getline(line_stream, item, ' ');
            point2D_t point2D_idx = std::stoul(item);

            point3D.track.emplace_back(image_id, point2D_idx);
        }

        point3D.track.shrink_to_fit();
        points3D.emplace(point3D_id, point3D);
    }
}
}  // namespace colmap


bool Model::Read(){
    std::string path = root_folder + "/" + model_folder;
    if (ExistsFile(path + "/cameras.bin") &&
            ExistsFile(path + "/images.bin") &&
            ExistsFile(path + "/points3D.bin")) {
        VLOG(2) << "reading binary model...";
        colmap::ReadCamerasBinary(cameras, path + "/cameras.bin");
        colmap::ReadImagesBinary(images, path + "/images.bin");
        colmap::ReadPoints3DBinary(points, path + "/points3D.bin");
    } else if (ExistsFile(path + "/cameras.txt") &&
               ExistsFile(path + "/images.txt") &&
               ExistsFile(path + "/points3D.txt")) {
        VLOG(2) << "reading txt model...";
        colmap::ReadCamerasTxt(cameras, path + "/cameras.txt");
        colmap::ReadImagesTxt(images, path + "/images.txt");
        colmap::ReadPoints3DTxt(points, path + "/points3D.txt");
    } else {
        LOG(FATAL) << "cameras, images, points3D files do not exist at " << path;
    }

    VLOG(2) << "computing depth_ranges...";

    depth_ranges = ComputeDepthRanges();

    for(auto iter = images.begin(); iter != images.end();) {
        if(!depth_ranges.count(iter->first)) {
            std::cout << "image " << iter->second.name << " has no depth, so it will be erased" << std::endl;
            iter = images.erase(iter);
        } else {
            iter++;
        }
    }

    VLOG(2) << "computing covis_vec...";

    const size_t num_images = 20;
    const double min_triangulation_angle = 2;
    covis_vec = GetMaxOverlappingImages(num_images, min_triangulation_angle);

    for(const auto& pair : images) {
        colmap::Image image = pair.second;
        auto image_id = image.image_id;
        image_id_to_image_name[image_id] = image.name;

        colmap::Camera camera_colmap = cameras.at(image.camera_id);
        Camera camera;
        camera.width = camera_colmap.width;
        camera.height = camera_colmap.height;
        const auto& range = depth_ranges.at(image_id);
        camera.depth_min = range.first;
        camera.depth_max = range.second;

        auto& K = camera.K;
        K[0] = camera_colmap.params[0]; K[1] = 0; K[2] = camera_colmap.params[2];
        K[3] = 0; K[4] = camera_colmap.params[1]; K[5] = camera_colmap.params[3];
        K[6] = 0; K[7] = 0; K[8] = 1;

        auto& Rcw = camera.R;
        Rcw[0] = image.Rcw(0, 0); Rcw[1] = image.Rcw(0, 1); Rcw[2] = image.Rcw(0, 2);
        Rcw[3] = image.Rcw(1, 0); Rcw[4] = image.Rcw(1, 1); Rcw[5] = image.Rcw(1, 2);
        Rcw[6] = image.Rcw(2, 0); Rcw[7] = image.Rcw(2, 1); Rcw[8] = image.Rcw(2, 2);

        auto& tcw = camera.t;
        tcw[0] = image.tcw[0]; tcw[1] = image.tcw[1]; tcw[2] = image.tcw[2];

        image_id_to_camera[image_id] = camera;
    }

    // create folders
    std::unordered_set<std::string> subfolders;
    for(const auto& pair : images) {
        std::string name = pair.second.name;
        std::string subfolder = GetFolder(name);
        if(subfolder != name) {
            subfolders.insert(subfolder);
        }
    }
    subfolders.insert("");
    std::string outputfolder;
    for(const auto& subfolder : subfolders) {
        for(const auto& folder2 : {depth_folder, normal_folder}) {
            outputfolder = root_folder + "/" + folder2 + "/" + subfolder;
            CreateFolder(outputfolder);
        }
    }
    return true;
}


void Model::ReduceMemory(){
    images.clear();
    points.clear();
    cameras.clear();
    depth_ranges.clear();
}


std::unordered_map<int, std::vector<int>> Model::GetMaxOverlappingImages(const size_t num_images, const double min_triangulation_angle) const {
    std::unordered_map<int, std::vector<int>> overlapping_images(images.size());

    const double min_triangulation_angle_rad = min_triangulation_angle * 0.0174532925199432954743716805978692718781530857086181640625;

    const auto shared_num_points = ComputeSharedPoints();

    const double kTriangulationAnglePercentile = 75;
    const auto triangulation_angles = ComputeTriangulationAngles(kTriangulationAnglePercentile);

    for (const auto& pair : images) {
        auto image_id = pair.first;
        if(!shared_num_points.count(image_id) || !triangulation_angles.count(image_id)) {
            std::cout << "image name: " << images.at(image_id).name << std::endl;
            if(depth_ranges.count(image_id)) {
                std::cout << "depth range: " << depth_ranges.at(image_id).first << " to " << depth_ranges.at(image_id).second << std::endl;
            } else {
                std::cerr << "how does this come?" << std::endl;
            }

            continue;
        }
        const auto& shared_images = shared_num_points.at(image_id);
        const auto& overlapping_triangulation_angles = triangulation_angles.at(image_id);

        std::vector<std::pair<int, int>> ordered_images;
        ordered_images.reserve(shared_images.size());
        for (const auto& image : shared_images) {
            if (overlapping_triangulation_angles.at(image.first) >=
                    min_triangulation_angle_rad) {
                ordered_images.emplace_back(image.first, image.second);
            }
        }

        const size_t eff_num_images = std::min(ordered_images.size(), num_images);
        if (eff_num_images < shared_images.size()) {
            std::partial_sort(ordered_images.begin(),
                              ordered_images.begin() + eff_num_images,
                              ordered_images.end(),
                              [](const std::pair<int, int> image1,
                              const std::pair<int, int> image2) {
                return image1.second > image2.second;
            });
        } else {
            std::sort(ordered_images.begin(), ordered_images.end(),
                      [](const std::pair<int, int> image1,
                      const std::pair<int, int> image2) {
                return image1.second > image2.second;
            });
        }

        overlapping_images[image_id].reserve(eff_num_images);
        for (size_t i = 0; i < eff_num_images; ++i) {
            overlapping_images[image_id].push_back(ordered_images[i].first);
        }
    }

    return overlapping_images;
}

std::unordered_map<int, std::map<int, int>> Model::ComputeSharedPoints() const {
    std::unordered_map<int, std::map<int, int>> shared_points(images.size());
    for (const auto& pair : points) {
        const auto& point = pair.second;
        for (size_t i = 0; i < point.track.size(); ++i) {
            const int image_idx1 = point.track[i].first;
            for (size_t j = 0; j < i; ++j) {
                const int image_idx2 = point.track[j].first;
                if (image_idx1 != image_idx2) {
                    shared_points[image_idx1][image_idx2] += 1;
                    shared_points[image_idx2][image_idx1] += 1;
                }
            }
        }
    }
    return shared_points;
}

double CalculateTriangulationAngle(const Eigen::Vector3d& proj_center1,
                                   const Eigen::Vector3d& proj_center2,
                                   const Eigen::Vector3d& point3D) {
    const double baseline_length_squared =
            (proj_center1 - proj_center2).squaredNorm();

    const double ray_length_squared1 = (point3D - proj_center1).squaredNorm();
    const double ray_length_squared2 = (point3D - proj_center2).squaredNorm();

    // Using "law of cosines" to compute the enclosing angle between rays.
    const double denominator =
            2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0) {
        return 0.0;
    }
    const double nominator =
            ray_length_squared1 + ray_length_squared2 - baseline_length_squared;
    const double angle = std::abs(std::acos(nominator / denominator));

    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    return std::min(angle, M_PI - angle);
}

std::unordered_map<int, std::map<int, double>> Model::ComputeTriangulationAngles(
        const double percentile) const {
    std::unordered_map<int, Eigen::Vector3d> proj_centers(images.size());
    for(const auto& pair : images) {
        const auto& image = pair.second;
        Eigen::Vector3d C = image.GetCenter();
        proj_centers[pair.first] = C.cast<double>();
    }

    std::unordered_map<int, std::map<int, std::vector<double>>> all_triangulation_angles(images.size());
    for(const auto& id_point_pair : points) {
        const auto& point = id_point_pair.second;
        for (size_t i = 0; i < point.track.size(); ++i) {
            const int image_id1 = point.track[i].first;
            for (size_t j = 0; j < i; ++j) {
                const int image_id2 = point.track[j].first;
                if (image_id1 != image_id2) {
                    const double angle = CalculateTriangulationAngle(
                                proj_centers.at(image_id1), proj_centers.at(image_id2), point.xyz);
                    all_triangulation_angles[image_id1][image_id2].push_back(angle);
                    all_triangulation_angles[image_id2][image_id1].push_back(angle);
                }
            }
        }
    }

    std::unordered_map<int, std::map<int, double>> triangulation_angles(images.size());
    for(const auto& pair : all_triangulation_angles) {
        auto image_id1 = pair.first;
        const auto& overlapping_images = pair.second;
        for (const auto& image : overlapping_images) {
            triangulation_angles[image_id1].emplace(image.first, Percentile(image.second, percentile));
        }
    }

    return triangulation_angles;
}

std::unordered_map<int, std::pair<double, double> > Model::ComputeDepthRanges() const {
    std::unordered_map<int, std::vector<double>>  depths(images.size());
    for (const auto& pair : points) {
        auto point = pair.second;
        for (const auto& track : point.track) {
            auto image_id = track.first;
            const auto& image = images.at(image_id);
            const double depth = image.Project(point.xyz)[2];
            if (depth > 0) {
                depths[image_id].push_back(depth);
            }
        }
    }

    std::unordered_map<int, std::pair<double, double>> depth_ranges(depths.size());
    for(auto& pair : depths) {
        int image_id = pair.first;
        auto& image_depths = pair.second;
        std::sort(image_depths.begin(), image_depths.end());

        auto& depth_range = depth_ranges[image_id];

        const double kMinPercentile = 0.01;
        const double kMaxPercentile = 0.99;
        depth_range.first = image_depths[image_depths.size() * kMinPercentile];
        depth_range.second = image_depths[image_depths.size() * kMaxPercentile];

        const double kStretchRatio = 0.25;
        depth_range.first *= (1.0 - kStretchRatio);
        depth_range.second *= (1.0 + kStretchRatio);
    }

    return depth_ranges;
}



// ***************************************************
// ************   util functions *********************
// ***************************************************
// ***************************************************


std::string GetFolder(std::string name) {
    std::string folder = name;
    size_t pos = name.find_last_of('/');
    if(pos < name.size()) {
        folder = name.substr(0, pos);
    }
    return folder;
}

bool ExistsFile(const std::string& path) {
    std::ifstream ifs;
    ifs.open(path);
    if(ifs.is_open()) {
        return true;
    } else {
        return false;
    }
}

void CreateFolder(std::string abspath) {
    if(abspath[abspath.size() - 1] == '/') {
        abspath = abspath.substr(0, abspath.size() - 1);
    }

    struct stat info;
    std::string parent_path = GetFolder(abspath);
    if(parent_path == abspath) {
        std::cout << "Warn: this should not happen in CreateFolder " << abspath << std::endl;
        return;
    }

    if(stat(parent_path.c_str(), &info) != 0) {
        CreateFolder(parent_path);
    }

    mode_t target_mode = 0777;
    if (mkdir(abspath.c_str(), target_mode) == 0)
        chmod(abspath.c_str(), target_mode);
}
