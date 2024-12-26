#include <malloc.h>

#include "colmap_interface/colmap_interface.h"
#include "colmap_interface/datacache.h"

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly = fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %lu\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
    float buffer_vertex[6];
    char buffer_color[3];
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        const float3& X = p.coord;
        const float3& normal = p.normal;
        const float3& color = p.color;
        buffer_vertex[0] = X.x;
        buffer_vertex[1] = X.y;
        buffer_vertex[2] = X.z;
        buffer_vertex[3] = normal.x;
        buffer_vertex[4] = normal.y;
        buffer_vertex[5] = normal.z;

        buffer_color[0] = color.z;
        buffer_color[1] = color.y;
        buffer_color[2] = color.x;

        fwrite(buffer_vertex, sizeof(buffer_vertex), 1, outputPly);
        fwrite(buffer_color, sizeof(buffer_color), 1, outputPly);
    }
    fclose(outputPly);
    std::cout << "save ply file done" << std::endl;
}


float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

cv::Vec3f TransformNormal(const Camera& camera, cv::Vec3f normal_local)
{
    cv::Vec3f normal_global;
    normal_global[0] = camera.R[0] * normal_local[0] + camera.R[3] * normal_local[1] + camera.R[6] * normal_local[2];
    normal_global[1] = camera.R[1] * normal_local[0] + camera.R[4] * normal_local[1] + camera.R[7] * normal_local[2];
    normal_global[2] = camera.R[2] * normal_local[0] + camera.R[5] * normal_local[1] + camera.R[8] * normal_local[2];
    return normal_global;
}

void RunFusion(const Model& model, bool geom_consistency)
{
    std::unordered_map<int, cv::Mat> images;
    std::unordered_map<int, Camera> cameras;
    std::unordered_map<int, cv::Mat_<float>> depths;
    std::unordered_map<int, cv::Mat_<cv::Vec3f>> normals;
    std::unordered_map<int, cv::Mat> masks;
    std::unordered_map<int, cv::Mat> sky_masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    // read all data
    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::string image_name = model.image_idx_to_image_name.at(ref_image_id);
        std::string image_path = model.root_folder + "/" + model.image_folder + "/" + image_name;
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path, cv::IMREAD_COLOR);

        Camera camera = model.image_idx_to_camera.at(ref_image_id);

        std::string suffix = ".photometric.bin";
        if (geom_consistency) {
            suffix = ".geometric.bin";
        }
        std::string depth_path = model.root_folder + "/" + model.depth_folder + "/" + image_name + suffix;
        std::string normal_path = model.root_folder + "/" + model.normal_folder + "/" + image_name + suffix;
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;

        if (!ExistsFile(depth_path)) {
            std::cerr << "Warnning: file does not exist: " << depth_path << std::endl;
            continue;
        }

        if (!ExistsFile(normal_path)) {
            std::cerr << "Warnning: file does not exist: " << normal_path << std::endl;
            continue;
        }

        ReadMap(depth_path, depth);
        ReadMap(normal_path, normal);

        cv::Mat depth_map_cv = depth;
        cv::Mat normal_map_cv = normal;

        std::string sky_mask_path = model.root_folder + "/skymask/" + image_name;
        cv::Mat sky_mask = cv::imread(sky_mask_path, cv::IMREAD_UNCHANGED);
        if (!sky_mask.empty()) {
            if (sky_mask.rows != depth.rows || sky_mask.cols != depth.cols) {
                cv::resize(sky_mask, sky_mask, depth.size());
            }
        }
        sky_masks[ref_image_id] = sky_mask;

//         {
//             cv::Mat depthmap = depth.clone();
//             depth_path = model.root_folder + "/refined_depth_maps/" + image_name + ".pfm";
//             depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
//             depthmap = depth.clone();
// //            depthmap = depthmap * 10.0;
//         }

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images[ref_image_id] = scaled_image;
        cameras[ref_image_id] = camera;
        depths[ref_image_id] = depth;
        normals[ref_image_id] = normal;
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks[ref_image_id] = mask;
    }

    std::vector<PointList> PointCloud;
    // std::vector<std::vector<int>> fused_points_visibility;
    PointCloud.clear();

    int fused_image_idx = 1;
    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::vector<int> src_image_ids = pair.second;
        std::cout << "Fusing image " << model.image_idx_to_image_name.at(ref_image_id) << "    " << fused_image_idx++ << "/" << model.covis_vec.size();

        if (!depths.count(ref_image_id) or !normals.count(ref_image_id)) {
            continue;
        }

        const auto& ref_image = images.at(ref_image_id);
        const auto& ref_camera = cameras.at(ref_image_id);
        const auto& ref_depthmap = depths.at(ref_image_id);
        const auto& ref_normalmap = normals.at(ref_image_id);
        auto& ref_mask = masks.at(ref_image_id);
        const auto& ref_sky_mask = sky_masks.at(ref_image_id);

        const int cols = ref_depthmap.cols;
        const int rows = ref_depthmap.rows;
        int num_ngb = src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (ref_mask.at<uchar>(r, c) > 128)
                    continue;
                if (!ref_sky_mask.empty() && ref_sky_mask.at<uchar>(r, c)) {
                    continue;
                }
                float ref_depth = ref_depthmap.at<float>(r, c);
                if (ref_depth <= 0.0)
                    continue;

                cv::Vec3f ref_normal = ref_normalmap.at<cv::Vec3f>(r, c);
                ref_normal = TransformNormal(ref_camera, ref_normal);

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, ref_camera);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                cv::Vec3b ref_color = ref_image.at<cv::Vec3b>(r, c);
                float consistent_Color[3] = {(float)ref_color[0], (float)ref_color[1], (float)ref_color[2]};
                int num_consistent = 0;

                std::unordered_set<int> fused_point_visibility;
                fused_point_visibility.insert(ref_image_id);

                for (int j = 0; j < num_ngb; ++j) {
                    int src_image_id = src_image_ids[j];
                    if (!depths.count(src_image_id) or !normals.count(src_image_id)) {
                        continue;
                    }
                    const auto& src_image = images.at(src_image_id);
                    const auto& src_camera = cameras.at(src_image_id);
                    const auto& src_depthmap = depths.at(src_image_id);
                    const auto& src_normalmap = normals.at(src_image_id);
                    const auto& src_mask = masks.at(src_image_id);
                    const auto& src_sky_mask = sky_masks.at(src_image_id);

                    const int src_cols = src_depthmap.cols;
                    const int src_rows = src_depthmap.rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, src_camera, point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (src_mask.at<uchar>(src_r, src_c) > 128)
                            continue;

                        if (!src_sky_mask.empty() && src_sky_mask.at<uchar>(src_r, src_c)) {
                            continue;
                        }

                        float src_depth = src_depthmap.at<float>(src_r, src_c);
                        cv::Vec3f src_normal = src_normalmap.at<cv::Vec3f>(src_r, src_c);
                        src_normal = TransformNormal(src_camera, src_normal);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, src_camera);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, ref_camera, tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        constexpr float angle_thresh = 10 * M_PI / 180;
                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < angle_thresh) {
                            consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += src_image.at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += src_image.at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += src_image.at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            fused_point_visibility.insert(src_image_id);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 4) {
                    consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[0] /= (num_consistent + 1.0f);
                    consistent_Color[1] /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);
                    // fused_points_visibility.emplace_back(fused_point_visibility.begin(), fused_point_visibility.end());

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks.at(src_image_ids[j]).at<uchar>(used_list[j].y, used_list[j].x) = 255;
                    }

                    ref_mask.ptr<uchar>(r)[c] = 255;
                }
            }
        }
        std::cout << " (" << PointCloud.size() << " points)" << std::endl;

        std::string depth_valid_mask_path = model.root_folder + "/depth_valid_mask/" + model.image_idx_to_image_name.at(ref_image_id) + ".png";;
        CreateFolder(GetFolder(depth_valid_mask_path));
        cv::imwrite(depth_valid_mask_path, ref_mask);
    }

    // release memory
    {
        images.clear();
        cameras.clear();
        depths.clear();
        normals.clear();
        malloc_trim(0);
        std::cout << "realease memory done" << std::endl;
    }

    std::string ply_path = model.root_folder + "/fused4.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
    // WritePointsVisibility(ply_path + ".vis", fused_points_visibility);
}

void RunFusion2(const Model& model, bool geom_consistency, DataCache& datacache)
{
    std::unordered_map<int, cv::Mat> masks;
    masks.clear();

    std::string suffix = ".photometric.bin";
    if (geom_consistency) {
        suffix = ".geometric.bin";
    }

    // read all data
    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::string image_name = model.image_idx_to_image_name.at(ref_image_id);
        std::string depth_path = model.root_folder + "/" + model.depth_folder + "/" + image_name + suffix;
        cv::Mat_<float> depth;

        size_t w, h, d;
        ReadMapHead<float>(w, h, d, depth_path);

        cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
        masks[ref_image_id] = mask;
    }

    std::vector<PointList> PointCloud;
//    std::vector<std::vector<int>> fused_points_visibility;
    std::vector<int*> fused_points_visibility;
    PointCloud.clear();
    size_t vis_memory_size = 0;

    int fused_image_idx = 1;
    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::vector<int> src_image_ids = pair.second;
        std::cout << "Fusing image " << model.image_idx_to_image_name.at(ref_image_id) << "    " << fused_image_idx++ << "/" << model.covis_vec.size() << std::endl;

        cv::Mat_<cv::Vec3b> ref_image;
        Camera ref_camera;
        cv::Mat_<float> ref_depthmap;
        cv::Mat_<cv::Vec3f> ref_normalmap;
        const auto& ref_mask = masks.at(ref_image_id);
        datacache.GetData(ref_image_id, ref_camera, ref_image, ref_depthmap, ref_normalmap, false);

        const int cols = ref_depthmap.cols;
        const int rows = ref_depthmap.rows;
        int num_ngb = src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));

        std::vector<Camera> src_cameras(num_ngb);
        std::vector<cv::Mat_<cv::Vec3b>> src_images(num_ngb);
        std::vector<cv::Mat_<float>> src_depths(num_ngb);
        std::vector<cv::Mat_<cv::Vec3f>> src_normals(num_ngb);
        for (int j = 0; j < num_ngb; j++) {
            int src_image_id = src_image_ids[j];
            datacache.GetData(src_image_id, src_cameras[j], src_images[j], src_depths[j], src_normals[j], false);
        }

        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (ref_mask.at<uchar>(r, c) > 128)
                    continue;
                float ref_depth = ref_depthmap.at<float>(r, c);
                if (ref_depth <= 0.0)
                    continue;

                cv::Vec3f ref_normal = ref_normalmap.at<cv::Vec3f>(r, c);
                ref_normal = TransformNormal(ref_camera, ref_normal);

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, ref_camera);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                cv::Vec3b ref_color = ref_image.at<cv::Vec3b>(r, c);
                float consistent_Color[3] = {(float)ref_color[0], (float)ref_color[1], (float)ref_color[2]};
                int num_consistent = 0;

                std::unordered_set<int> fused_point_visibility;
                fused_point_visibility.insert(ref_image_id);

                for (int j = 0; j < num_ngb; ++j) {
                    int src_image_id = src_image_ids[j];
                    const auto& src_image = src_images[j];
                    const auto& src_camera = src_cameras[j];
                    const auto& src_depthmap = src_depths[j];
                    const auto& src_normalmap = src_normals[j];
                    const auto& src_mask = masks.at(src_image_id);

                    const int src_cols = src_depthmap.cols;
                    const int src_rows = src_depthmap.rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, src_camera, point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (src_mask.at<uchar>(src_r, src_c) > 128)
                            continue;

                        float src_depth = src_depthmap.at<float>(src_r, src_c);
                        cv::Vec3f src_normal = src_normalmap.at<cv::Vec3f>(src_r, src_c);
                        src_normal = TransformNormal(src_camera, src_normal);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, src_camera);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, ref_camera, tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += src_image.at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += src_image.at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += src_image.at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            fused_point_visibility.insert(src_image_id);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 3) {
                    consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[0] /= (num_consistent + 1.0f);
                    consistent_Color[1] /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    {
                        std::vector<int> vis(fused_point_visibility.begin(), fused_point_visibility.end());
//                        int* buffer = new int[vis.size()];
//                        memcpy(buffer, vis.data(), vis.size() * sizeof (int));
//                        fused_points_visibility.push_back(buffer);
                        vis_memory_size += vis.size() * sizeof (int);
                    }

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks.at(src_image_ids[j]).at<uchar>(used_list[j].y, used_list[j].x) = 255;
                    }
                }
            }
        }
//        std::cout << " (" << PointCloud.size() << " points)" << std::endl;

//        size_t point_memory_size = PointCloud.size() * sizeof(PointList);
//        std::cout << "point memory: " << point_memory_size << "    vis memory: " << vis_memory_size << std::endl;
    }

    // release memory
    {
        datacache.images_map_.clear();
        datacache.cameras_map_.clear();
        datacache.depths_map_.clear();
        datacache.normals_map_.clear();

//        for (size_t k = 0; k < fused_points_visibility.size(); k++) {
//            delete fused_points_visibility[k];
//        }
        malloc_trim(0);
        std::cout << "realease memory done" << std::endl;
    }

    std::string ply_path = model.root_folder + "/fused2.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
//    WritePointsVisibility(ply_path + ".vis", fused_points_visibility);

    // save masks
    for (const auto& pair : masks) {
        int id = pair.first;
        cv::Mat mask = pair.second;
        std::string image_name = model.image_idx_to_image_name.at(id);
        std::string mask_path = model.root_folder + "/depth_valid_mask/" + image_name + ".png";
        std::string subfolder = GetFolder(mask_path);
        CreateFolder(subfolder);
        cv::imwrite(mask_path, mask);
    }
}

int main(int argc, char** argv)
{
    // if (1){
    //     cv::Mat a = cv::imread("/media/data/zengcancan/Project/EasyAR/custom_tools/build/Release/depth.pfm", -1);
    //     a.setTo(0, a > 1e10);
    //     cv::Mat_<float> depth_est;
    //     ReadMap("/media/data/Dataset/Ours/HuiJian/芯桥智城芯智坊/2024-06-20_11.43.59_result4/undistorted/openmvs/stereo/depth_maps/cam3/00002.jpg.geometric.bin", depth_est);

    //     cv::Mat depth_map_est = depth_est;

    //     cv::Mat_<cv::Vec3f> normal_est;
    //     ReadMap("/media/data/Dataset/OpenSource/TankAndTemples/TNT_GOF/TrainingSet/Truck/stereo/normal_maps/000001.jpg.geometric.bin", normal_est);
    //     cv::Mat normal_map_est = normal_est;

    //     cv::Mat depth_map_gt = cv::imread("/media/data/Dataset/OpenSource/GTAV_720_dtutype/0000/depths/0000.pfm", cv::IMREAD_UNCHANGED);
    //     cv::Mat diff;
    //     // cv::absdiff(depth_map_est, depth_map_gt, diff);
    //     std::cout << "" << std::endl;
    //     return 0;
    // }

//    {
//        std::string dense_folder = "/media/data/Dataset/Ours/ARKit/JapanComputerVisionCorp-LiangCha/3/colmap_type/dense2";

//        std::vector<cv::String> image_names;
//        cv::glob(dense_folder + "/images", image_names);

//        for (auto image_name : image_names) {
//            image_name = image_name.substr((dense_folder + "/images/").size());
//            std::string depth_path = dense_folder + "/stereo/depth_maps/" + image_name + ".geometric.bin";
//            cv::Mat_<float> depth_original;
//            ReadMap(depth_path, depth_original);

//            cv::Mat depth = depth_original;

////            int output_size = std::min(depth.rows, depth.cols);
////            int top = depth.rows / 2 - output_size / 2;
////            int left = depth.cols / 2 - output_size / 2;

////            depth = depth.rowRange(top, top + output_size).colRange(left, left + output_size).clone();

//            cv::imwrite(dense_folder + "/depth_maps/" + image_name + ".pfm", depth);

////            depth = depth * 10000;
////            depth.convertTo(depth, CV_16UC1);

////            cv::imwrite("prior_depth.png", depth);


////            std::string prior_depth_mask_path = dense_folder + "/depth_valid_mask/" + image_name + ".png";
////            cv::Mat depth_mask = cv::imread(prior_depth_mask_path, cv::IMREAD_UNCHANGED);

////            depth_mask = depth_mask.rowRange(top, top + output_size).colRange(left, left + output_size).clone();

////            cv::imwrite("prior_depth_map_mask.png", depth_mask);
//        }


//        return 0;
//    }


    double t_start = cv::getTickCount();

    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;
    FLAGS_v = 2;

    if (argc < 2) {
        std::cout << "USAGE: ACMP dense_folder" << std::endl;
        return -1;
    }

    std::string dense_folder = argv[1];
    Model model(dense_folder, "sparse", "stereo/depth_maps", "stereo/normal_maps", "images");
    model.Read();
    model.ReduceMemory();

    RunFusion(model, true);

//    DataCache datacache(model, ".geometric.bin", 16e9);
//    {
//        std::cout << "PointList size=" << sizeof(PointList) << std::endl;
//    }
//    RunFusion2(model, true, datacache);

    double t_end = cv::getTickCount();
    double t_used = (t_end - t_start) / cv::getTickFrequency() / 60;
    std::cout << "Total time: " << t_used << " min" << std::endl;

    return 0;
}
