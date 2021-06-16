#include "colmap_interface/colmap_interface.h"

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

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
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
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

void RunFusion(const Model& model, bool geom_consistency)
{
    std::unordered_map<int, cv::Mat> images;
    std::unordered_map<int, Camera> cameras;
    std::unordered_map<int, cv::Mat_<float>> depths;
    std::unordered_map<int, cv::Mat_<cv::Vec3f>> normals;
    std::unordered_map<int, cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();

    // read all data
    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::string image_name = model.image_id_to_image_name.at(ref_image_id);
        std::string image_path = model.root_folder + "/" + model.image_folder + "/" + image_name;
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path, cv::IMREAD_COLOR);

        Camera camera = model.image_id_to_camera.at(ref_image_id);

        std::string suffix = ".photometric.bin";
        if (geom_consistency) {
            suffix = ".geometric.bin";
        }
        std::string depth_path = model.root_folder + "/" + model.depth_folder + "/" + image_name + suffix;
        std::string normal_path = model.root_folder + "/" + model.normal_folder + "/" + image_name + suffix;
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        ReadMap(depth_path, depth);
        ReadMap(normal_path, normal);

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
    PointCloud.clear();

    for (const auto& pair : model.covis_vec) {
        int ref_image_id = pair.first;
        std::vector<int> src_image_ids = pair.second;
        std::cout << "Fusing image " << model.image_id_to_image_name.at(ref_image_id) << std::endl;

        const auto& ref_image = images.at(ref_image_id);
        const auto& ref_camera = cameras.at(ref_image_id);
        const auto& ref_depthmap = depths.at(ref_image_id);
        const auto& ref_normalmap = normals.at(ref_image_id);
        const auto& ref_mask = masks.at(ref_image_id);

        const int cols = ref_depthmap.cols;
        const int rows = ref_depthmap.rows;
        int num_ngb = src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (ref_mask.at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = ref_depthmap.at<float>(r, c);
                cv::Vec3f ref_normal = ref_normalmap.at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, ref_camera);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                cv::Vec3b ref_color = ref_image.at<cv::Vec3b>(r, c);
                float consistent_Color[3] = {(float)ref_color[0], (float)ref_color[1], (float)ref_color[2]};
                int num_consistent = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_image_id = src_image_ids[j];
                    const auto& src_image = images.at(src_image_id);
                    const auto& src_camera = cameras.at(src_image_id);
                    const auto& src_depthmap = depths.at(src_image_id);
                    const auto& src_normalmap = normals.at(src_image_id);
                    const auto& src_mask = masks.at(src_image_id);

                    const int src_cols = src_depthmap.cols;
                    const int src_rows = src_depthmap.rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, src_camera, point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (src_mask.at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = src_depthmap.at<float>(src_r, src_c);
                        cv::Vec3f src_normal = src_normalmap.at<cv::Vec3f>(src_r, src_c);
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

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks.at(src_image_ids[j]).at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = model.root_folder + "/ACMP_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

int main(int argc, char** argv)
{
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
    Model model(dense_folder, "sparse", "stereo/depth_maps", "stereo/normal_maps", "stereo/cost_maps", "images");
    model.Read();
    model.ReduceMemory();

    RunFusion(model, true);

    double t_end = cv::getTickCount();
    double t_used = (t_end - t_start) / cv::getTickFrequency() / 60;
    std::cout << "Total time: " << t_used << " min" << std::endl;

    return 0;
}
