#include "colmap_interface/colmap_interface.h"
#include "ACMP.h"

void CreateProblems(const Model& model, std::vector<Problem>& problems) {
    for (const auto& pair : model.covis_vec) {
        Problem problem;
        problem.src_image_ids.clear();

        problem.ref_image_id = pair.first;
        for (const int id : pair.second) {
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

void ProcessProblem(const Problem &problem, const Model& model, bool geom_consistency, bool planar_prior, bool multi_geometry = false)
{
    std::string dense_folder = model.root_folder;
    std::string imageName = model.image_id_to_image_name.at(problem.ref_image_id);
    std::cout << "Processing image " << imageName << "..." << std::endl;
    cudaSetDevice(0);

    ACMP acmp(model.image_id_to_image_name, model.image_id_to_camera, model.depth_folder, model.normal_folder, model.image_folder, model.cost_folder);
    if (geom_consistency) {
        acmp.SetGeomConsistencyParams(multi_geometry);
    }
    acmp.InuputInitialization(dense_folder, problem);

    acmp.CudaSpaceInitialization(dense_folder, problem);
    acmp.RunPatchMatch();

    const int width = acmp.GetReferenceImageWidth();
    const int height = acmp.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = acmp.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmp.GetCost(center);
        }
    }

    if (planar_prior) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        acmp.SetPlanarPriorParams();

        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;

        acmp.GetSupportPoints(support2DPoints);
        const auto triangles = acmp.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = acmp.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto& triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = dense_folder + "/" + model.depth_folder + "/" + imageName + ".png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto& triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = acmp.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = acmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= acmp.GetMaxDepth() && d >= acmp.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }
        // std::string depth_path = result_folder + "/depths_prior.dmb";
        //  writeDepthDmb(depth_path, priordepths);

        acmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        acmp.RunPatchMatch();

        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = acmp.GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = acmp.GetCost(center);
            }
        }
    }

    std::string suffix = ".photometric.bin";
    if (geom_consistency) {
        suffix = ".geometric.bin";
    }
    std::string depth_path = dense_folder + "/" + model.depth_folder + "/" + imageName + suffix;
    std::string normal_path = dense_folder + "/" + model.normal_folder + "/" + imageName + suffix;
    std::string cost_path = dense_folder + "/" + model.cost_folder + "/" + imageName + suffix;
    WriteMap(depth_path, depths);
    WriteMap(normal_path, normals);
    WriteMap(cost_path, costs);
    std::cout << "Processing image " << imageName << " done!" << std::endl;
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

    std::vector<Problem> problems;
    CreateProblems(model, problems);

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    bool geom_consistency = false;
    bool planar_prior = true;
    for (size_t i = 0; i < num_images; ++i) {
        ProcessProblem(problems[i], model, geom_consistency, planar_prior);
    }

    bool multi_geometry = false;
    int geom_iteration = 2;
    geom_consistency = true;
    planar_prior = false;
    for(int geom_iter = 0; geom_iter < geom_iteration; ++geom_iter) {
        if(geom_iter == 0) {
            multi_geometry = false;
        } else {
            multi_geometry = true;
        }
        for (size_t i = 0; i < num_images; ++i) {
            ProcessProblem(problems[i], model, geom_consistency, planar_prior, multi_geometry);
        }
    }

    double t_end = cv::getTickCount();
    double t_used = (t_end - t_start) / cv::getTickFrequency() / 60;
    std::cout << "Total time: " << t_used << " min" << std::endl;

    return 0;
}
