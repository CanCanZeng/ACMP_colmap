#include "ACMP.h"

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

ACMP::ACMP(const std::unordered_map<int, std::string>& _image_id_to_image_name, const std::unordered_map<int, Camera> &_image_id_to_camera, const std::string &_depth_folder, const std::string &_normal_folder, const std::string &_image_folder, const std::string &_cost_folder)
    :image_id_to_image_name(_image_id_to_image_name), image_id_to_camera(_image_id_to_camera), depth_folder(_depth_folder), normal_folder(_normal_folder), cost_folder(_cost_folder), image_folder(_image_folder){}

ACMP::~ACMP()
{
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }

    if (params.planar_prior) {
        delete[] prior_planes_host;
        delete[] plane_masks_host;

        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }
}


float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    return pointX;
}


void ACMP::SetGeomConsistencyParams(bool multi_geometry)
{
    params.geom_consistency = true;
    params.max_iterations = 2;
    if(multi_geometry) {
        params.multi_geometry = true;
    }
}

void ACMP::SetPlanarPriorParams()
{
    params.planar_prior = true;
}

void ACMP::InuputInitialization(const std::string &dense_folder, const Problem &problem)
{
    images.clear();
    cameras.clear();

    // ref image
    std::string image_path = dense_folder + "/" + image_folder + "/" + image_id_to_image_name.at(problem.ref_image_id);
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);

    Camera camera = image_id_to_camera.at(problem.ref_image_id);
    cameras.push_back(camera);

    // target images
    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        std::string image_path = dense_folder + "/" + image_folder + "/" + image_id_to_image_name.at(problem.src_image_ids[i]);
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);

        Camera camera = image_id_to_camera.at(problem.src_image_ids[i]);
        cameras.push_back(camera);
    }

    // Scale cameras and images
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].cols <= params.max_image_size && images[i].rows <= params.max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(params.max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(params.max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }

    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    std::cout << "depthe range: " << params.depth_min << " " << params.depth_max << std::endl;
    params.num_images = (int)images.size();
    std::cout << "num images: " << params.num_images << std::endl;
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();

        std::string suffix = params.multi_geometry ? ".geometric.bin" : ".photometric.bin";
        std::string depth_path = dense_folder + "/" + depth_folder + "/" + image_id_to_image_name.at(problem.ref_image_id) + suffix;
        cv::Mat_<float> ref_depth;
        ReadMap(depth_path, ref_depth);
        depths.push_back(ref_depth);

        size_t num_src_images = problem.src_image_ids.size();
        for (size_t i = 0; i < num_src_images; ++i) {
            std::string depth_path = dense_folder + "/" + depth_folder + "/" + image_id_to_image_name.at(problem.src_image_ids[i]) + suffix;
            cv::Mat_<float> depth;
            ReadMap(depth_path, depth);
            depths.push_back(depth);
        }
    }
}

void ACMP::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem)
{
    num_images = (int)images.size();

    for (int i = 0; i < num_images; ++i) {
        int rows = images[i].rows;
        int cols = images[i].cols;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width)); // Updated by Qingshan 2020-01-15

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        std::string suffix = params.multi_geometry ? ".geometric.bin" : ".photometric.bin";
        std::string depth_path = dense_folder + "/" + depth_folder + "/" + image_id_to_image_name.at(problem.ref_image_id) + suffix;
        std::string normal_path = dense_folder + "/" + normal_folder + "/" + image_id_to_image_name.at(problem.ref_image_id) + suffix;
        std::string cost_path = dense_folder + "/" + cost_folder + "/" + image_id_to_image_name.at(problem.ref_image_id) + suffix;
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        ReadMap(depth_path, ref_depth);
        depths.push_back(ref_depth);
        ReadMap(normal_path, ref_normal);
        ReadMap(cost_path, ref_cost);
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center] = ref_cost(row, col);
            }
        }

        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }
}

void ACMP::CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks)
{
    prior_planes_host = new float4[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * (cameras[0].height * cameras[0].width));

    plane_masks_host = new unsigned int[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    for (int i = 0; i < cameras[0].width; ++i) {
        for (int j = 0; j < cameras[0].height; ++j) {
            int center = j * cameras[0].width + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
        }
    }

    cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
    cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * (cameras[0].height * cameras[0].width), cudaMemcpyHostToDevice);
}

int ACMP::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int ACMP::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat ACMP::GetReferenceImage()
{
    return images[0];
}

float4 ACMP::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float ACMP::GetCost(const int index)
{
    return costs_host[index];
}

float ACMP::GetMinDepth()
{
    return params.depth_min;
}

float ACMP::GetMaxDepth()
{
    return params.depth_max;
}

void ACMP::GetSupportPoints(std::vector<cv::Point>& support2DPoints)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    int center = r * width + c;
                    if (GetCost(center) < 2.0f && min_cost > GetCost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost = GetCost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

std::vector<Triangle> ACMP::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points)
{
    if (points.empty()) {
        return std::vector<Triangle>();
    }

    std::vector<Triangle> results;

    std::vector<cv::Vec6f> temp_results;
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    subdiv2d.getTriangleList(temp_results);

    for (const auto temp_vec : temp_results) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

float4 ACMP::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0]);

    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 = make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float norm2 = sqrt(pow(n4.x, 2) + pow(n4.y, 2) + pow(n4.z, 2));
    if (n4.w < 0) {
        norm2 *= -1;
    }
    n4.x /= norm2;
    n4.y /= norm2;
    n4.z /= norm2;
    n4.w /= norm2;

    return n4;
}

float ACMP::GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y)
{
    return -plane_hypothesis.w * cameras[0].K[0] / ((x - cameras[0].K[2]) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y + cameras[0].K[0] * plane_hypothesis.z);
}
