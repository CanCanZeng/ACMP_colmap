#include "ACMP.h"
#include <device_functions.h>

__device__  void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(float *d, float *w, int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) {
            d[j] = d[j - 1];
            w[j] = w[j - 1];
        }
        d[j] = tmp;
        w[j] = tmp_w;
    }
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ int FindMaxCostIndex(const float *costs, const int n)
{
    float max_cost = costs[0];
    int max_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] >= max_cost) {
            max_cost = costs[idx];
            max_cost_idx = idx;
        }
    }
    return max_cost_idx;
}

__device__  void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
  result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
  result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
  result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float3 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        const float prob = probs[i] * inv_prob_sum;
        cum_prob += prob;
        probs[i] = cum_prob;
    }
}

__device__ float GetDistance2Origin(const float3 view_dir, const float depth, const float4 normal)
{
    float scale = depth / view_dir.z;
    float x = view_dir.x * scale;
    float y = view_dir.y * scale;
    float z = depth;
    return -(normal.x * x + normal.y * y + normal.z * z);
}

__device__ float ComputeDepthfromPlaneHypothesis(const float3 view_dir, const float4 plane_hypothesis)
{
    return -plane_hypothesis.w * view_dir.z / (view_dir.x * plane_hypothesis.x + view_dir.y * plane_hypothesis.y + view_dir.z * plane_hypothesis.z);
}

__device__ float4 GenerateRandomNormal(const float3 view_dir, curandState *rand_state)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * curand_uniform(rand_state) -1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float dot_product = normal.x * view_dir.x + normal.y * view_dir.y + normal.z * view_dir.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = - normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

__device__ float4 GeneratePerturbedNormal(float3 view_dir, const float4 normal, curandState *rand_state, const float perturbation)
{
    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_dir) >= 0.0f) {
        normal_perturbed = normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ float4 GenerateRandomPlaneHypothesis(const float3 view_dir, curandState *rand_state, const float depth_min, const float depth_max)
{
    float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(view_dir, rand_state);
    plane_hypothesis.w = GetDistance2Origin(view_dir, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const float3 view_dir, curandState *rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max)
{
    float depth_perturbed = depth_now;

    float dist_perturbed = plane_hypothesis_now.w;
    const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
    const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
    float4 plane_hypothesis_temp = plane_hypothesis_now;
    do {
        dist_perturbed = curand_uniform(rand_state) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
        plane_hypothesis_temp.w = dist_perturbed;
        depth_perturbed = ComputeDepthfromPlaneHypothesis(view_dir, plane_hypothesis_temp);
    } while (depth_perturbed < depth_min && depth_perturbed > depth_max);

    float4 plane_hypothesis = GeneratePerturbedNormal(view_dir, plane_hypothesis_now, rand_state, perturbation * M_PI);
    plane_hypothesis.w = dist_perturbed;
    return plane_hypothesis;
}

__device__ void ComputeNormalizedHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H)
{
    float ref_C[3];
    float src_C[3];
    ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
    ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
    ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
    src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
    src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
    src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] *ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] *ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] *ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] *ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] *ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] *ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] *ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] *ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] *ref_camera.R[8];
    C_relative[0] = (ref_C[0] - src_C[0]);
    C_relative[1] = (ref_C[1] - src_C[1]);
    C_relative[2] = (ref_C[2] - src_C[2]);
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

    // H_normalized = R - t * n^t / d
    H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
    H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
    H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
    H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
    H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
    H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
    H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
    H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
    H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;
}

__device__ float3 ComputeCorrespondingPointViewDir(const float *H, const float3 ref_view_dir)
{
    float3 src_view_dir;
    src_view_dir.x = H[0] * ref_view_dir.x + H[1] * ref_view_dir.y + H[2] * ref_view_dir.z;
    src_view_dir.y = H[3] * ref_view_dir.x + H[4] * ref_view_dir.y + H[5] * ref_view_dir.z;
    src_view_dir.z = H[6] * ref_view_dir.x + H[7] * ref_view_dir.y + H[8] * ref_view_dir.z;
    return src_view_dir;
}

__device__ void OpenCVDistort(const float* extra_params, float u, float v, float& du, float& dv) {
    const float k1 = extra_params[0];
    const float k2 = extra_params[1];
    const float p1 = extra_params[2];
    const float p2 = extra_params[3];

    const float u2 = u * u;
    const float uv = u * v;
    const float v2 = v * v;
    const float r2 = u2 + v2;
    const float radial = k1 * r2 + k2 * r2 * r2;
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2);
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2);
}

__device__ void OpenCVFisheyeDistort(const float* extra_params, float u, float v, float& du, float& dv) {
    const float k1 = extra_params[0];
    const float k2 = extra_params[1];
    const float k3 = extra_params[2];
    const float k4 = extra_params[3];

    const float r = sqrt(u * u + v * v);

    if (r > 0.001f) {
      const float theta = atan(r);
      const float theta2 = theta * theta;
      const float theta4 = theta2 * theta2;
      const float theta6 = theta4 * theta2;
      const float theta8 = theta4 * theta4;
      const float thetad =
          theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
      du = u * thetad / r - u;
      dv = v * thetad / r - v;
    } else {
      du = 0;
      dv = 0;
    }
}

__device__ float2 ViewDirToImagePoint(const Camera camera, const float3 view_dir) {
    // pixel coord on normalized plane
    float x = view_dir.x / view_dir.z;
    float y = view_dir.y / view_dir.z;

    float2 pt;
    if (camera.distortion_type == 0) {
        pt.x = camera.K[0] * x + camera.K[2];
        pt.y = camera.K[4] * y + camera.K[5];
    } else if (camera.distortion_type == 1) {
        // Distortion
        float du, dv;
        OpenCVDistort(camera.distortion, x, y, du, dv);

        // Transform to image coordinates
        pt.x = camera.K[0] * (x + du) + camera.K[2];
        pt.y = camera.K[4] * (y + dv) + camera.K[5];
    } else {
        // Distortion
        float du, dv;
        OpenCVFisheyeDistort(camera.distortion, x, y, du, dv);

        // Transform to image coordinates
        pt.x = camera.K[0] * (x + du) + camera.K[2];
        pt.y = camera.K[4] * (y + dv) + camera.K[5];
    }
    return pt;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float3* ref_view_dirs, const float4 plane_hypothesis, const PatchMatchParams params)
{
    const float cost_max = 2.0f;
    int radius = params.patch_size / 2;

    float H[9];
    ComputeNormalizedHomography(ref_camera, src_camera, plane_hypothesis, H);
    float3 src_view_dir = ComputeCorrespondingPointViewDir(H, ref_view_dirs[p.y * ref_camera.width + p.x]);
    float2 pt = ViewDirToImagePoint(src_camera, src_view_dir);
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
        return cost_max;
    }

    float cost = 0.0f;
    {
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        for (int i = -radius; i < radius + 1; i += params.radius_increment) {
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            for (int j = -radius; j < radius + 1; j += params.radius_increment) {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
//                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                float3 src_view_dir = ComputeCorrespondingPointViewDir(H, ref_view_dirs[ref_pt.y * ref_camera.width + ref_pt.x]);
                float2 src_pt = ViewDirToImagePoint(src_camera, src_view_dir);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

                sum_ref_row += weight * ref_pix;
                sum_ref_ref_row += weight * ref_pix * ref_pix;
                sum_src_row += weight * src_pix;
                sum_src_src_row += weight * src_pix * src_pix;
                sum_ref_src_row += weight * ref_pix * src_pix;
                bilateral_weight_sum_row += weight;
            }

            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar) {
            return cost = cost_max;
        } else {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
    }
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float3* ref_view_dirs, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    for (int i = 1; i < params.num_images; ++i) {
        float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, ref_view_dirs, plane_hypothesis, params);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    int top_k = min(num_valid_views, params.top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params.num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                setBit(*selected_views, i);
            }
        }
        return cost / top_k;
    } else {
        return cost_max;
    }
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const float3* ref_view_dirs, const PatchMatchParams params)
{
    for (int i = 1; i < params.num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, ref_view_dirs, plane_hypothesis, params);
    }
}

__device__ float3 Get3DPointonWorld_cu(const float3 view_dir, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    float scale = depth / view_dir.z;
    pointX.x = view_dir.x * scale;
    pointX.y = view_dir.y * scale;
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

__device__ void ProjectonCameraGeneral_cu(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    // global --> local
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    // local 3d point(view_dir) --> image point
    point= ViewDirToImagePoint(camera, tmp);
    depth = tmp.z;
}

__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p, const float3 ref_view_dir, const float3* src_view_dirs)
{
    const float max_cost = 5.0f;

    float depth = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypothesis);
    float3 forward_point = Get3DPointonWorld_cu(ref_view_dir, depth, ref_camera);

    float2 src_pt;
    float src_d;
    ProjectonCameraGeneral_cu(forward_point, src_camera, src_pt, src_d);

    if(src_pt.x < 0 || src_pt.x > src_camera.width - 1 || src_pt.y < 0 || src_pt.y > src_camera.height - 1) {
        return max_cost;
    }

    const float src_depth = tex2D<float>(depth_image,  (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

    if (src_depth == 0.0f) {
        return max_cost;
    }


    float3 src_view_dir = src_view_dirs[int(src_pt.y * src_camera.width + src_pt.x)];
    float3 src_3D_pt = Get3DPointonWorld_cu(src_view_dir, src_depth, src_camera);

    float2 backward_point;
    float ref_d;
    ProjectonCameraGeneral_cu(src_3D_pt, ref_camera, backward_point, ref_d);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks,  const float3* ref_view_dirs, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    curand_init(0x1234567, p.y, p.x, &rand_states[center]);

    float3 view_dir = ref_view_dirs[center];

    if (params.geom_consistency) {
        float4 plane_hypothesis = plane_hypotheses[center];
        float depth = plane_hypothesis.w;
        plane_hypothesis.w = GetDistance2Origin(view_dir, depth, plane_hypothesis);
        plane_hypotheses[center] = plane_hypothesis;
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, ref_view_dirs, plane_hypotheses[center], &selected_views[center], params);
    }
    else if (params.planar_prior) {
        if (plane_masks[center] > 0 && costs[center] >= 0.1f) {
            float perturbation = 0.02f;

            float4 plane_hypothesis = prior_planes[center];
            float depth_perturbed = plane_hypothesis.w;
            const float depth_min_perturbed = (1 - 3 * perturbation) * depth_perturbed;
            const float depth_max_perturbed = (1 + 3 * perturbation) * depth_perturbed;
            depth_perturbed = curand_uniform(&rand_states[center]) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
            float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(view_dir, plane_hypothesis, &rand_states[center], 3 * perturbation * M_PI);
            plane_hypothesis_perturbed.w = depth_perturbed;
            plane_hypotheses[center] = plane_hypothesis_perturbed;
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, ref_view_dirs, plane_hypotheses[center], &selected_views[center], params);
        }
        else {
            float4 plane_hypothesis = plane_hypotheses[center];
            float depth = plane_hypothesis.w;
            plane_hypothesis.w = GetDistance2Origin(view_dir, depth, plane_hypothesis);
            plane_hypotheses[center] = plane_hypothesis;
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, ref_view_dirs, plane_hypotheses[center], &selected_views[center], params);
        }
    }
    else {
        plane_hypotheses[center] = GenerateRandomPlaneHypothesis(view_dir, &rand_states[center], params.depth_min, params.depth_max);
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, ref_view_dirs, plane_hypotheses[center], &selected_views[center], params);
    }
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *plane_hypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, float4 *prior_planes, unsigned int *plane_masks, float *restricted_cost, const int2 p, float3* view_dirs_vec[], const PatchMatchParams params)
{
    float perturbation = 0.02f;
    const int center = p.y * cameras[0].width + p.x;

    float3 ref_view_dir = view_dirs_vec[0][center];

    float gamma = 0.5f;
    float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
    float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
    float angle_sigma = M_PI * (5.0f / 180.0f);
    float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
    float beta = 0.18f;
    float depth_prior = 0.0f;

    float depth_rand;
    float4 plane_hypothesis_rand;
    if (params.planar_prior && plane_masks[center] > 0) {
        depth_prior = ComputeDepthfromPlaneHypothesis(ref_view_dir, prior_planes[center]);
        depth_rand = curand_uniform(rand_state) * 6 * depth_sigma + (depth_prior - 3 * depth_sigma);
        plane_hypothesis_rand = GeneratePerturbedNormal(ref_view_dir, prior_planes[center], rand_state, angle_sigma);
    }
    else {
        depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
        plane_hypothesis_rand = GenerateRandomNormal(ref_view_dir, rand_state);
    }
    float depth_perturbed = *depth;
    const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
    const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
    do {
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
    } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);
    float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(ref_view_dir, *plane_hypothesis, rand_state, perturbation * M_PI); // GeneratePertubedPlaneHypothesis(cameras[0], p, rand_state, perturbation, *plane_hypothesis, *depth, params.depth_min, params.depth_max);

    const int num_planes = 5;
    float depths[num_planes] = {depth_rand, *depth, depth_rand, *depth, depth_perturbed};
    float4 normals[num_planes] = {*plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis};

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[32] = {2.0f};
        float4 temp_plane_hypothesis = normals[i];
        temp_plane_hypothesis.w = GetDistance2Origin(ref_view_dir, depths[i], temp_plane_hypothesis); // dists[i];
        ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, view_dirs_vec[0], params);

        float temp_cost = 0.0f;
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    temp_cost += view_weights[j] * (cost_vector[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p, ref_view_dir, view_dirs_vec[j + 1]));
                }
                else {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        temp_cost /= weight_norm;

        float depth_before = ComputeDepthfromPlaneHypothesis(ref_view_dir, temp_plane_hypothesis);
        if (params.planar_prior && plane_masks[center] > 0) {
            float depth_diff = depths[i] - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
            float restricted_temp_cost = exp(-temp_cost * temp_cost / beta) * prior;
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_temp_cost > *restricted_cost) {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
                *restricted_cost = restricted_temp_cost;
            }
        }
        else {
            if (depth_before >= params.depth_min && depth_before <= params.depth_max && temp_cost < *cost) {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
            }
        }
    }
}

__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, float3** view_dirs_vec, const int2 p, const PatchMatchParams params, const int iter)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    float3 ref_view_dir = view_dirs_vec[0][center];
    int left_near = center - 1;
    int left_far = center - 3;
    int right_near = center + 1;
    int right_far = center + 3;
    int up_near = center - width;
    int up_far = center - 3 * width;
    int down_near = center + width;
    int down_far = center + 3 * width;

    // Adaptive Checkerboard Sampling
    float cost_array[8][32] = {{2.0f}};
    // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
    bool flag[8] = {false};
    int num_valid_pixels = 0;

    float costMin;
    int costMinPoint;

    // up_far
    if (p.y > 2) {
        flag[1] = true;
        num_valid_pixels++;
        costMin = costs[up_far];
        costMinPoint = up_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int pointTemp = up_far - 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_far], cost_array[1], view_dirs_vec[0], params);
    }

    // dwon_far
    if (p.y < height - 3) {
        flag[3] = true;
        num_valid_pixels++;
        costMin = costs[down_far];
        costMinPoint = down_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int pointTemp = down_far + 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_far], cost_array[3], view_dirs_vec[0], params);
    }

    // left_far
    if (p.x > 2) {
        flag[5] = true;
        num_valid_pixels++;
        costMin = costs[left_far];
        costMinPoint = left_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int pointTemp = left_far - 2 * i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_far], cost_array[5], view_dirs_vec[0], params);
    }

    // right_far
    if (p.x < width - 3) {
        flag[7] = true;
        num_valid_pixels++;
        costMin = costs[right_far];
        costMinPoint = right_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int pointTemp = right_far + 2 * i;
                if (costMin < costs[pointTemp]) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_far], cost_array[7], view_dirs_vec[0], params);
    }

    // up_near
    if (p.y > 0) {
        flag[0] = true;
        num_valid_pixels++;
        costMin = costs[up_near];
        costMinPoint = up_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int pointTemp = up_near - (1 + i) * width - i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y > 1 + i && p.x < width - 1 - i) {
                int pointTemp = up_near - (1 + i) * width + i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near], cost_array[0], view_dirs_vec[0], params);
    }

    // down_near
    if (p.y < height - 1) {
        flag[2] = true;
        num_valid_pixels++;
        costMin = costs[down_near];
        costMinPoint = down_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y < height - 2 - i && p.x > i) {
                int pointTemp = down_near + (1 + i) * width - i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y < height - 2 - i && p.x < width - 1 - i) {
                int pointTemp = down_near + (1 + i) * width + i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near], cost_array[2], view_dirs_vec[0], params);
    }

    // left_near
    if (p.x > 0) {
        flag[4] = true;
        num_valid_pixels++;
        costMin = costs[left_near];
        costMinPoint = left_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int pointTemp = left_near - (1 + i) - i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x > 1 + i && p.y < height - 1 - i) {
                int pointTemp = left_near - (1 + i) + i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near], cost_array[4], view_dirs_vec[0], params);
    }

    // right_near
    if (p.x < width - 1) {
        flag[6] = true;
        num_valid_pixels++;
        costMin = costs[right_near];
        costMinPoint = right_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x < width - 2 - i && p.y > i) {
                int pointTemp = right_near + (1 + i) - i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x < width - 2 - i && p.y < height - 1- i) {
                int pointTemp = right_near + (1 + i) + i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near], cost_array[6], view_dirs_vec[0], params);
    }
    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Multi-hypothesis Joint View Selection
    float view_weights[32] = {0.0f};
    float view_selection_priors[32] = {0.0f};
    int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
    for (int i = 0; i < 4; ++i) {
        if (flag[2 * i]) {
            for (int j = 0; j < params.num_images - 1; ++j) {
                if (isSet(selected_views[neighbor_positions[i]], j) == 1) {
                    view_selection_priors[j] += 0.9f;
                } else {
                    view_selection_priors[j] += 0.1f;
                }
            }
        }
    }

    float sampling_probs[32] = {0.0f};
    for (int i = 0; i < params.num_images - 1; ++i) {
        float temp_w = 0.0f;
        for (int j = 0; j < 8; ++j) {
            if (flag[j]) {
                temp_w += exp(cost_array[j][i] * cost_array[j][i] / -0.18f);
            }
        }
        sampling_probs[i] = temp_w / num_valid_pixels * view_selection_priors[i];
    }

    TransformPDFToCDF(sampling_probs, params.num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

        for (int image_id = 0; image_id < params.num_images - 1; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    int num_selected_view = 0;
    float weight_norm = 0;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(temp_selected_views, i);
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    float final_costs[8] = {0.0f};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    if (flag[i]) {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * ComputeGeomConsistencyCost(depths[j+1], cameras[0], cameras[j+1], plane_hypotheses[positions[i]], p, ref_view_dir, view_dirs_vec[j+1]));
                    }
                    else {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 5.0f);
                    }
                }
                else {
                    final_costs[i] += view_weights[j] * cost_array[i][j];
                }
            }
        }
        final_costs[i] /= weight_norm;
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    float cost_vector_now[32] = {2.0f};
    ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, view_dirs_vec[0], params);
    float cost_now = 0.0f;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (params.geom_consistency) {
            cost_now += view_weights[i] * (cost_vector_now[i] + 0.1f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p, ref_view_dir, view_dirs_vec[i+1]));
        }
        else {
            cost_now += view_weights[i] * cost_vector_now[i];
        }
    }
    cost_now /= weight_norm;
    costs[center] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[center]);

    float restricted_cost = 0.0f;
    if (params.planar_prior) {
        float restricted_final_costs[8] = {0.0f};
        float gamma = 0.5f;
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
        float angle_sigma = M_PI * (5.0f / 180.0f);
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
        float depth_prior = ComputeDepthfromPlaneHypothesis(ref_view_dir, prior_planes[center]);
        float beta = 0.18f;

        if (plane_masks[center] > 0) {
            for (int i = 0; i < 8; i++) {
                if (flag[i]) {
                    float depth_now = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[positions[i]]);
                    float depth_diff = depth_now - depth_prior;
                    float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
                    float angle_diff = acos(angle_cos);
                    float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                    restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                }
            }
            const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

            float restricted_cost_now = 0.0f;
            float depth_now = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[center]);
            float depth_diff = depth_now - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
            restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

            if (flag[max_cost_idx]) {
                float depth_before = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[positions[max_cost_idx]]);

                if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_final_costs[max_cost_idx] > restricted_cost_now) {
                    depth_now   = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]];
                    costs[center] = final_costs[max_cost_idx];
                    restricted_cost = restricted_final_costs[max_cost_idx];
                    selected_views[center] = temp_selected_views;
                }
            }
        }
        else if (flag[min_cost_idx]) {
            float depth_before = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[positions[min_cost_idx]]);

            if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
                depth_now = depth_before;
                plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]];
                costs[center] = final_costs[min_cost_idx];
            }
        }
    }

    if (!params.planar_prior && flag[min_cost_idx]) {
       float depth_before = ComputeDepthfromPlaneHypothesis(ref_view_dir, plane_hypotheses[positions[min_cost_idx]]);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
            plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]];
            costs[center] = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses[center], &depth_now, &costs[center], &rand_states[center], view_weights, weight_norm, prior_planes, plane_masks, &restricted_cost, p, view_dirs_vec, params);
}

__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, float3** view_dirs_vec, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, rand_states, selected_views, prior_planes, plane_masks, view_dirs_vec, p, params, iter);
}

__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, float3** view_dirs_vec, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, rand_states, selected_views, prior_planes, plane_masks, view_dirs_vec, p, params, iter);
}

__global__ void GetDepthandNormal(const int width, const int height, const float3* view_dirs, float4 *plane_hypotheses)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    float3 view_dir = view_dirs[center];
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(view_dir, plane_hypotheses[center]);
}

__device__ void CheckerboardFilter(const int width, const int height, float4 *plane_hypotheses, float *costs, const int2 p)
{
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int left = center - 1;
    const int leftleft = center - 3;

    // Up
    const int up = center - width;
    const int upup = center - 3 * width;

    // Down
    const int down = center + width;
    const int downdown = center + 3 * width;

    // Right
    const int right = center + 1;
    const int rightright = center + 3;

    if (costs[center] < 0.001f) {
        return;
    }

    if (p.y>0) {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y>2) {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y>4) {
        filter[index++] = plane_hypotheses[upup-width*2].w;
    }
    if (p.y<height-1) {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y<height-3) {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y<height-5) {
        filter[index++] = plane_hypotheses[downdown+width*2].w;
    }
    if (p.x>0) {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x>2) {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x>4) {
        filter[index++] = plane_hypotheses[leftleft-2].w;
    }
    if (p.x<width-1) {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x<width-3) {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x<width-5) {
        filter[index++] = plane_hypotheses[rightright+2].w;
    }
    if (p.y>0 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[up+2].w;
    }
    if (p.y< height-1 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[down+2].w;
    }
    if (p.y>0 &&
        p.x>1)
    {
        filter[index++] = plane_hypotheses[up-2].w;
    }
    if (p.y<height-1 &&
        p.x>1) {
        filter[index++] = plane_hypotheses[down-2].w;
    }
    if (p.x>0 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[left  - width*2].w;
    }
    if (p.x<width-1 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[right - width*2].w;
    }
    if (p.x>0 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[left  + width*2].w;
    }
    if (p.x<width-1 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[right + width*2].w;
    }

    sort_small(filter,index);
    int median_index = index / 2;
    if (index % 2 == 0) {
        plane_hypotheses[center].w = (filter[median_index-1] + filter[median_index]) / 2;
    } else {
        plane_hypotheses[center].w = filter[median_index];
    }
}

__global__ void BlackPixelFilter(const int width, const int height, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter(width, height, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const int width, const int height, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardFilter(width, height, plane_hypotheses, costs, p);
}

void ACMP::RunPatchMatch()
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;
    // std::cout << width << " " << height << std::endl;

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard;
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard;
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = params.max_iterations;

    RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, view_dirs_vec_cuda[0], params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int i = 0; i < max_iterations; ++i) {
        BlackPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, view_dirs_vec_cuda_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        RedPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, view_dirs_vec_cuda_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("iteration: %d\n", i);
    }

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(width, height, view_dirs_vec_cuda[0], plane_hypotheses_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BlackPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(width, height, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    RedPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(width, height, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
