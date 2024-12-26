#include <unordered_map>
#include <list>

#include <opencv2/core.hpp>

#include "./colmap_interface.h"

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


struct DataCache {
    DataCache(Model model,
              std::string depth_suffix,
              size_t max_bytes = 32e9 /* memory 32 Gb */
              )
        :MAX_BYTES(max_bytes), total_bytes_(0), model_(model), suffix_(depth_suffix){}

    int GetData(const int image_idx,
            Camera& camera,
            cv::Mat_<cv::Vec3b>& image,
            cv::Mat_<float>& depth,
            cv::Mat_<cv::Vec3f>& normal,
            bool deep_copy = false,
            bool change_order = true
            ) {

        if(change_order) {
            // 把最近访问的这个挪到最前面
            auto iter = id_position_in_order_list_.find(image_idx);
            if(iter != id_position_in_order_list_.end()) {
                // 不在最前面才需要调换位置，在最前面就不用调换位置了
                if(visit_order_.front() != image_idx) {
                    // 删掉老的
                    visit_order_.erase(iter->second);
                    // 插入最新的
                    visit_order_.push_front(image_idx);
                }
            } else {
                visit_order_.push_front(image_idx);
            }
            // 最近访问的这个元素一定会出现在order list的最前面
            id_position_in_order_list_[image_idx] = visit_order_.begin();
        }

        if(depths_map_.count(image_idx)) {
            camera = cameras_map_.at(image_idx);
            if(deep_copy) {
                image = images_map_.at(image_idx).clone();
                depth = depths_map_.at(image_idx).clone();
                normal = normals_map_.at(image_idx).clone();
            } else {
                image = images_map_.at(image_idx);
                depth = depths_map_.at(image_idx);
                normal = normals_map_.at(image_idx);
            }
            return 0;
        }

        camera = model_.image_idx_to_camera.at(image_idx);  // original camera, needs to be scaled

        std::string image_name = model_.image_idx_to_image_name.at(image_idx);
        std::cout << "Reading image " << image_name << "..." << std::endl;
        std::string image_path = model_.root_folder + "/" + model_.image_folder + "/" + image_name;
        cv::Mat_<cv::Vec3b> image_origin = cv::imread (image_path, cv::IMREAD_COLOR);

        std::string depth_path = model_.root_folder + "/" + model_.depth_folder + "/" + image_name + suffix_;
        std::string normal_path = model_.root_folder + "/" + model_.normal_folder + "/" + image_name + suffix_;
        ReadMap(depth_path, depth);
        ReadMap(normal_path, normal);
        RescaleImageAndCamera(image_origin, image, depth, camera);
        cameras_map_[image_idx] = camera;
        images_map_[image_idx] = image;
        depths_map_[image_idx] = depth;
        normals_map_[image_idx] = normal;

        total_bytes_ += sizeof(Camera);
        total_bytes_ += image.rows * image.cols * sizeof(cv::Vec3b);
        total_bytes_ += depth.rows * depth.cols * sizeof(float);
        total_bytes_ += normal.rows * normal.cols * sizeof(cv::Vec3f);

        if(total_bytes_ > MAX_BYTES) {
            // remove least usaged data
            int key_to_remove = visit_order_.back();
            if(key_to_remove == image_idx) {
                std::cerr << "Error! id=" << image_idx << std::endl;
            }

            cameras_map_.erase(key_to_remove);
            images_map_.erase(key_to_remove);
            depths_map_.erase(key_to_remove);
            normals_map_.erase(key_to_remove);
            std::cout << "erase id " << key_to_remove << std::endl;

            total_bytes_ -= sizeof(Camera);
            total_bytes_ -= image.rows * image.cols * sizeof(cv::Vec3b);
            total_bytes_ -= depth.rows * depth.cols * sizeof(float);
            total_bytes_ -= normal.rows * normal.cols * sizeof(cv::Vec3f);

            visit_order_.pop_back();
            id_position_in_order_list_.erase(key_to_remove);
        }

        std::cout << "total bytes: " << total_bytes_ << std::endl;

        return 0;
    }

    std::unordered_map<int, Camera> cameras_map_;
    std::unordered_map<int, cv::Mat> images_map_;
    std::unordered_map<int, cv::Mat_<float>> depths_map_;
    std::unordered_map<int, cv::Mat_<cv::Vec3f>> normals_map_;

    const size_t MAX_BYTES;
    size_t total_bytes_;

    std::list<int> visit_order_;
    std::unordered_map<int, std::list<int>::iterator> id_position_in_order_list_;

    const Model model_;
    const std::string suffix_;
};
