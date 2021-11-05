#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>
#include <iostream>
#include "../include/gpd/grasp_detector.h"
#include "../include/gpd/util/cloud.h"
#include "../include/gpd/candidate/hand.h"

// 3 for position, 9 for orientation, 1 for width, 1 for score
#define GRASP_DESC_LEN 14

pybind11::array_t<double> detectGrasps(
    gpd::GraspDetector &detector,
    pybind11::array_t<double, pybind11::array::c_style> pcd)
{
    // TODO: Make these conditions raise actual exceptions
    if (pcd.ndim() != 2) {
        throw std::invalid_argument("PCD must have two dimensions");
    }
    ssize_t len = pcd.shape()[0];
    if (len == 0) {
        throw std::invalid_argument("PCD must more than 0 points");
    }
    if (pcd.shape()[1] < 3) {
        throw std::invalid_argument("Each point must have at least 3 coordinates");
    }
    pcl::PointCloud<pcl::PointXYZRGBA> e_cloud;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr = e_cloud.makeShared();
    for(ssize_t i = 0; i < len; i++){
        pcl::PointXYZRGBA pt(pcd.at(i, 0), pcd.at(i, 1), pcd.at(i, 2));
        cloud_ptr->push_back(pt);
    }
    gpd::util::Cloud gpd_cloud(cloud_ptr);
    detector.preprocessPointCloud(gpd_cloud);
    std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps = detector.detectGrasps(gpd_cloud);
    std::vector<std::array<double, GRASP_DESC_LEN>> ret;
    for(size_t i = 0; i < grasps.size(); i++){
        const Eigen::Vector3d pos = grasps[i]->getPosition();
        const Eigen::Matrix3d rot = grasps[i]->getOrientation();
        const double w = grasps[i]->getGraspWidth();
        const double score = grasps[i]->getScore();
        std::array<double, GRASP_DESC_LEN> gd = {
            pos[0], pos[1], pos[2],
            rot(0,0), rot(0,1), rot(0,2),
            rot(1,0), rot(1,1), rot(1,2),
            rot(2,0), rot(2,1), rot(2,2),
            w, score
        };
        ret.push_back(gd);
    }
    return pybind11::cast(ret);
}

PYBIND11_MODULE(gpd_python, m) {
    pybind11::class_<gpd::GraspDetector>(m, "GraspDetector")
        .def(pybind11::init<const std::string &>());
    m.def("detectGrasps", &detectGrasps);
}
