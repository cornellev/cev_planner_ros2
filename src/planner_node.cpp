#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <eigen3/Eigen/Dense>
#include <cev_msgs/msg/trajectory.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"
#include <iostream>

#include "local_planning/mpc.h"
#include "cost_map/gaussian_conv.h"

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        Dimensions dimensions = Dimensions{1.0, 1.0, 1.0};
        Constraints constraints = Constraints{
            {-1000, 1000},          // x
            {-1000, 1000},          // y
            {-M_PI / 4, M_PI / 4},  // tau
            {0, 10},                // vel
            {-1, 1},                // accel
            {-M_PI / 4, M_PI / 4}   // dtau
        };

        planner = std::make_shared<local_planner::MPC>(dimensions, constraints,
            std::make_shared<cost_map::GaussianConvolution>(15, 10.0));

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("path", 1);

        nav_path_pub = this->create_publisher<nav_msgs::msg::Path>("nav_path", 1);
    }

private:
    Grid grid = Grid();
    State start = State();
    State target = State{5, 5, 0, 0, 0};

    bool map_initialized = false;
    bool odom_initialized = false;
    bool target_initialized = true;

    cev_msgs::msg::Trajectory current_path;

    std::shared_ptr<local_planner::MPC> planner;

    //// ROS

    // TF
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Init frames odom map baselink so that we can localize base_link inside map given the odom to
    // map transform
    std::string odom_frame = "odom";
    std::string map_frame = "map";
    std::string base_link_frame = "base_link";

    // Listener to the map
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;

    // Listener to odom
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;

    // Listener to the target
    rclcpp::Subscription<cev_msgs::msg::Waypoint>::SharedPtr target_sub;

    // Publisher for the planned path
    rclcpp::Publisher<cev_msgs::msg::Trajectory>::SharedPtr path_pub;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr nav_path_pub;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        float qw = msg->pose.pose.orientation.w;
        float qx = msg->pose.pose.orientation.x;
        float qy = msg->pose.pose.orientation.y;
        float qz = msg->pose.pose.orientation.z;

        float yaw = restrict_angle(atan2(2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)));

        start = State{msg->pose.pose.position.x, msg->pose.pose.position.y, yaw,
            msg->twist.twist.linear.x, 0.0};
        odom_initialized = true;

        // Transform into map frame
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(map_frame, odom_frame, tf2::TimePointZero);
        } catch (tf2::TransformException& ex) {
            return;
        }

        tf2::Transform tf_transform;
        tf2::fromMsg(transform.transform, tf_transform);

        start.pose.x += transform.transform.translation.x;
        start.pose.y += transform.transform.translation.y;
        start.pose.theta = restrict_angle(start.pose.theta
                                          + tf2::getYaw(transform.transform.rotation));

        // std::cout << start.pose.x << " , " << start.pose.y << " , " << start.pose.theta
        //           << std::endl;

        if (map_initialized && odom_initialized && target_initialized) {
            Trajectory path = planner->plan_path(grid, start, target, Trajectory());

            current_path.header.stamp = msg->header.stamp;
            current_path.header.frame_id = "map";
            current_path.waypoints.clear();
            for (State waypoint: path.waypoints) {
                cev_msgs::msg::Waypoint msg;
                msg.x = waypoint.pose.x;
                msg.y = waypoint.pose.y;
                msg.v = waypoint.vel;
                current_path.waypoints.push_back(msg);
            }

            path_pub->publish(current_path);

            nav_msgs::msg::Path nav_path;
            nav_path.header.stamp = msg->header.stamp;
            nav_path.header.frame_id = "map";
            nav_path.poses.clear();

            for (State waypoint: path.waypoints) {
                geometry_msgs::msg::PoseStamped pose;
                pose.pose.position.x = waypoint.pose.x;
                pose.pose.position.y = waypoint.pose.y;
                pose.pose.position.z = 0;
                pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
                    waypoint.pose.theta));
                nav_path.poses.push_back(pose);
            }

            nav_path_pub->publish(nav_path);
        }
    }

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        grid = Grid();
        grid.origin = Pose{msg->info.origin.position.x, msg->info.origin.position.y, 0};
        grid.resolution = msg->info.resolution;

        grid.data = Eigen::MatrixXf(msg->info.width, msg->info.height);

        for (int i = 0; i < msg->info.width; i++) {
            for (int j = 0; j < msg->info.height; j++) {
                // Divide by 100 to get probability of occupancy in the range [0, 1]
                if (msg->data[j * msg->info.width + i] < 50) {
                    grid.data(i, j) = 0.0;
                } else {
                    grid.data(i, j) = std::min(msg->data[j * msg->info.width + i] / 100.0, 1.0);
                }
            }
        }

        // int i_increment = 1;
        // int j_increment = 1;
        // int i = 0;
        // int j = 0;

        // int width = 100;
        // int length = 50;

        // grid.data = Eigen::MatrixXf(width, length);

        // while (i < width) {
        //     j = 0;
        //     j_increment = 1;
        //     while (j < length) {
        //         grid.data(i, j) = 1.0;
        //         j += j_increment;
        //         // j_increment += 1;
        //     }
        //     i += i_increment;
        //     i_increment += 1;
        // }

        map_initialized = true;
    }

    void target_callback(const cev_msgs::msg::Waypoint msg) {
        target = State{msg.x, msg.y, 0, msg.v, 0};
        target_initialized = true;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}