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
#include "global_planning/rrt.h"
#include "cost_map/gaussian_conv.h"
#include "cost_map/nearest.h"

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        Dimensions dimensions = Dimensions{.3, .3, .3};
        Constraints positive_constraints = Constraints{
            {-1000, 1000},  // x
            {-1000, 1000},  // y
            {-.34, .34},    // tau
            {0, .5},        // vel
            {0, .25},       // accel
            {-.34, .34}     // dtau
        };

        Constraints full_constraints = Constraints{
            {-1000, 1000},  // x
            {-1000, 1000},  // y
            {-.34, .34},    // tau
            {-.5, 1.0},     // vel
            {-.75, .75},    // accel
            {-.34, .34}     // dtau
        };

        // planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints,
        //     std::make_shared<cost_map::NearestGenerator>(5, 1.05));

        planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints,
            std::make_shared<cost_map::NearestGenerator>(3, .5));

        global_planner = std::make_shared<global_planner::RRT>(dimensions, full_constraints);


        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("trajectory", 1);

        nav_path_pub = this->create_publisher<nav_msgs::msg::Path>("nav_path", 1);

        target_rviz_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("goal_pose", 1,
            std::bind(&PlannerNode::rviz_target_callback, this, std::placeholders::_1));

        estimated_position_in_map_frame = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "estimated_position_in_map_frame", 1);
    }

private:
    Grid grid = Grid();
    State start = State();
    State prev_start = State();
    bool second_iteration_passed = false;
    State target = State();
    float prev_path_cost = 100000000;

    bool map_initialized = false;
    bool odom_initialized = false;
    bool target_initialized = false;

    cev_msgs::msg::Trajectory current_path;

    std::shared_ptr<local_planner::MPC> planner;
    std::shared_ptr<global_planner::RRT> global_planner;

    //// ROS

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Init frames odom map baselink so that we can localize base_link inside map given the odom to
    // map transform

    // Listener to the map
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;

    // Listener to odom
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;

    // Listener to the target
    rclcpp::Subscription<cev_msgs::msg::Waypoint>::SharedPtr target_sub;

    // Publisher for the planned path
    rclcpp::Publisher<cev_msgs::msg::Trajectory>::SharedPtr path_pub;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr nav_path_pub;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr estimated_position_in_map_frame;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Transform into map frame
        geometry_msgs::msg::TransformStamped transform;

        geometry_msgs::msg::PoseStamped base_link_pose;
        base_link_pose.header = msg->header;
        base_link_pose.pose = msg->pose.pose;

        geometry_msgs::msg::PoseStamped map_pose;

        try {
            transform = tf_buffer_.lookupTransform("map", "odom", tf2::TimePointZero);
            tf2::doTransform(base_link_pose, map_pose, transform);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform odom to map: %s", ex.what());
            return;
        }

        float qw = map_pose.pose.orientation.w;
        float qx = map_pose.pose.orientation.x;
        float qy = map_pose.pose.orientation.y;
        float qz = map_pose.pose.orientation.z;

        float yaw = restrict_angle(atan2(2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)));

        start = State{
            map_pose.pose.position.x, map_pose.pose.position.y, yaw, msg->twist.twist.linear.x, 0};
        odom_initialized = true;

        geometry_msgs::msg::PoseStamped point;
        point.header.stamp = msg->header.stamp;
        point.header.frame_id = "map";
        point.pose.position.x = start.pose.x;
        point.pose.position.y = start.pose.y;
        point.pose.position.z = 0;
        point.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
            start.pose.theta));

        estimated_position_in_map_frame->publish(point);

        float dist = start.pose.distance_to(prev_start.pose);

        if (map_initialized && odom_initialized && target_initialized
            && (!second_iteration_passed
                || (dist > .1))) {              // Ensure that enough dist has changed before replan
            Trajectory path = planner->plan_path(grid, start, target, Trajectory());

            if (path.cost >= prev_path_cost) {  // Worse path
                return;
            }

            prev_path_cost = path.cost;
            second_iteration_passed = true;
            prev_start = start;

            current_path.header.stamp = msg->header.stamp;
            current_path.header.frame_id = "map";
            current_path.waypoints.clear();
            for (State waypoint: path.waypoints) {
                cev_msgs::msg::Waypoint msg;
                msg.x = waypoint.pose.x;
                msg.y = waypoint.pose.y;

                if (std::abs(waypoint.vel) < .3
                    && std::abs(waypoint.vel) > 0.0) {  // Car cannot move that slow lol
                    waypoint.vel = (waypoint.vel >= 0) ? .3 : -.3;
                }

                msg.v = waypoint.vel;
                msg.theta = waypoint.pose.theta;
                msg.tau = waypoint.tau;
                current_path.waypoints.push_back(msg);
                // std::cout << "Waypoint: " << msg.theta << ", " << msg.tau << std::endl;
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
                if (msg->data[j * msg->info.width + i] < 0) {
                    grid.data(i, j) = -1.0;
                } else if (msg->data[j * msg->info.width + i] < 50) {
                    grid.data(i, j) = 0.0;
                } else {
                    // grid.data(i, j) = std::min(msg->data[j * msg->info.width + i] / 100.0, 1.0);
                    grid.data(i, j) = 1.0;
                }
            }
        }

        map_initialized = true;
    }

    void target_callback(const cev_msgs::msg::Waypoint msg) {
        target = State{msg.x, msg.y, 0, msg.v, 0};
        prev_path_cost = 100000000;
        second_iteration_passed = false;
        target_initialized = true;
    }

    void rviz_target_callback(const geometry_msgs::msg::PoseStamped msg) {
        target = State{msg.pose.position.x, msg.pose.position.y, 0, 0, 0};
        prev_path_cost = 100000000;
        second_iteration_passed = false;
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