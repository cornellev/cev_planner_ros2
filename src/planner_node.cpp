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
#include "cost_map/nothing.h"

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
            {-.5, .75},     // vel
            {-.3, .4},      // accel
            {-.34, .34}     // dtau
        };

        local_planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints,
            std::make_shared<cost_map::Nothing>(2, .5));

        global_planner = std::make_shared<global_planner::RRT>(dimensions, full_constraints);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("trajectory", 1);

        // Global plan every 2 seconds
        global_plan_timer = this->create_wall_timer(std::chrono::milliseconds(2000),
            std::bind(&PlannerNode::global_plan_callback, this));

        // RVIZ Debug
        global_path_pub = this->create_publisher<nav_msgs::msg::Path>("global_path", 1);
        local_path_pub = this->create_publisher<nav_msgs::msg::Path>("local_path", 1);

        target_rviz_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("goal_pose", 1,
            std::bind(&PlannerNode::rviz_target_callback, this, std::placeholders::_1));
    }

private:
    Grid grid = Grid();
    State start = State();
    State prev_start = State();
    State target = State();

    bool map_initialized = false;
    bool odom_initialized = false;
    bool target_initialized = false;

    // Global Planner
    std::shared_ptr<global_planner::RRT> global_planner;
    Trajectory global_path;
    double global_path_cost = 100000000;
    bool global_path_initialized = false;

    // Local Planner
    bool second_iteration_passed = false;
    float prev_path_cost = 100000000;
    Trajectory current_local_plan;
    int current_waypoint_in_global = 0;

    std::shared_ptr<local_planner::MPC> local_planner;

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

    // Wall timer for global plan
    rclcpp::TimerBase::SharedPtr global_plan_timer;

    // RVIZ
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub;

    // -------------------------------

    void global_plan_callback() {
        if (odom_initialized && map_initialized && target_initialized && !global_path_initialized) {
            std::optional<Trajectory> optional = global_planner->plan_path(grid, start, target);
            if (!optional.has_value()) {
                std::cout << "Global Path Failed." << std::endl;
                global_path_initialized = false;
            } else {
                std::cout << "Global Path Planned." << std::endl;
                // Fill in the global path with all nodes except the first and last from the global
                // path
                // global_path.waypoints = std::vector<State>(optional.value().waypoints.begin() +
                // 1,
                //     optional.value().waypoints.end() - 1);
                global_path.waypoints = optional.value().waypoints;

                // TODO: Cost shenanigans, waypoint pass checking

                // Publish global path
                nav_msgs::msg::Path global_nav_path;
                global_nav_path.header.stamp = this->now();
                global_nav_path.header.frame_id = "map";
                global_nav_path.poses.clear();

                for (State waypoint: global_path.waypoints) {
                    geometry_msgs::msg::PoseStamped pose;
                    pose.pose.position.x = waypoint.pose.x;
                    pose.pose.position.y = waypoint.pose.y;
                    pose.pose.position.z = 0;
                    pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
                        waypoint.pose.theta));
                    global_nav_path.poses.push_back(pose);
                }

                global_path_pub->publish(global_nav_path);

                // cev_msgs::msg::Trajectory current_plan;

                // current_plan.header.stamp = this->now();
                // current_plan.header.frame_id = "map";
                // current_plan.waypoints.clear();
                // current_plan.timestep = global_path.timestep;

                // // std::cout << "What 3" << std::endl;

                // for (State waypoint: global_path.waypoints) {
                //     cev_msgs::msg::Waypoint msg;
                //     msg.x = waypoint.pose.x;
                //     msg.y = waypoint.pose.y;

                //     // if (std::abs(waypoint.vel) < .3
                //     //     && std::abs(waypoint.vel) > 0.0) {  // Car cannot move that slow lol
                //     //     waypoint.vel = (waypoint.vel >= 0) ? .3 : -.3;
                //     // }

                //     msg.v = waypoint.vel;
                //     msg.theta = waypoint.pose.theta;
                //     msg.tau = waypoint.tau;
                //     current_plan.waypoints.push_back(msg);
                // }

                // std::cout << "Publishing path" << std::endl;

                // path_pub->publish(current_plan);

                global_path_initialized = true;
            }
        }
    }

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
            RCLCPP_DEBUG(this->get_logger(), "Could not transform odom to map: %s", ex.what());
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

        float dist = start.pose.distance_to(prev_start.pose);

        if (map_initialized && odom_initialized && target_initialized && global_path_initialized) {
            // && (!second_iteration_passed
            //     || (dist > .025))) {  // Ensure that enough dist has changed before
            // replan
            // Keep only waypoints not including start or target from the global path

            float dist =
                start.pose.distance_to(global_path.waypoints[current_waypoint_in_global].pose);

            float dist_to_dest = start.pose.distance_to(target.pose);

            if (dist_to_dest < .4) {
                // Write a trajectory with 0 velocity
                cev_msgs::msg::Trajectory current_plan;

                current_plan.header.stamp = msg->header.stamp;
                current_plan.header.frame_id = "map";
                current_plan.waypoints.clear();

                cev_msgs::msg::Waypoint msg;
                msg.x = target.pose.x;
                msg.y = target.pose.y;
                msg.v = 0;
                msg.theta = target.pose.theta;
                msg.tau = target.tau;
                current_plan.waypoints.push_back(msg);

                path_pub->publish(current_plan);

                return;
            }

            while (dist < 1.0 && current_waypoint_in_global < global_path.waypoints.size()) {
                // std::cout << "Skipping" << std::endl;
                current_waypoint_in_global += 1;
                dist =
                    start.pose.distance_to(global_path.waypoints[current_waypoint_in_global].pose);
            }

            // std::cout << "Current: " << current_waypoint_in_global << std::endl;
            // std::cout << "Dist to waypoint: " << dist << std::endl;

            Trajectory waypoints;

            if (current_waypoint_in_global < global_path.waypoints.size() - 1) {
                waypoints.waypoints.push_back(global_path.waypoints[current_waypoint_in_global]);

                if ((current_waypoint_in_global + 1) < global_path.waypoints.size()) {
                    waypoints.waypoints.push_back(
                        global_path.waypoints[current_waypoint_in_global + 1]);
                } else {
                    waypoints.waypoints.push_back(target);
                }
            } else {
                waypoints.waypoints.push_back(target);
            }

            // std::cout << waypoints.waypoints.size() << std::endl;

            Trajectory path = local_planner->plan_path(grid, start, target, waypoints);
            // Trajectory path = local_planner->plan_path(grid, start, target, Trajectory());

            if (path.cost >= prev_path_cost && dist < .3) {  // Worse path and hasn't moved too much
                return;
            }

            // std::cout << "Keeping path" << std::endl;

            current_local_plan = path;

            prev_path_cost = path.cost;
            second_iteration_passed = true;
            prev_start = start;

            // std::cout << "What 2" << std::endl;

            cev_msgs::msg::Trajectory current_plan;

            current_plan.header.stamp = msg->header.stamp;
            current_plan.header.frame_id = "map";
            current_plan.waypoints.clear();
            current_plan.timestep = path.timestep;

            // std::cout << "What 3" << std::endl;

            // std::cout << "Publishing path." << std::endl;

            for (State waypoint: path.waypoints) {
                cev_msgs::msg::Waypoint msg;
                msg.x = waypoint.pose.x;
                msg.y = waypoint.pose.y;

                // if (std::abs(waypoint.vel) < .3
                //     && std::abs(waypoint.vel) > 0.0) {  // Car cannot move that slow lol
                //     waypoint.vel = (waypoint.vel >= 0) ? .3 : -.3;
                // }

                msg.v = waypoint.vel;
                msg.theta = waypoint.pose.theta;
                msg.tau = waypoint.tau;
                current_plan.waypoints.push_back(msg);
            }

            // std::cout << "Publishing path" << std::endl;

            path_pub->publish(current_plan);

            // Publish local path
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

            local_path_pub->publish(nav_path);
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
        current_waypoint_in_global = 0;
        global_path_initialized = false;
        target = State{msg.x, msg.y, 0, msg.v, 0};
        prev_path_cost = 100000000;
        second_iteration_passed = false;
        target_initialized = true;
    }

    void rviz_target_callback(const geometry_msgs::msg::PoseStamped msg) {
        current_waypoint_in_global = 0;
        global_path_initialized = false;
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