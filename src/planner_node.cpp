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
#include <memory>

#include "local_planning/mpc.h"
#include "global_planning/rrt.h"
#include "cost_map/gaussian_conv.h"
#include "cost_map/nearest.h"
#include "cost_map/nothing.h"
#include "cost_map/dist_map.h"
#include "cost_finder/cost_finder.h"

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
            {-1.0, 1.0},    // vel
            {-.5, .5},      // accel
            {-.20, .20}     // dtau
        };

        // local_planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints,
        //     std::make_shared<cost_map::Nothing>(2, .5));

        local_planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints);
        global_planner = std::make_shared<global_planner::RRT>(dimensions, full_constraints);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("trajectory", 1);

        // Global plan every 2 seconds
        // global_plan_timer = this->create_wall_timer(std::chrono::milliseconds(2000),
        //     std::bind(&PlannerNode::global_plan_callback, this));

        // RVIZ Debug
        global_path_pub = this->create_publisher<nav_msgs::msg::Path>("global_path", 1);
        local_path_pub = this->create_publisher<nav_msgs::msg::Path>("local_path", 1);

        target_rviz_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("goal_pose", 1,
            std::bind(&PlannerNode::rviz_target_callback, this, std::placeholders::_1));

        // cost_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("planner/cost_map", 1);
    }

private:
    Grid grid = Grid();
    State start = State();
    State prev_start = State();
    State target = State();

    //cost_map::NearestGenerator local_plan_cost_generator = cost_map::NearestGenerator(2, .5);
    // cost_map::DistGenerator local_plan_cost_generator = cost_map::DistGenerator(10, 20);
    std::shared_ptr<cost_finder::CostFinder> local_plan_cost =
        std::make_shared<cost_finder::CostFinder>(10, 20);
    cost_map::Nothing global_plan_cost_generator = cost_map::Nothing(1, .5);

    // std::shared_ptr<cost_map::CostMap> local_plan_cost;
    // std::shared_ptr<cev_planner::cost_finder::CostFinder> local_plan_cost;
    std::shared_ptr<cost_map::CostMap> global_plan_cost;
    bool cost_map_initialized = false;

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
    int current_waypoint_in_global = 0;
    Trajectory last_path = Trajectory();

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
    // rclcpp::TimerBase::SharedPtr global_plan_timer;

    // RVIZ
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub;

    float avg_costmap_time = 0;
    int cost_map_iters = 0;
    float avg_planning_time = 0;
    int planning_iters = 0;

    std::chrono::_V2::system_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();
    // -------------------------------

    void global_plan_callback() {
        std::cout << "I am attempting a global plan." << std::endl;

        std::optional<Trajectory> optional = global_planner->plan_path(grid, start, target);

        if (!optional.has_value()) {
            std::cout << "Global Path Failed." << std::endl;
            global_path_initialized = false;
        } else {
            std::cout << "Global Path Planned." << std::endl;
            global_path.waypoints = optional.value().waypoints;

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

    bool passed_waypoint(State state, State waypoint, State prev_waypoint, State next_waypoint,
        bool last_waypoint) {
        float dot;

        if (last_waypoint) {
            dot = (state.pose.x - waypoint.pose.x) * (waypoint.pose.x - prev_waypoint.pose.x)
                  + (state.pose.y - waypoint.pose.y) * (waypoint.pose.y - prev_waypoint.pose.y);
        } else {
            dot = (state.pose.x - waypoint.pose.x) * (next_waypoint.pose.x - waypoint.pose.x)
                  + (state.pose.y - waypoint.pose.y) * (next_waypoint.pose.y - waypoint.pose.y);
        }

        float dist = state.pose.distance_to(waypoint.pose);

        return dist < .5 || (dist < .7 && dot > 0);
    }

    bool hits_obstacle(Trajectory path) {
        for (State waypoint: path.waypoints) {
            if (global_plan_cost->cost(waypoint) > .5) {
                return true;
            }
        }

        return false;
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

        // Plan path
        if (map_initialized && odom_initialized && target_initialized && !global_path_initialized) {
            // std::cout << "Planning" << std::endl;
            global_plan_callback();
        } else {
            // std::cout << "Map: " << map_initialized << std::endl;
            // std::cout << "Odom: " << odom_initialized << std::endl;
            // std::cout << "Target: " << target_initialized << std::endl;
            // std::cout << "Global Path: " << global_path_initialized << std::endl;
        }

        // float moved_dist = start.pose.distance_to(prev_start.pose);

        if (map_initialized && odom_initialized && target_initialized && global_path_initialized) {
            // std::cout << "I am entering the local planning loop." << std::endl;
            float dist =
                start.pose.distance_to(global_path.waypoints[current_waypoint_in_global].pose);

            float dist_to_dest = start.pose.distance_to(target.pose);

            bool passed = passed_waypoint(start, target,
                global_path.waypoints[global_path.waypoints.size() - 2], State(), true);

            if (dist_to_dest < .3) {  // Reached destination
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
                msg.tau = 0;
                current_plan.waypoints.push_back(msg);

                path_pub->publish(current_plan);

                target_initialized = false;

                std::cout << "Passed target." << std::endl;

                return;
            }

            // std::cout << "I did not pass the target" << std::endl;

            while (dist < .75
                   && current_waypoint_in_global
                          < global_path.waypoints.size()) {  // Progress waypoint
                current_waypoint_in_global += 1;
                prev_path_cost = 100000000;                  // Reset plan costs for new waypoint
                dist =
                    start.pose.distance_to(global_path.waypoints[current_waypoint_in_global].pose);
            }

            // Give next two waypoints to target for planner
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

            // Fast forward last path to current position
            // for (int i = 0; i < last_path.waypoints.size(); i++) {
            //     float dist_to_waypoint = start.pose.distance_to(last_path.waypoints[i].pose);
            //     if (dist_to_waypoint < .5) {
            //         last_path.waypoints.erase(last_path.waypoints.begin(),
            //             last_path.waypoints.begin() + i);
            //         break;
            //     }
            // }

            // std::cout << "Planning local path" << std::endl;
            Trajectory path = local_planner->plan_path(grid, start, target, waypoints, last_path,
                local_plan_cost);

            if (path.cost >= prev_path_cost
                || hits_obstacle(path)) {  // Worse path and hasn't targeted next waypoint
                // std::cout << "Worse path" << std::endl;
                return;
            }

            std::chrono::_V2::system_clock::time_point end_time =
                std::chrono::high_resolution_clock::now();

            avg_planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time
                                                                                      - start_time)
                                    .count();

            std::cout << "Planning Update Time: " << avg_planning_time << "ms" << std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            // std::cout << "Better path" << std::endl;

            last_path = path;

            prev_path_cost = path.cost;
            second_iteration_passed = true;
            prev_start = start;

            cev_msgs::msg::Trajectory current_plan;

            current_plan.header.stamp = msg->header.stamp;
            current_plan.header.frame_id = "map";
            current_plan.waypoints.clear();
            current_plan.timestep = path.timestep;

            for (State waypoint: path.waypoints) {
                cev_msgs::msg::Waypoint msg;
                msg.x = waypoint.pose.x;
                msg.y = waypoint.pose.y;
                msg.v = waypoint.vel;
                msg.theta = waypoint.pose.theta;
                msg.tau = waypoint.tau;
                current_plan.waypoints.push_back(msg);
            }

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

        // std::cout << "map_callback: grid size (rows,cols): " << grid.data.rows() << "," << grid.data.cols() << std::endl;

        // auto start_time = std::chrono::high_resolution_clock::now();
        // local_plan_cost = local_plan_cost_generator.generate_cost_map(grid);
        // local_plan_cost = std::make_shared<cev_planner::cost_finder::CostFinder>(10, 20);
        
        nav_msgs::msg::OccupancyGrid cost_map_msg;
        cost_map_msg.header.stamp = this->now();
        cost_map_msg.header.frame_id = "map";
        cost_map_msg.info.resolution = grid.resolution;
        cost_map_msg.info.width = grid.data.cols();
        cost_map_msg.info.height = grid.data.rows();
        cost_map_msg.info.origin.position.x = grid.origin.x;
        cost_map_msg.info.origin.position.y = grid.origin.y;
        cost_map_msg.info.origin.position.z = 0;
        cost_map_msg.info.origin.orientation.x = 0;
        cost_map_msg.info.origin.orientation.y = 0;
        cost_map_msg.info.origin.orientation.z = 0;
        cost_map_msg.info.origin.orientation.w = 1;

        cost_map_msg.data.clear();

        /*
        int largest_val = 0;
        
        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                std::cout << "Check? " << i << " " << j << " " << grid.data(i, j) << " " << largest_val << "\n";
                if (grid.data(i, j) > largest_val) {
                    largest_val = grid.data(i, j);
                }
            }
        }
        */

        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                // std::cout << i*grid.resolution+grid.origin.x << " " << j*grid.resolution+grid.origin.y << "\n";
                //if (i >= 80) std::cout << "Cost Normalized " << i << " " << j << " " << local_plan_cost->cost({i, j}) << "\n"; 
                //cost_map_msg.data.push_back((local_plan_cost->cost({i, j})/largest_val));
                // cost_map_msg.data.push_back(local_plan_cost->cost({i, j}));
                if (grid.data(i, j) > 0.7f) {  // Threshold for occupied cells
                    double x = grid.origin.x + i * grid.resolution;
                    double y = grid.origin.y + j * grid.resolution;
                    if (local_plan_cost) {
                        local_plan_cost->addPoint(State{x, y});
                    }
                }
                                                  
                // std::cout<<local_plan_cost->cost({i*grid.resolution+grid.origin.x, j*grid.resolution+grid.origin.y})<<std::endl;
            }
        }
        nav_msgs::msg::OccupancyGrid mirrored_msg;
        mirrored_msg.header.stamp = this->now();
        mirrored_msg.header.frame_id = "map";
        mirrored_msg.info.resolution = grid.resolution;
        mirrored_msg.info.width = grid.data.rows();
        mirrored_msg.info.height = grid.data.cols();
        mirrored_msg.info.origin.position.z = 0;
        mirrored_msg.info.origin.orientation.x = 0;
        mirrored_msg.info.origin.orientation.y = 0;
        mirrored_msg.info.origin.orientation.z = 0;
        mirrored_msg.info.origin.orientation.w = 1;

        mirrored_msg.data.clear();

        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                // mirrored_msg.data.push_back(0);
                double x = grid.origin.x + i * grid.resolution;
                double y = grid.origin.y + j * grid.resolution;
                double c = 0.0;
                if (local_plan_cost) {
                    try {
                        c = local_plan_cost->cost(State{x, y});
                    } catch (const std::exception &e) {
                        std::cerr << "CostFinder::cost threw: " << e.what() << std::endl;
                        c = 0.0;
                    }
                }
                cost_map_msg.data.push_back(c);
            }
        }

        // // Debug visualization code - commented out
        // mirrored_msg.data.resize(cost_map_msg.info.height * cost_map_msg.info.width, 0);
        // for (int y = 0; y < cost_map_msg.info.height; ++y) {
        //     for (int x = 0; x < cost_map_msg.info.width; ++x) {
        //         int new_x = y;
        //         int new_y = x;
        //         mirrored_msg.data[new_y * cost_map_msg.info.height + new_x] =
        //             cost_map_msg.data[y * cost_map_msg.info.width + x];
        //     }
        // }
        // mirrored_msg.info.origin.position.x = cost_map_msg.info.origin.position.x;
        // mirrored_msg.info.origin.position.y = cost_map_msg.info.origin.position.y;
        // cost_map_pub_->publish(mirrored_msg);

        global_plan_cost = global_plan_cost_generator.generate_cost_map(grid);
        // auto end_time = std::chrono::high_resolution_clock::now();
        // avg_costmap_time +=
        //     std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // cost_map_iters += 1;

        // if (cost_map_iters > 100) {
        //     std::cout << "Average cost map generation time: " << avg_costmap_time /
        //     cost_map_iters
        //               << "ms" << std::endl;
        //     avg_costmap_time = 0;
        //     cost_map_iters = 0;
        // }
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
        std::cout << "I received a target." << std::endl;

        current_waypoint_in_global = 0;
        global_path_initialized = false;
        target = State{msg.pose.position.x, msg.pose.position.y, 0, 0, 0};
        prev_path_cost = 100000000;
        second_iteration_passed = false;
        target_initialized = true;
    }
    // rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}