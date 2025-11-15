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
#include <limits>

#include "local_planning/mpc.h"
#include "global_planning/rrt.h"
#include "cost_map/gaussian_conv.h"
#include "cost_map/nearest.h"
#include "cost_map/nothing.h"
#include "cost_map/dist_map.h"
#include "cost_finder/cost_finder.h"
#include "cev_msgs/srv/query_costmap.hpp"

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        Dimensions dimensions = Dimensions{.3, .3, .3};
        full_constraints = Constraints{
            {-1000.0, 1000.0},  // x
            {-1000.0, 1000.0},  // y
            {-.586, .586},      // tau
            {-2.2, 2.2},        // vel
            {-2.5, 2.5},        // accel
            {-.785, .785}       // dtau
        };

        auto safe_load = [&](auto& target, const char* name) {
            try {
                this->declare_parameter(name, rclcpp::PARAMETER_DOUBLE_ARRAY);
                rclcpp::Parameter param = this->get_parameter(name);
                auto v = param.as_double_array();
                target[0] = v[0];
                target[1] = v[1];
            } 
            catch (...) { RCLCPP_WARN(this->get_logger(), "Failed to load %s constraint, using default values (%f, %f).", name, target[0], target[1]); }
        };

        safe_load(full_constraints.x, "x");
        safe_load(full_constraints.y, "y");
        safe_load(full_constraints.tau, "tau");
        safe_load(full_constraints.vel, "vel");
        safe_load(full_constraints.accel, "accel");
        safe_load(full_constraints.dtau, "dtau");

        local_planner = std::make_shared<local_planner::LaneFollowingMPC>(dimensions, full_constraints);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        trajectory_sub = this->create_subscription<cev_msgs::msg::Trajectory>("/igvc_waypoints", 1,
            std::bind(&PlannerNode::waypoints_callback, this, std::placeholders::_1));

        lane_centerline_sub = this->create_subscription<cev_msgs::msg::Trajectory>("/igvc_lane", 1,
            std::bind(&PlannerNode::lane_centerline_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("trajectory", 1);

        local_path_pub = this->create_publisher<nav_msgs::msg::Path>("local_path", 1);

        target_rviz_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("goal_pose", 1,
            std::bind(&PlannerNode::rviz_target_callback, this, std::placeholders::_1));

        costmap_query_service_ = this->create_service<cev_msgs::srv::QueryCostmap>(
            "/query_costmap",
            std::bind(&PlannerNode::handle_costmap_query, this,
                std::placeholders::_1, std::placeholders::_2));
    }

private:
    Constraints full_constraints;
    Grid grid = Grid();
    State start = State();
    State goal_state = State();

    std::shared_ptr<cost_finder::CostFinder> local_plan_cost =
        std::make_shared<cost_finder::CostFinder>(5, 10);
    // cost_map::NearestGenerator local_plan_cost_generator = cost_map::NearestGenerator(3.5, 1);
    // std::shared_ptr<cost_map::CostMap> local_plan_cost;

    bool map_initialized = false;
    bool mission_active = false;

    std::vector<State> waypoint_plan;
    int current_waypoint_index = 0;
    Trajectory last_path = Trajectory();
    float prev_path_cost = 100000000;

    std::shared_ptr<local_planner::LaneFollowingMPC> local_planner;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cev_msgs::msg::Waypoint>::SharedPtr target_sub;
    rclcpp::Subscription<cev_msgs::msg::Trajectory>::SharedPtr trajectory_sub;
    rclcpp::Subscription<cev_msgs::msg::Trajectory>::SharedPtr lane_centerline_sub;

    rclcpp::Publisher<cev_msgs::msg::Trajectory>::SharedPtr path_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub;
    rclcpp::Service<cev_msgs::srv::QueryCostmap>::SharedPtr costmap_query_service_;

    Trajectory lane_centerline_;
    int lane_reference_index_ = 0;

    float avg_planning_time = 0;
    std::chrono::_V2::system_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    // To receive IGVC waypoints
    void waypoints_callback(const cev_msgs::msg::Trajectory::SharedPtr msg) {
        if (!msg) {
            return;
        }

        std::vector<State> new_waypoints;
        new_waypoints.reserve(msg->waypoints.size());

        for (const auto& waypoint_msg: msg->waypoints) {
            State waypoint;
            waypoint.pose.x = waypoint_msg.x;
            waypoint.pose.y = waypoint_msg.y;
            waypoint.pose.theta = waypoint_msg.theta;
            waypoint.tau = waypoint_msg.tau;
            // waypoint.vel = waypoint_msg.v;
            waypoint.vel = full_constraints.vel[1];
            new_waypoints.push_back(waypoint);
        }

        if (new_waypoints.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty trajectory on /igvc_waypoints.");
        }

        update_mission(std::move(new_waypoints));
    }

    void lane_centerline_callback(const cev_msgs::msg::Trajectory::SharedPtr msg) {
        lane_centerline_ = Trajectory();
        lane_reference_index_ = 0;
        if (!msg) {
            return;
        }

        lane_centerline_.waypoints.clear();
        lane_centerline_.waypoints.reserve(msg->waypoints.size());
        for (const auto& wp: msg->waypoints) {
            State lane_wp;
            lane_wp.pose.x = wp.x;
            lane_wp.pose.y = wp.y;
            lane_wp.pose.theta = wp.theta;
            lane_centerline_.waypoints.push_back(lane_wp);
        }
    }

    void handle_costmap_query(
        const std::shared_ptr<cev_msgs::srv::QueryCostmap::Request> request,
        std::shared_ptr<cev_msgs::srv::QueryCostmap::Response> response) {
        if (!local_plan_cost) {
            response->success = false;
            response->message = "costmap unavailable";
            response->cost = 0.0;
            return;
        }

        State probe;
        probe.pose.x = request->x;
        probe.pose.y = request->y;
        probe.pose.theta = request->theta;
        probe.tau = 0.0;
        probe.vel = 0.0;

        try {
            const double value = local_plan_cost->cost(probe);
            if (value >= std::numeric_limits<double>::max() * 0.5) {
                response->success = false;
                response->message = "out_of_bounds";
                response->cost = 0.0;
                return;
            }

            response->cost = value;
            response->success = true;
            response->message.clear();
        } catch (const std::exception& ex) {
            response->success = false;
            response->message = ex.what();
            response->cost = 0.0;
        }
    }

    // main callback
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Transform the odom message into the map frame
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

        double qw = map_pose.pose.orientation.w;
        double qx = map_pose.pose.orientation.x;
        double qy = map_pose.pose.orientation.y;
        double qz = map_pose.pose.orientation.z;

        double yaw = restrict_angle(atan2(2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)));

        // set start to current pose
        start.pose.x = map_pose.pose.position.x;
        start.pose.y = map_pose.pose.position.y;
        start.pose.theta = yaw;
        start.tau = 0.0;
        start.vel = msg->twist.twist.linear.x;

        // Quit if not ready
        if (!map_initialized || waypoint_plan.empty() || !mission_active || !local_plan_cost) {
            return;
        }

        int waypoint_count = waypoint_plan.size();
        goal_state = waypoint_plan.back();
        goal_state.vel = 0.0;

        // determine current waypoint
        const double waypoint_threshold = 0.3;
        if (waypoint_count > 1) {
            while (current_waypoint_index < waypoint_count - 1) {
                double dist = start.pose.distance_to(waypoint_plan[current_waypoint_index].pose);
                if (dist < waypoint_threshold) {
                    current_waypoint_index += 1;
                    prev_path_cost = 100000000;
                } else {
                    break;
                }
            }
        }

        // Check if we are targeting the goal state
        const double goal_threshold = 0.3;
        double dist_to_goal = start.pose.distance_to(goal_state.pose);
        bool passed_goal = false;


        bool targeting_goal = current_waypoint_index >= waypoint_count - 1;

        if (targeting_goal && (dist_to_goal < goal_threshold || passed_goal)) {
            // Publish a stop trajectory and set mission_active to false
            publish_stop_trajectory(msg->header.stamp);
            mission_active = false;
            last_path = Trajectory();
            prev_path_cost = 100000000;
            current_waypoint_index = waypoint_count;
            if (passed_goal) {
                RCLCPP_INFO(this->get_logger(), "Passed final IGVC waypoint, stopping mission.");
            }
            return;
        }

        // Get the target index
        int target_index =
            (current_waypoint_index >= waypoint_count) ? waypoint_count - 1
                                                       : current_waypoint_index;
        bool planning_goal = target_index == waypoint_count - 1;

        State planning_target = waypoint_plan[target_index];
        if (planning_goal) {
            planning_target.vel = 0.0;
        }

        // Create a waypoint target message
        Trajectory waypoint_targets;
        waypoint_targets.waypoints.clear();
        waypoint_targets.cost = 0;
        waypoint_targets.timestep = 0;

        // Add the current waypoint to the waypoint targets message
        waypoint_targets.waypoints.push_back(planning_target);
        // If the target index is less than the waypoint count minus one,
        // add the next waypoint to the waypoint targets message
        if (target_index + 1 < waypoint_count) {
            waypoint_targets.waypoints.push_back(waypoint_plan[target_index + 1]);
        }

        if (!lane_centerline_.waypoints.empty()) {
            auto nearest_lane_point = nearest_waypoint(start.pose, lane_centerline_);
            lane_reference_index_ = nearest_lane_point.first;

            
            int n_ = std::min(6, static_cast<int>(lane_centerline_.waypoints.size())-lane_reference_index_);
            if (n_ < 2) {
                RCLCPP_DEBUG(this->get_logger(),
                    "Not enough lane centerline points to fit polynomial (n=%d).", n_);
                publish_stop_trajectory(msg->header.stamp);
                return;
            }

            local_planner->set_reference_polynomial(lane_centerline_, lane_reference_index_, n_);
        }

        // Plan a path using the local planner
        Trajectory path = local_planner->plan_path(grid, start, waypoint_targets.waypoints.at(current_waypoint_index), waypoint_targets,
            last_path, local_plan_cost);

        // Just use old path if new path has worse cost and we still near beginning of old path
        if (path.cost > prev_path_cost) {
            if (last_path.waypoints.empty()) {
                return;
            } else {
                auto nearest = nearest_waypoint(start.pose, last_path);
                if (nearest.first <= 2 &&
                    nearest.second <= 3 * last_path.waypoints[1].pose.distance_to(last_path.waypoints[0].pose)) {
                    return;
                }
            }
        }

        // Update the planning time
        auto end_time = std::chrono::high_resolution_clock::now();
        avg_planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time
                                 - start_time)
                                .count();

        std::cout << "Planning Update Time: " << avg_planning_time << "ms" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();

        // Update the last path
        last_path = path;
        prev_path_cost = path.cost;

        // Publish the planned path
        cev_msgs::msg::Trajectory current_plan;

        current_plan.header.stamp = msg->header.stamp;
        current_plan.header.frame_id = "map";
        current_plan.waypoints.clear();
        current_plan.timestep = path.timestep;

        for (const State& waypoint: path.waypoints) {
            cev_msgs::msg::Waypoint waypoint_msg;
            waypoint_msg.x = waypoint.pose.x;
            waypoint_msg.y = waypoint.pose.y;
            waypoint_msg.v = waypoint.vel;
            waypoint_msg.theta = waypoint.pose.theta;
            waypoint_msg.tau = waypoint.tau;
            current_plan.waypoints.push_back(waypoint_msg);
        }

        path_pub->publish(current_plan);

        // Publish the planned path as a nav_msgs::msg::Path
        nav_msgs::msg::Path nav_path;
        nav_path.header.stamp = msg->header.stamp;
        nav_path.header.frame_id = "map";
        nav_path.poses.clear();

        for (const State& waypoint: path.waypoints) {
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

    // generate costmap
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
        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                double x = grid.origin.x + i * grid.resolution;
                double y = grid.origin.y + j * grid.resolution;
                if (local_plan_cost) {
                    local_plan_cost->addPoint(State{x, y});
                }
            }
        }
    }

    void target_callback(const cev_msgs::msg::Waypoint msg) {
        State waypoint;
        waypoint.pose.x = msg.x;
        waypoint.pose.y = msg.y;
        waypoint.pose.theta = msg.theta;
        waypoint.tau = msg.tau;
        waypoint.vel = msg.v;

        std::vector<State> mission{waypoint};
        update_mission(std::move(mission));
    }

    void rviz_target_callback(const geometry_msgs::msg::PoseStamped msg) {
        State waypoint;
        waypoint.pose.x = msg.pose.position.x;
        waypoint.pose.y = msg.pose.position.y;
        waypoint.pose.theta = tf2::getYaw(msg.pose.orientation);
        waypoint.tau = 0;
        waypoint.vel = 0;

        std::vector<State> mission{waypoint};
        update_mission(std::move(mission));
    }

    void update_mission(std::vector<State>&& new_waypoints) {
        waypoint_plan = std::move(new_waypoints);
        current_waypoint_index = 0;
        last_path = Trajectory();
        prev_path_cost = 100000000;

        if (waypoint_plan.empty()) {
            mission_active = false;
            goal_state = State();
            return;
        }

        waypoint_plan.back().vel = 0.0;
        goal_state = waypoint_plan.back();
        mission_active = true;

        RCLCPP_INFO(this->get_logger(), "Loaded IGVC trajectory with %zu waypoint(s).",
            waypoint_plan.size());
    }

    // bool path_hits_obstacle(const Trajectory& path) const {
    //     if (!local_plan_cost) {
    //         return false;
    //     }

    //     for (const State& waypoint: path.waypoints) {
    //         if (local_plan_cost->cost(waypoint) > .5) {
    //             return true;
    //         }
    //     }

    //     return false;
    // }

    // (index, distance)
    std::pair<int, double> nearest_waypoint(const Pose& pose, const Trajectory& path) {
        int nearest_index = -1;
        double nearest_distance = std::numeric_limits<double>::infinity();

        for (int i = 0; i < path.waypoints.size(); ++i) {
            double dist = pose.distance_to(path.waypoints[i].pose);
            if (dist < nearest_distance) {
                nearest_distance = dist;
                nearest_index = i;
            }
        }

        return {nearest_index, nearest_distance};
    }

    void publish_stop_trajectory(const rclcpp::Time& stamp) {
        cev_msgs::msg::Trajectory stop_plan;
        stop_plan.header.stamp = stamp;
        stop_plan.header.frame_id = "map";

        cev_msgs::msg::Waypoint waypoint_msg;
        waypoint_msg.x = goal_state.pose.x;
        waypoint_msg.y = goal_state.pose.y;
        waypoint_msg.theta = goal_state.pose.theta;
        waypoint_msg.v = 0.0;
        waypoint_msg.tau = 0.0;

        stop_plan.waypoints.push_back(waypoint_msg);
        stop_plan.timestep = 0.0;

        path_pub->publish(stop_plan);

        RCLCPP_INFO(this->get_logger(), "Reached final IGVC waypoint.");
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
