#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <eigen3/Eigen/Dense>

#include <cev_msgs/msg/trajectory.hpp>
#include <global_planning/rrt.h>
#include <iostream>

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        auto dimensions = Dimensions();
        auto constraints = Constraints();
        planner = std::make_shared<global_planner::RRT>(dimensions, constraints);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        target_sub = this->create_subscription<cev_msgs::msg::Waypoint>("target", 1,
            std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("path", 1);
    }

private:
    Grid grid = Grid();
    Pose start = Pose();
    Pose target = Pose();

    bool map_initialized = false;
    bool odom_initialized = false;
    bool target_initialized = false;

    cev_msgs::msg::Trajectory current_path;

    std::shared_ptr<global_planner::RRT> planner;

    // Listener to the map
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;

    // Listener to odom
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;

    // Listener to the target
    rclcpp::Subscription<cev_msgs::msg::Waypoint>::SharedPtr target_sub;

    // Publisher for the planned path
    rclcpp::Publisher<cev_msgs::msg::Trajectory>::SharedPtr path_pub;

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        float qw = msg->pose.pose.orientation.w;
        float qx = msg->pose.pose.orientation.x;
        float qy = msg->pose.pose.orientation.y;
        float qz = msg->pose.pose.orientation.z;

        float yaw = restrict_angle(atan2(2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)));

        start = Pose{msg->pose.pose.position.x, msg->pose.pose.position.y, yaw};
        odom_initialized = true;

        std::cout << start.x << " , " << start.y << " , " << yaw << std::endl;

        if (map_initialized && target_initialized) {
            Trajectory path = planner->plan_path(grid, start, target);

            current_path.header.stamp = msg->header.stamp;
            current_path.header.frame_id = "odom";
            current_path.waypoints.clear();
            for (Waypoint waypoint: path.waypoints) {
                cev_msgs::msg::Waypoint msg;
                msg.x = waypoint.pose.x;
                msg.y = waypoint.pose.y;
                msg.v = 0.0;
                current_path.waypoints.push_back(msg);
            }

            path_pub->publish(current_path);
        }
    }

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        auto grid = Grid();
        grid.origin = Pose{msg->info.origin.position.x, msg->info.origin.position.y, 0};
        grid.resolution = msg->info.resolution;

        grid.data = Eigen::MatrixXd(msg->info.width, msg->info.height);

        for (int i = 0; i < msg->info.width; i++) {
            for (int j = 0; j < msg->info.height; j++) {
                // Divide by 100 to get probability of occupancy in the range [0, 1]
                // Unknown is -1
                grid.data(i, j) = msg->data[i * msg->info.width + j] / 100.0;
            }
        }

        map_initialized = true;
    }

    void target_callback(const cev_msgs::msg::Waypoint msg) {
        target = Pose{msg.x, msg.y, 0};
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