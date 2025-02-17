#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class OdomToMapTransformer : public rclcpp::Node {
public:
    OdomToMapTransformer()
        : Node("odom_to_map_transformer"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        RCLCPP_INFO(this->get_logger(), "starting...");
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&OdomToMapTransformer::odom_callback, this, std::placeholders::_1));

        transformed_pose_pub_ =
            this->create_publisher<geometry_msgs::msg::PoseStamped>("/base_link_in_map", 1);
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::PoseStamped base_link_pose;
        base_link_pose.header = msg->header;
        base_link_pose.pose = msg->pose.pose;

        // Transform from "odom" to "map"
        try {
            geometry_msgs::msg::TransformStamped transform_stamped =
                tf_buffer_.lookupTransform("map", "odom", tf2::TimePointZero);

            geometry_msgs::msg::PoseStamped map_pose;
            tf2::doTransform(base_link_pose, map_pose, transform_stamped);

            // Publish transformed pose in "map" frame
            transformed_pose_pub_->publish(map_pose);
            RCLCPP_INFO(this->get_logger(), "Published transformed pose in map frame");
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Could not transform odom to map: %s", ex.what());
        }
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr transformed_pose_pub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomToMapTransformer>());
    rclcpp::shutdown();
    return 0;
}
