#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <eigen3/Eigen/Dense>
#include <cev_msgs/msg/trajectory.hpp>
#include <cev_msgs/msg/waypoint.hpp>
#include <std_msgs/msg/string.hpp>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2/utils.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#include "local_planning/mpc.h"

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        Dimensions dimensions = Dimensions{.6096, .9144, .4699};  // width, length, wheelbase — matching sim vehicle
        Constraints full_constraints_init = Constraints{
            {-1000.0, 1000.0},  // x
            {-1000.0, 1000.0},  // y
            {-0.5879, 0.5879},  // tau (steering angle) — full vehicle range: 33.6° = 0.5879 rad
            {-2.2352, 2.2352},  // vel
            {-2.5, 2.5},        // accel
            {-1.0, 1.0}         // dtau (steering rate)
        };
        full_constraints = full_constraints_init;

        auto safe_load = [&](auto& target, const char* name) {
            try {
                this->declare_parameter(name, rclcpp::PARAMETER_DOUBLE_ARRAY);
                rclcpp::Parameter param = this->get_parameter(name);
                auto v = param.as_double_array();
                target[0] = v[0];
                target[1] = v[1];
            }
            catch (...) {
                RCLCPP_WARN(this->get_logger(),
                    "Failed to load %s constraint, using default (%f, %f).",
                    name, target[0], target[1]);
            }
        };

        safe_load(full_constraints.x, "x");
        safe_load(full_constraints.y, "y");
        safe_load(full_constraints.tau, "tau");
        safe_load(full_constraints.vel, "vel");
        safe_load(full_constraints.accel, "accel");
        safe_load(full_constraints.dtau, "dtau");

        local_planner = std::make_shared<local_planner::CartesianMPC>(dimensions, full_constraints);

        map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>("map", 1,
            std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 1,
            std::bind(&PlannerNode::odom_callback, this, std::placeholders::_1));

        trajectory_sub = this->create_subscription<cev_msgs::msg::Trajectory>("/igvc_waypoints", 1,
            std::bind(&PlannerNode::waypoints_callback, this, std::placeholders::_1));

        sim_state_sub = this->create_subscription<cev_msgs::msg::Waypoint>("sim_state", 1,
            std::bind(&PlannerNode::sim_state_callback, this, std::placeholders::_1));

        path_pub = this->create_publisher<cev_msgs::msg::Trajectory>("trajectory", 1);
        local_path_pub = this->create_publisher<nav_msgs::msg::Path>("local_path", 1);
        mode_pub = this->create_publisher<std_msgs::msg::String>("planner_mode", 1);
        lane_cl_pub = this->create_publisher<nav_msgs::msg::Path>("detected_lane_cl", 1);
        cost_field_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("cost_field", 1);

        {
            auto lane_qos = rclcpp::QoS(1).transient_local().reliable();
            lane_left_sub = this->create_subscription<nav_msgs::msg::Path>(
                "lane/left", lane_qos,
                [this](nav_msgs::msg::Path::SharedPtr msg) {
                    lane_left_pts.clear();
                    for (auto& ps : msg->poses)
                        lane_left_pts.push_back({ps.pose.position.x, ps.pose.position.y});
                    lane_boundaries_received = true;
                });
            lane_right_sub = this->create_subscription<nav_msgs::msg::Path>(
                "lane/right", lane_qos,
                [this](nav_msgs::msg::Path::SharedPtr msg) {
                    lane_right_pts.clear();
                    for (auto& ps : msg->poses)
                        lane_right_pts.push_back({ps.pose.position.x, ps.pose.position.y});
                });
            lane_cl_acc_sub = this->create_subscription<nav_msgs::msg::Path>(
                "lane/centerline", lane_qos,
                [this](nav_msgs::msg::Path::SharedPtr msg) {
                    acc_lane_cl_pts.clear();
                    acc_lane_cl_pts.reserve(msg->poses.size());
                    for (auto& ps : msg->poses)
                        acc_lane_cl_pts.push_back({ps.pose.position.x, ps.pose.position.y});
                });
        }

        target_rviz_sub = this->create_subscription<geometry_msgs::msg::PoseStamped>("goal_pose", 1,
            std::bind(&PlannerNode::rviz_target_callback, this, std::placeholders::_1));
    }

private:
    struct LocalPt { double fwd, lat; };
    static constexpr bool ENABLE_EXTRAPOLATED_LANE_COST = false;

    Constraints full_constraints;
    Grid grid = Grid();
    Eigen::MatrixXf base_cost_field;
    State start = State();
    State goal_state = State();
    nav_msgs::msg::MapMetaData last_map_info;
    std::string last_map_frame_id = "map";

    bool map_initialized = false;
    bool mission_active = false;

    std::vector<State> waypoint_plan;
    int current_waypoint_index = 0;
    Trajectory last_path = Trajectory();

    std::shared_ptr<local_planner::CartesianMPC> local_planner;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<cev_msgs::msg::Trajectory>::SharedPtr trajectory_sub;
    rclcpp::Subscription<cev_msgs::msg::Waypoint>::SharedPtr sim_state_sub;
    rclcpp::Publisher<cev_msgs::msg::Trajectory>::SharedPtr path_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr lane_cl_pub;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_field_pub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub;

    // Camera-based lane boundary topics (sim: white polygon medial axes)
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr lane_left_sub;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr lane_right_sub;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr lane_cl_acc_sub;
    std::vector<std::pair<double,double>> lane_left_pts;
    std::vector<std::pair<double,double>> lane_right_pts;
    std::vector<std::pair<double,double>> acc_lane_cl_pts;
    bool lane_boundaries_received = false;

    std::string current_mode = "GPS";
    std::string current_mode_data = "GPS";

    static constexpr int LANE_THRESHOLD    = 5;
    static constexpr int NO_LANE_THRESHOLD = 10;
    int consecutive_lane_frames    = 0;
    int consecutive_no_lane_frames = 0;


    bool in_gps_pair_active = false;

    float avg_planning_time = 0;
    std::chrono::_V2::system_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    void sim_state_callback(const cev_msgs::msg::Waypoint::SharedPtr msg) {
        start.tau = msg->tau;
    }

    void collect_lane_boundary_local(std::vector<LocalPt>& left_local,
                                     std::vector<LocalPt>& right_local) const {
        left_local.clear();
        right_local.clear();
        if (!lane_boundaries_received) return;

        double cos_h = std::cos(start.pose.theta);
        double sin_h = std::sin(start.pose.theta);

        auto classify = [&](double wx, double wy) {
            double dx = wx - start.pose.x;
            double dy = wy - start.pose.y;
            double fwd = cos_h * dx + sin_h * dy;
            double lat = -sin_h * dx + cos_h * dy;  // positive = left
            if (fwd < -0.5 || fwd > 6.0) return;
            if (lat > 0.15 && lat < 3.0)        left_local.push_back({fwd, lat});
            else if (lat < -0.15 && lat > -3.0) right_local.push_back({fwd, lat});
        };

        for (const auto& p : lane_left_pts)  classify(p.first, p.second);
        for (const auto& p : lane_right_pts) classify(p.first, p.second);

        auto cmp = [](const LocalPt& a, const LocalPt& b) { return a.fwd < b.fwd; };
        std::sort(left_local.begin(), left_local.end(), cmp);
        std::sort(right_local.begin(), right_local.end(), cmp);
    }

    static double interp_lat(const std::vector<LocalPt>& pts, double fwd) {
        if (pts.empty() || fwd < pts.front().fwd || fwd > pts.back().fwd)
            return std::numeric_limits<double>::quiet_NaN();
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            if (pts[i].fwd <= fwd && pts[i+1].fwd >= fwd) {
                double t = (fwd - pts[i].fwd) / (pts[i+1].fwd - pts[i].fwd + 1e-12);
                return pts[i].lat + t * (pts[i+1].lat - pts[i].lat);
            }
        }
        return std::numeric_limits<double>::quiet_NaN();
    }

    static double fit_tail_slope(const std::vector<LocalPt>& pts) {
        int count = std::min<int>(4, pts.size());
        if (count < 2) return 0.0;

        double mean_f = 0.0, mean_l = 0.0;
        for (int i = static_cast<int>(pts.size()) - count; i < static_cast<int>(pts.size()); ++i) {
            mean_f += pts[i].fwd;
            mean_l += pts[i].lat;
        }
        mean_f /= count;
        mean_l /= count;

        double num = 0.0, den = 0.0;
        for (int i = static_cast<int>(pts.size()) - count; i < static_cast<int>(pts.size()); ++i) {
            double df = pts[i].fwd - mean_f;
            double dl = pts[i].lat - mean_l;
            num += df * dl;
            den += df * df;
        }
        if (den < 1e-6) return 0.0;
        return std::clamp(num / den, -0.7, 0.7);
    }

    void add_virtual_cost_point(double wx, double wy, float peak,
                                double sigma_m = 0.22, double cutoff_m = 0.75) {
        if (grid.cost_field.size() == 0 || peak <= 0.0f) return;

        int cx = static_cast<int>(std::round((wx - grid.origin.x) / grid.resolution));
        int cy = static_cast<int>(std::round((wy - grid.origin.y) / grid.resolution));
        int radius = static_cast<int>(cutoff_m / grid.resolution) + 1;
        float inv2s2 = static_cast<float>(-0.5 / (sigma_m * sigma_m));
        float cutoff2 = static_cast<float>(cutoff_m * cutoff_m);

        for (int i = cx - radius; i <= cx + radius; ++i) {
            if (i < 0 || i >= grid.cost_field.rows()) continue;
            for (int j = cy - radius; j <= cy + radius; ++j) {
                if (j < 0 || j >= grid.cost_field.cols()) continue;
                double gx = grid.origin.x + i * grid.resolution;
                double gy = grid.origin.y + j * grid.resolution;
                float d2 = static_cast<float>((gx - wx) * (gx - wx) + (gy - wy) * (gy - wy));
                if (d2 > cutoff2) continue;
                float c = peak * std::exp(inv2s2 * d2);
                if (c > grid.cost_field(i, j))
                    grid.cost_field(i, j) = std::min(1.0f, c);
            }
        }
    }

    void overlay_extrapolated_lane_cost(double lane_width) {
        std::vector<LocalPt> left_local, right_local;
        collect_lane_boundary_local(left_local, right_local);
        if (left_local.size() < 2 || right_local.size() < 2) return;

        double seen_fwd = std::min(left_local.back().fwd, right_local.back().fwd);
        if (seen_fwd < 1.0) return;

        double left_slope = fit_tail_slope(left_local);
        double right_slope = fit_tail_slope(right_local);
        double left0 = left_local.back().lat;
        double right0 = right_local.back().lat;
        double width_ref = (lane_width > 1.0) ? lane_width : (left0 - right0);
        width_ref = std::clamp(width_ref, 1.5, 3.2);

        double cos_h = std::cos(start.pose.theta);
        double sin_h = std::sin(start.pose.theta);
        const double max_fwd = 5.0;
        const double step = 0.4;
        const double decay_sigma = 1.4;

        for (double fwd = seen_fwd + step; fwd <= max_fwd; fwd += step) {
            double left_lat = left0 + left_slope * (fwd - left_local.back().fwd);
            double right_lat = right0 + right_slope * (fwd - right_local.back().fwd);
            double width = left_lat - right_lat;

            if (width < 1.2 || width > 4.0) {
                double center_lat = 0.5 * (left_lat + right_lat);
                left_lat = center_lat + 0.5 * width_ref;
                right_lat = center_lat - 0.5 * width_ref;
                width = width_ref;
            }
            if (left_lat < 0.2 || right_lat > -0.2) continue;

            double extra = fwd - seen_fwd;
            float peak = static_cast<float>(0.7 * std::exp(-0.5 * extra * extra /
                (decay_sigma * decay_sigma)));
            if (peak < 0.05f) continue;

            double lx = start.pose.x + cos_h * fwd - sin_h * left_lat;
            double ly = start.pose.y + sin_h * fwd + cos_h * left_lat;
            double rx = start.pose.x + cos_h * fwd - sin_h * right_lat;
            double ry = start.pose.y + sin_h * fwd + cos_h * right_lat;

            add_virtual_cost_point(lx, ly, peak);
            add_virtual_cost_point(rx, ry, peak);
        }
    }

    void publish_cost_field(const rclcpp::Time& stamp) {
        if (grid.cost_field.size() == 0) return;

        nav_msgs::msg::OccupancyGrid cf_msg;
        cf_msg.header.stamp = stamp;
        cf_msg.header.frame_id = last_map_frame_id;
        cf_msg.info = last_map_info;

        int w = grid.cost_field.rows(), h = grid.cost_field.cols();
        cf_msg.data.resize(w * h);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                cf_msg.data[j * w + i] = static_cast<int8_t>(
                    std::clamp(static_cast<int>(grid.cost_field(i, j) * 100.0f), 0, 100));

        cost_field_pub->publish(cf_msg);
    }

    std::vector<State> compute_centerline_from_boundaries(double& lane_width_out) {
        std::vector<State> centerline;
        lane_width_out = 0.0;

        if (!lane_boundaries_received) return centerline;

        double cos_h = std::cos(start.pose.theta);
        double sin_h = std::sin(start.pose.theta);

        std::vector<LocalPt> left_local, right_local;
        collect_lane_boundary_local(left_local, right_local);

        if (left_local.size() < 2 || right_local.size() < 2) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "CL fail: recv=%d left_raw=%zu right_raw=%zu left_cl=%zu right_cl=%zu",
                (int)lane_boundaries_received,
                lane_left_pts.size(), lane_right_pts.size(),
                left_local.size(), right_local.size());
            return centerline;
        }

        const double max_fwd  = 5.0;
        const double fwd_step = 0.5;
        double total_width = 0.0;
        int valid_count = 0;
        int num_steps = static_cast<int>(std::round(max_fwd / fwd_step));

        for (int step = 1; step <= num_steps; ++step) {
            double fwd = step * fwd_step;
            double ll = interp_lat(left_local, fwd);   // positive (left)
            double rl = interp_lat(right_local, fwd);  // negative (right)
            if (std::isnan(ll) || std::isnan(rl)) continue;
            if (ll < 0.1 || rl > -0.1) continue;

            double offset = (ll + rl) / 2.0;  // signed: positive if car is right of center
            State s;
            s.pose.x     = start.pose.x + cos_h * fwd - sin_h * offset;
            s.pose.y     = start.pose.y + sin_h * fwd + cos_h * offset;
            s.pose.theta = start.pose.theta;
            s.vel        = full_constraints.vel[1];
            centerline.push_back(s);
            total_width += (ll - rl);  // always positive
            valid_count++;
        }

        if (valid_count < 2) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "CL interp fail: left_cl=%zu[%.1f..%.1f] right_cl=%zu[%.1f..%.1f] valid=%d/%d",
                left_local.size(),
                left_local.empty() ? 0.0 : left_local.front().fwd,
                left_local.empty() ? 0.0 : left_local.back().fwd,
                right_local.size(),
                right_local.empty() ? 0.0 : right_local.front().fwd,
                right_local.empty() ? 0.0 : right_local.back().fwd,
                valid_count, num_steps);
            centerline.clear();
        } else {
            lane_width_out = total_width / valid_count;
        }
        return centerline;
    }

    // scan_lane kept as dead code for reference; replaced by compute_centerline_from_boundaries.
    // Scan the occupancy grid perpendicular to the car's heading at multiple forward distances.
    // Returns a sequence of centerline States (at ~0.5m intervals ahead), or empty if no lane found.
    // Also writes lane_width_out with the average detected corridor width.
    // don't use probably
    std::vector<State> scan_lane(double& lane_width_out) {
        std::vector<State> centerline;
        lane_width_out = 0.0;
        if (grid.data.size() == 0) return centerline;

        const double max_fwd  = 5.0;
        const double fwd_step = 0.5;
        const double max_lat  = 3.0;    // scan +/-3m sideways
        const double lat_step = 0.05;   // 5cm steps
        const double min_lat  = 0.2;    // ignore walls closer than 20cm (car body) might remove

        double cos_h = std::cos(start.pose.theta);
        double sin_h = std::sin(start.pose.theta);

        double total_width = 0.0;
        int valid_count = 0;

        int num_steps = static_cast<int>(std::round(max_fwd / fwd_step));
        for (int step = 1; step <= num_steps; ++step) {
            double fwd = step * fwd_step;
            double px = start.pose.x + cos_h * fwd;
            double py = start.pose.y + sin_h * fwd;

            // --- Scan left wall (perpendicular CCW from heading) ---
            double left_wall  = -1.0;  // -1 = not found / hit boundary
            for (double lat = min_lat; lat <= max_lat; lat += lat_step) {
                double wx = px - sin_h * lat;
                double wy = py + cos_h * lat;
                float v = grid.at(wx, wy);
                if (v < 0.0f) { left_wall = -1.0; break; }  // grid boundary
                if (v > 0.5f) { left_wall = lat; break; }   // wall hit
            }

            // --- Scan right wall (perpendicular CW from heading) ---
            double right_wall = -1.0;
            for (double lat = min_lat; lat <= max_lat; lat += lat_step) {
                double wx = px + sin_h * lat;
                double wy = py - cos_h * lat;
                float v = grid.at(wx, wy);
                if (v < 0.0f) { right_wall = -1.0; break; }  // grid boundary
                if (v > 0.5f) { right_wall = lat; break; }   // wall hit
            }

            // Only count steps where both walls were found (positive distance), sketchy
            if (left_wall > 0.0 && right_wall > 0.0) {
                double offset = (left_wall - right_wall) / 2.0;  // positive = car left of center
                State s;
                s.pose.x     = px - sin_h * offset;
                s.pose.y     = py + cos_h * offset;
                s.pose.theta = start.pose.theta;
                s.vel        = full_constraints.vel[1];
                centerline.push_back(s);
                total_width += (left_wall + right_wall);
                valid_count++;
            }
        }

        // Require at least 5 consistent hits (out of 10 forward steps)
        if (valid_count < 5) {
            centerline.clear();
        } else {
            lane_width_out = total_width / valid_count;
        }
        return centerline;
    }

    void publish_mode(const std::string& mode, int wp_idx = -1) {
        std::string data = mode;
        if (mode == "GPS" && wp_idx >= 0)
            data = "GPS:" + std::to_string(wp_idx);
        if (data == current_mode_data) return;  // avoid flooding
        current_mode_data = data;
        current_mode = mode;
        std_msgs::msg::String msg;
        msg.data = data;
        mode_pub->publish(msg);
    }

    void waypoints_callback(const cev_msgs::msg::Trajectory::SharedPtr msg) {
        if (!msg) return;

        std::vector<State> new_waypoints;
        new_waypoints.reserve(msg->waypoints.size());

        for (const auto& wp : msg->waypoints) {
            State s;
            s.pose.x = wp.x;
            s.pose.y = wp.y;
            s.pose.theta = wp.theta;
            s.tau = 0;
            s.vel = full_constraints.vel[1];
            new_waypoints.push_back(s);
        }

        if (new_waypoints.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty trajectory on /igvc_waypoints.");
            return;
        }

        update_mission(std::move(new_waypoints));
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::TransformStamped transform;
        geometry_msgs::msg::PoseStamped base_link_pose;
        base_link_pose.header = msg->header;
        base_link_pose.pose = msg->pose.pose;
        geometry_msgs::msg::PoseStamped map_pose;

        try {
            transform = tf_buffer_.lookupTransform("map", "odom", tf2::TimePointZero);
            tf2::doTransform(base_link_pose, map_pose, transform);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_DEBUG(this->get_logger(), "TF: %s", ex.what());
            return;
        }

        double qw = map_pose.pose.orientation.w;
        double qx = map_pose.pose.orientation.x;
        double qy = map_pose.pose.orientation.y;
        double qz = map_pose.pose.orientation.z;
        double yaw = restrict_angle(atan2(2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)));

        start.pose.x = map_pose.pose.position.x;
        start.pose.y = map_pose.pose.position.y;
        start.pose.theta = yaw;
        // start.tau is updated by the sim_state_callback — do NOT reset it here.
        start.vel = msg->twist.twist.linear.x;

        if (!map_initialized || waypoint_plan.empty() || !mission_active) return;

        int waypoint_count = static_cast<int>(waypoint_plan.size());

        // Advance past reached waypoints.
        // rules, again iffy:
        //  1. Within 0.8m of the waypoint → reached, advance.  (tight: avoid exiting track)
        //  2. Within 2.0m AND waypoint is behind the car (heading_dot < 0) → overshot, advance.
        // We advance AT MOST ONE waypoint per odom callback to prevent chain-skipping.
        const double waypoint_threshold = 0.8;
        const double overshot_dist     = 2.0;
        if (current_waypoint_index < waypoint_count - 1) {
            const Pose& wp = waypoint_plan[current_waypoint_index].pose;
            double dist = start.pose.distance_to(wp);
            double dx = wp.x - start.pose.x;
            double dy = wp.y - start.pose.y;
            double heading_dot = dx * std::cos(start.pose.theta) + dy * std::sin(start.pose.theta);
            bool close   = (dist < waypoint_threshold);
            bool overshot = (dist < overshot_dist && heading_dot < 0.0);
            if (close || overshot) {
                int old_idx = current_waypoint_index;
                current_waypoint_index++;
                // GPS pair logic: pairs are indexed (0,1), (2,3), (4,5) ...
                // Passing an ODD WP (pair end) → leaving GPS segment → back to lane mode.
                // proximity check, not lane loss
                if (old_idx % 2 == 1) {
                    in_gps_pair_active = false;
                }
                std::cout << "[WAYPOINT] Reached wp " << old_idx
                          << (overshot ? " (overshot)" : " (close)")
                          << " -> now targeting " << current_waypoint_index
                          << "/" << (waypoint_count - 1)
                          << " gps_pair=" << in_gps_pair_active
                          << " Pos=(" << start.pose.x << "," << start.pose.y << ")" << std::endl;
            }
        }

        goal_state = waypoint_plan.back();
        goal_state.vel = 0.0;
        double dist_to_goal = start.pose.distance_to(goal_state.pose);

        if (current_waypoint_index >= waypoint_count - 1 && dist_to_goal < 0.8) {
            in_gps_pair_active = false;
            current_waypoint_index = 0;
            RCLCPP_INFO(this->get_logger(), "Loop: back to WP0, GPS lock cleared.");
        }
        int idx = std::min(current_waypoint_index, waypoint_count - 1);
        Trajectory local_targets;

        double lane_width = 0.0;
        std::vector<State> lane_cl = compute_centerline_from_boundaries(lane_width);

        static constexpr double GPS_PROXIMITY = 1.0;

        if (!in_gps_pair_active && idx % 2 == 0) {
            const Pose& cur_wp = waypoint_plan[idx].pose;
            double dist_to_cur = start.pose.distance_to(cur_wp);
            if (dist_to_cur < GPS_PROXIMITY) {
                in_gps_pair_active = true;
                std::cout << "[GPS PROX] Within " << dist_to_cur << "m of WP" << idx
                          << " -> switching to GPS mode." << std::endl;
            }
        }
        bool raw_lane = !lane_cl.empty() && !in_gps_pair_active;
        int gps_primary_idx = idx;
        if (in_gps_pair_active && idx % 2 == 0 && idx + 1 < waypoint_count)
            gps_primary_idx = idx + 1;

        (void)consecutive_lane_frames;
        (void)consecutive_no_lane_frames;
        bool display_lane = raw_lane;
        (void)display_lane;

        if (base_cost_field.size() != 0) {
            grid.cost_field = base_cost_field;
            if (ENABLE_EXTRAPOLATED_LANE_COST && raw_lane)
                overlay_extrapolated_lane_cost(lane_width);
            publish_cost_field(msg->header.stamp);
        }

        // Publish detected centerline for visualization every frame
        {
            nav_msgs::msg::Path cl_msg;
            cl_msg.header.stamp = msg->header.stamp;
            cl_msg.header.frame_id = "map";
            for (const auto& s : lane_cl) {
                geometry_msgs::msg::PoseStamped ps;
                ps.pose.position.x = s.pose.x;
                ps.pose.position.y = s.pose.y;
                cl_msg.poses.push_back(ps);
            }
            lane_cl_pub->publish(cl_msg);
        }

        if (raw_lane) {
            int n = static_cast<int>(lane_cl.size());

            local_targets.waypoints.push_back(lane_cl[n - 1]);
            local_targets.waypoints.push_back(waypoint_plan[idx]);
            publish_mode("LANE");
        } else {
            local_targets.waypoints.push_back(waypoint_plan[gps_primary_idx]);
            if (gps_primary_idx + 1 < waypoint_count)
                local_targets.waypoints.push_back(waypoint_plan[gps_primary_idx + 1]);
            publish_mode(in_gps_pair_active ? "GPS" : "LANE", gps_primary_idx);
        }

        local_planner->gps_mode = in_gps_pair_active;
        Trajectory path = local_planner->plan_path(grid, start, local_targets.waypoints[0],
            local_targets, last_path);

        // 30 ish callbacks
        static int status_counter = 0;
        if (++status_counter >= 30) {
            status_counter = 0;
            const Pose& wp = local_targets.waypoints[0].pose;
            double dist = start.pose.distance_to(wp);
            std::cout << "[STATUS/" << current_mode << "] Pos=("
                      << start.pose.x << "," << start.pose.y
                      << "," << (start.pose.theta * 180.0 / M_PI) << "deg)"
                      << " tau=" << start.tau << " vel=" << start.vel
                      << " lane_w=" << lane_width << "m"
                      << " | WP[" << current_waypoint_index << "]=(" << wp.x << "," << wp.y << ")"
                      << " dist=" << dist << "m | cost=" << path.cost << std::endl;
        }

        last_path = path;
        cev_msgs::msg::Trajectory plan_msg;
        plan_msg.header.stamp = msg->header.stamp;
        plan_msg.header.frame_id = "map";
        plan_msg.timestep = path.timestep;

        for (const State& s : path.waypoints) {
            cev_msgs::msg::Waypoint w;
            w.x = s.pose.x;
            w.y = s.pose.y;
            w.v = s.vel;
            w.theta = s.pose.theta;
            w.tau = s.tau;
            plan_msg.waypoints.push_back(w);
        }
        path_pub->publish(plan_msg);

        // Publish as nav_msgs::Path for visualisation
        nav_msgs::msg::Path nav_path;
        nav_path.header = plan_msg.header;
        for (const State& s : path.waypoints) {
            geometry_msgs::msg::PoseStamped ps;
            ps.pose.position.x = s.pose.x;
            ps.pose.position.y = s.pose.y;
            ps.pose.orientation = tf2::toMsg(
                tf2::Quaternion(tf2::Vector3(0, 0, 1), s.pose.theta));
            nav_path.poses.push_back(ps);
        }
        local_path_pub->publish(nav_path);

        auto end = std::chrono::high_resolution_clock::now();
        avg_planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start_time).count();
        start_time = std::chrono::high_resolution_clock::now();
    }

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        grid = Grid();
        grid.origin = Pose{msg->info.origin.position.x, msg->info.origin.position.y, 0};
        grid.resolution = msg->info.resolution;
        grid.data = Eigen::MatrixXf(msg->info.width, msg->info.height);
        last_map_info = msg->info;
        last_map_frame_id = msg->header.frame_id;

        for (unsigned int i = 0; i < msg->info.width; i++) {
            for (unsigned int j = 0; j < msg->info.height; j++) {
                int8_t val = msg->data[j * msg->info.width + i];
                if (val < 0) {
                    grid.data(i, j) = -1.0f;
                } else if (val < 50) {
                    grid.data(i, j) = 0.0f;
                } else {
                    grid.data(i, j) = 1.0f;
                }
            }
        }
        map_initialized = true;
        grid.compute_cost_field();
        base_cost_field = grid.cost_field;
    }

    void rviz_target_callback(const geometry_msgs::msg::PoseStamped msg) {
        State wp;
        const auto& q = msg.pose.orientation;
        double yaw = std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
        wp.pose = Pose(msg.pose.position.x, msg.pose.position.y, yaw);
        wp.tau = 0;
        wp.vel = 0;
        std::vector<State> mission{wp};
        update_mission(std::move(mission));
    }

    void update_mission(std::vector<State>&& new_waypoints) {
        waypoint_plan = std::move(new_waypoints);
        current_waypoint_index = 0;
        last_path = Trajectory();

        if (waypoint_plan.empty()) {
            mission_active = false;
            return;
        }

        waypoint_plan.back().vel = 0.0;
        goal_state = waypoint_plan.back();
        mission_active = true;

        RCLCPP_INFO(this->get_logger(), "Loaded mission with %zu waypoint(s).",
            waypoint_plan.size());
    }

    void publish_stop_trajectory(const rclcpp::Time& stamp) {
        cev_msgs::msg::Trajectory stop;
        stop.header.stamp = stamp;
        stop.header.frame_id = "map";
        stop.timestep = 0.0;

        cev_msgs::msg::Waypoint w;
        w.x = goal_state.pose.x;
        w.y = goal_state.pose.y;
        w.theta = goal_state.pose.theta;
        w.v = 0.0;
        w.tau = 0.0;
        stop.waypoints.push_back(w);

        path_pub->publish(stop);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PlannerNode>());
    rclcpp::shutdown();
    return 0;
}
