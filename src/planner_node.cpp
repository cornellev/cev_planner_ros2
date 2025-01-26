#include <global_planning/rrt.h>
#include <rclcpp/rclcpp.hpp>

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node") {
        auto dimensions = Dimensions();
        auto constraints = Constraints();
        planner = std::make_shared<global_planner::RRT>(dimensions, constraints);
    }

    void plan_path() {
        auto grid = Grid();
        auto start = Pose();
        auto target = Pose();
        auto trajectory = planner->plan_path(grid, start, target);
    }

private:
    std::shared_ptr<global_planner::RRT> planner;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::shutdown();
    return 0;
}