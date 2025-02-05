#include <nlopt.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <tuple>
#include "matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

struct Dimensions {
    double wheelbase;
    double width;
    double length;
};

// Define the bounds of the state space (min, max)
struct Bounds {
    double x[2];
    double y[2];
    double tau[2];
    double vel[2];
    double accel[2];
    double dtau[2];
};

struct State {
    double x;
    double y;
    double theta;  // Heading
    double tau;    // Steering angle
    double vel;    // Velocity
};

// Update step
struct Input {
    double tau;
    double vel;
};

double bound(double x, double bound[2]) {
    return std::clamp(x, bound[0], bound[1]);
}

/**
 * @brief Return a `State` updated with `Input u`
 *
 * @return `State x` updated with `Input u` and bounded by `Bounds bounds` over timestep `dt`
 */
State update(State& x, Input& u, Bounds& bounds, double dt, Dimensions& dims) {
    State _x;

    double accel = bound(u.vel - x.vel, bounds.accel);
    double dtau = bound(u.tau - x.tau, bounds.dtau);

    _x.vel = bound(x.vel + accel * dt, bounds.vel);
    _x.tau = bound(x.tau + dtau * dt, bounds.tau);

    double avg_vel = (x.vel + _x.vel) / 2;

    double R = dims.wheelbase / tan(x.tau);
    double dtheta = (avg_vel / R) * dt;
    _x.theta = x.theta + dtheta;

    double avg_theta = (x.theta + _x.theta) / 2;
    double avg_tau = (x.tau + _x.tau) / 2;

    _x.x = bound(x.x + avg_vel * cos(avg_theta + avg_tau) * dt, bounds.x);
    _x.y = bound(x.y + avg_vel * sin(avg_theta + avg_tau) * dt, bounds.y);

    return _x;
}

std::string to_string(State& x) {
    return (
        "State(\n"
        "      x: " + std::to_string(x.x) + "\n"
        "      y: " + std::to_string(x.y) + "\n"
        "  theta: " + std::to_string(x.theta) + "\n"
        "    tau: " + std::to_string(x.tau) + "\n"
        "    vel: " + std::to_string(x.vel) + "\n"
        ")"
    );
}

// INITIALIZE AS IF CLASS
std::vector<State> waypoints = {
    {-1.0, -1.0, 0, 0, 0}, {1.0, 2.0, 0, 0, 0}, {10.0, 10.0, 0, 0, 0}};  // Waypoints
// std::vector<std::vector<double>> obstacles = {{1, 1}, {1, 2}, {2, 2}};  // Obstacle coordinates

// 10 obstacles randomly placed
std::vector<std::vector<double>> obstacles = {{1, 1}, {1, 2}, {2, 1}, {3, 4}, {4, 6.5}, {4, 7.5},
    {4, 8.5}, {4, 9.5}, {4, 5}, {5, 4}, {6, 4}, {7, 8}, {8, 7}, {9, 10}};

// std::vector<std::vector<double>> obstacles = {};

std::vector<State> path = {waypoints[0]};

Bounds bounds = {
    {-10, 10},              // x
    {-10, 10},              // y
    {-M_PI / 4, M_PI / 4},  // tau
    {0, 10},                // vel
    {-2, 2},                // accel
    {-M_PI / 8, M_PI / 8}   // dtau
};
Dimensions dims = {
    .2,                     // wheelbase
    1,                      // width
    1                       // length
};
double dt = .1;

double distance(State a, State b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double distance(State a, std::vector<double> b) {
    double dx = a.x - b[0];
    double dy = a.y - b[1];
    return sqrt(dx * dx + dy * dy);
}

double distance(const std::vector<double>& a, const std::vector<double>& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    return sqrt(dx * dx + dy * dy);
}

double path_obs_cost(std::vector<State>& path, std::vector<std::vector<double>>& obstacles) {
    double cost = 0;

    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < obstacles.size(); j++) {
            cost += 3 / distance(path[i], obstacles[j]);
            // std::cout << 1 / distance({path[i].x, path[i].y}, obstacles[j]) << std::endl;
        }
    }
    return cost;
}

double path_waypoints_cost(std::vector<State>& path, std::vector<State>& waypoints) {
    double cost = 0;
    for (int i = 1; i < path.size(); i++) {
        // Waypoints
        for (int j = 1; j < waypoints.size() - 1; j++) {
            cost += .3 * distance(path[i], waypoints[j]);
        }
        // Goal
        cost += 2 * distance(path[i], waypoints[waypoints.size() - 1]);
    }
    return cost;
}

std::vector<State> decompose(State start_state, std::vector<double> u, Bounds& bounds, double dt,
    Dimensions& dims) {
    std::vector<State> path;
    State state = start_state;
    path.push_back(state);
    for (int i = 0; i < u.size(); i += 2) {
        Input input = {u[i], u[i + 1]};
        state = update(state, input, bounds, dt, dims);
        path.push_back(state);
    }
    return path;
}

// Objective function
double objective_function(const std::vector<double>& x, std::vector<double>& grad, void* data) {
    std::vector<State> path_ = decompose(path[path.size() - 1], x, bounds, dt, dims);
    double cost = path_obs_cost(path_, obstacles) + path_waypoints_cost(path_, waypoints);

    // std::cout << "Cost: " << cost << std::endl;
    // Path obs
    // std::cout << "Path obs cost: " << path_obs_cost(path, obstacles) << std::endl;
    // Path waypoints
    // std::cout << "Path waypoints cost: " << path_waypoints_cost(path, waypoints) << std::endl;

    return cost;
}

void test_state() {
    // TEST STATE
    State state = {
        0,                      // x
        0,                      // y
        M_PI / 4,               // theta
        0,                      // tau
        0                       // vel
    };
    Input input = {
        0,                      // dtau
        -2                      // vel
    };
    Bounds bounds = {
        {-10, 10},              // x
        {-10, 10},              // y
        {-M_PI / 4, M_PI / 4},  // tau
        {-10, 10},              // vel
        {-2.0, 2.0},            // accel
        {-M_PI / 8, M_PI / 8}   // dtau
    };
    Dimensions dims = {
        .2,                     // wheelbase
        1,                      // width
        1                       // length
    };

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 30; i++) {
        state = update(state, input, bounds, .1, dims);
    }

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed = end_time - start_time;

    std::cout << to_string(state) << std::endl;
    std::cout << "State update ran for " << elapsed.count() << " seconds." << std::endl;

    std::cout << "Goal: " << waypoints[1].x << ", " << waypoints[1].y << std::endl;
    std::cout << "\n Distance to Goal: " << distance({state.x, state.y}, waypoints[1]) << std::endl;
}

void plot_path(const std::vector<State>& path, const std::vector<State>& waypoints,
    const std::vector<std::vector<double>>& obstacles, double obstacle_radius = 0.3) {
    std::vector<double> path_x, path_y;
    for (const auto& state: path) {
        path_x.push_back(state.x);
        path_y.push_back(state.y);
    }

    std::vector<double> wp_x, wp_y;
    for (const auto& wp: waypoints) {
        wp_x.push_back(wp.x);
        wp_y.push_back(wp.y);
    }

    // Plot path
    plt::figure_size(800, 600);
    plt::plot(path_x, path_y,
        {{"label", "Path"}, {"color", "blue"}, {"linestyle", "-"}, {"marker", "o"}});

    // Plot waypoints
    plt::scatter(wp_x, wp_y, 100, {{"label", "Waypoints"}, {"color", "green"}});

    // Draw obstacles as circles
    for (const auto& obs: obstacles) {
        std::vector<double> circle_x, circle_y;
        for (double angle = 0; angle <= 2 * M_PI; angle += 0.1) {
            circle_x.push_back(obs[0] + obstacle_radius * cos(angle));
            circle_y.push_back(obs[1] + obstacle_radius * sin(angle));
        }
        plt::plot(circle_x, circle_y, {{"color", "red"}, {"label", "Obstacle"}});
    }

    // Add legend (avoiding duplicate obstacle labels)
    plt::legend();
    plt::title("Path Planning Visualization");
    plt::xlabel("X Position");
    plt::ylabel("Y Position");
    plt::grid(true);
    plt::axis("equal");  // Maintain aspect ratio

    plt::show();
}

void optimize_iter(nlopt::opt& opt, std::vector<double>& initial_input) {
    double minf;
    nlopt::result result = opt.optimize(initial_input, minf);
}

void plan() {
    int num_states = 10;

    // nlopt::opt opt(nlopt::LN_COBYLA, num_states * 2);
    // nlopt::opt opt(nlopt::LN_BOBYQA, num_states * 2);
    nlopt::opt opt(nlopt::LN_NELDERMEAD, num_states * 2);
    // nlopt::opt opt(nlopt::LN_SBPLX, num_states * 2);
    // nlopt::opt opt(nlopt::LN_PRAXIS, num_states * 2);
    opt.set_min_objective(objective_function, nullptr);
    opt.set_xtol_rel(1e-4);

    // Set bounds
    std::vector<double> lb, ub;
    // for (int i = 0; i < num_states; i++) {
    //     lb.push_back(-M_PI / 8);
    //     lb.push_back(-2);
    //     ub.push_back(M_PI / 8);
    //     ub.push_back(2);
    // }
    for (int i = 0; i < num_states; i++) {
        lb.push_back(-M_PI / 4);
        lb.push_back(0);
        ub.push_back(M_PI / 4);
        ub.push_back(10);
    }
    // opt.set_lower_bounds(lb);
    // opt.set_upper_bounds(ub);

    std::vector<double> x = {};
    for (int i = 0; i < num_states; i++) {
        x.push_back(0);
        x.push_back(0);
    }

    std::vector<double> time_per_iter = {};

    // std::vector<State> path = {waypoints[0]};

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mid_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();

    double max_time = 0;

    int num_iters = 0;

    while (distance(path[path.size() - 1], waypoints[waypoints.size() - 1]) > 0.5) {
        mid_time = std::chrono::high_resolution_clock::now();
        optimize_iter(opt, x);
        num_iters++;
        end_time = std::chrono::high_resolution_clock::now();

        if (std::chrono::duration<double>(end_time - mid_time).count() > max_time) {
            max_time = std::chrono::duration<double>(end_time - mid_time).count();
        }

        time_per_iter.push_back(std::chrono::duration<double>(end_time - mid_time).count());
        std::cout << "Iteration time: "
                  << std::chrono::duration<double>(end_time - mid_time).count() << std::endl;

        std::vector<State> new_path = decompose(path[path.size() - 1], x, bounds, dt, dims);
        // Keep only the first 2 states of the new path
        path.push_back(new_path[1]);
        path.push_back(new_path[2]);

        // Print cost
        // std::cout << "Cost: " << objective_function(x, x, nullptr) << std::endl;

        // Shift all inititial inputs left by two input sets
        // x.erase(x.begin(), x.begin() + 4);
        // // // Add two new input sets to the end
        // x.push_back(0);
        // x.push_back(0);
        // x.push_back(0);
        // x.push_back(0);

        // std::cout << "Inputs: " << std::endl;
        // for (int i = 0; i < x.size(); i += 2) {
        //     std::cout << "dtau: " << x[i] << ", accel: " << x[i + 1] << std::endl;
        // }

        // Set all x to 0
        x.clear();
        for (int i = 0; i < num_states; i++) {
            x.push_back(0);
            // Push back final velocity
            x.push_back(path[path.size() - 1].vel);
        }

        // plot_path(path, waypoints, obstacles);
    }

    end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Max iteration time: " << max_time << std::endl;
    std::cout << "Avg iteration time: "
              << std::chrono::duration<double>(end_time - start_time).count() / num_iters
              << std::endl;

    // std::cout << "Time Per iter: " << std::endl;
    // for (int i = 0; i < time_per_iter.size(); i++) {
    // std::cout << time_per_iter[i] << std::endl;
    // }

    plot_path(path, waypoints, obstacles);
}

void optimize() {
    int num_states = 10;

    nlopt::opt opt(nlopt::LN_COBYLA, num_states * 2);
    opt.set_min_objective(objective_function, nullptr);
    opt.set_xtol_rel(1e-4);

    // Set maximum time limit to 3 seconds
    // opt.set_maxtime(3.0);

    std::vector<double> x = {};
    for (int i = 0; i < num_states; i++) {
        x.push_back(0);
        x.push_back(0);
    }
    double minf;

    // Start time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        nlopt::result result = opt.optimize(x, minf);
        end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Optimization completed successfully." << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Optimization failed: " << e.what() << std::endl;
    }

    // End time measurement
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Optimization ran for " << elapsed.count() << " seconds." << std::endl;
    std::cout << "\nCOST: " << minf << "\n" << std::endl;

    std::cout << "Inputs List: " << std::endl;
    for (int i = 0; i < x.size(); i += 2) {
        std::cout << "dtau: " << x[i] << ", accel: " << x[i + 1] << std::endl;
    }

    std::cout << "States Table: " << std::endl;
    std::vector<State> path = decompose(waypoints[0], x, bounds, dt, dims);
    // Headers with tabs
    printf("x\ty\ttheta\ttau\tvel\n");

    // Print table using tabs inside printf instead
    for (int i = 0; i < path.size(); i++) {
        printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", path[i].x, path[i].y, path[i].theta, path[i].tau,
            path[i].vel);
    }

    plot_path(path, waypoints, obstacles);
}

int main() {
    // optimize();
    plan();
    return 0;
}
