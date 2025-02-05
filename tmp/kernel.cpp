#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <chrono>
#include <opencv2/core.hpp>  // Include OpenCV header
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <bits/stdc++.h>

using namespace Eigen;
using namespace std;

VectorXf createGaussianKernel(int search_radius, float sigma) {
    int kernel_size = 2 * search_radius + 1;
    VectorXf kernel(kernel_size);
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; ++i) {
        int dist = i - search_radius;
        kernel(i) = exp(-(dist * dist) / (2 * sigma * sigma));
        sum += kernel(i);
    }

    // Normalize the kernel to ensure the sum is 1
    kernel /= sum;
    return kernel;
}

void visualizeCostmap(const std::vector<std::vector<double>>& costmap, const MatrixXf& grid,
    const std::string& filename = "costmap.png") {
    int rows = costmap.size();
    int cols = costmap[0].size();

    // Flatten the 2D vector into a 1D array for cv::Mat
    std::vector<uchar> flatCostmap;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Normalize cost values to range [0, 255] and add them to flatCostmap
            flatCostmap.push_back(static_cast<uchar>(costmap[i][j]
                                                     * 255.0));  // Assuming 0 <= cost <= 1
            // cout << costmap[i][j] << endl;
        }
    }

    // Create a cv::Mat from the flattened 1D array
    cv::Mat image(rows, cols, CV_8UC1, flatCostmap.data());  // 8-bit single-channel image

    // Apply a colormap (e.g., 'JET', 'HOT', 'PLASMA')
    cv::Mat coloredImage;
    cv::applyColorMap(image, coloredImage, cv::COLORMAP_HOT);

    // Overlay the original grid on the colored image
    for (int i = 0; i < grid.rows(); ++i) {
        for (int j = 0; j < grid.cols(); ++j) {
            if (grid(i, j) > 0) {
                // cv::circle(coloredImage, cv::Point(j, i), 1, cv::Scalar(200, 200, 200), -1);
                cv::rectangle(coloredImage, cv::Point(j, i), cv::Point(j + 1, i + 1),
                    cv::Scalar(200, 200, 200), -1);
            }
        }
    }

    // Save the colored image to a file
    cv::imwrite(filename, coloredImage);

    std::cout << "Costmap image saved as " << filename << std::endl;
}

// MatrixXf random_dfs(int rows, int cols) {
//     MatrixXf grid = MatrixXf::Zero(rows, cols);
//     int num_obstacles = 5000;

//     std::unordered_set<std::tuple<>> y;
// }

MatrixXf generate_random_obstacles(int rows, int cols) {
    int num_obstacles = 10;
    int obstacle_radius = 5;
    MatrixXf grid = MatrixXf::Zero(rows, cols);

    srand(1000);

    for (int i = 0; i < num_obstacles; ++i) {
        int x = rand() % rows;
        int y = rand() % cols;

        for (int j = -obstacle_radius; j <= obstacle_radius; ++j) {
            for (int k = -obstacle_radius; k <= obstacle_radius; ++k) {
                int new_x = x + j;
                int new_y = y + k;

                float dist = sqrt(j * j + k * k);

                if (new_x >= 0 && new_x < rows && new_y >= 0 && new_y < cols
                    && dist <= obstacle_radius) {
                    grid(new_x, new_y) = 1;
                }
            }
        }
    }

    return grid;
}

int main() {
    // Example grid initialization (replace with your actual data)
    // MatrixXf grid(5, 5);  // Example 5x5 grid
    // grid << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    // 24,
    //     25;

    // 5x5 grid with random 1s and 0s
    MatrixXf grid = generate_random_obstacles(100, 100);
    grid = (grid.array() > .5).cast<float>();

    // Print grid
    // cout << "Grid:\n" << grid << endl;

    int search_radius = 10;
    int kernel_size = 2 * search_radius + 1;
    float sigma = 5.0;

    // Kernel initialization
    VectorXf kernel = createGaussianKernel(search_radius, sigma);

    auto start_time = chrono::high_resolution_clock::now();

    // Convolution along rows
    MatrixXf row_conv = MatrixXf::Zero(grid.rows(), grid.cols());
    for (int i = 0; i < grid.rows(); ++i) {
        for (int j = 0; j < grid.cols(); ++j) {
            float sum = 0.0f;
            for (int k = -search_radius; k <= search_radius; ++k) {
                int idx = j + k;
                if (idx >= 0 && idx < grid.cols()) {
                    sum += grid(i, idx) * kernel(k + search_radius);
                }
            }
            row_conv(i, j) = sum;
        }
    }

    // Convolution along columns
    MatrixXf cost_map = MatrixXf::Zero(grid.rows(), grid.cols());
    for (int j = 0; j < row_conv.cols(); ++j) {
        for (int i = 0; i < row_conv.rows(); ++i) {
            float sum = 0.0f;
            for (int k = -search_radius; k <= search_radius; ++k) {
                int idx = i + k;
                if (idx >= 0 && idx < row_conv.rows()) {
                    sum += row_conv(idx, j) * kernel(k + search_radius);
                }
            }
            cost_map(i, j) = sum;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    // Output the cost map
    // cout << "Cost Map:\n" << cost_map << endl;
    cout << "Elapsed Time: " << elapsed.count() << " seconds" << endl;

    // Visualize the cost map
    // vis_cost_map(cost_map);
    // Convert cost map to 2D vector for visualization

    cout << cost_map.rows() << endl;

    std::vector<std::vector<double>> cost_map_vec(cost_map.rows(),
        std::vector<double>(cost_map.cols()));

    for (int i = 0; i < cost_map.rows(); ++i) {
        for (int j = 0; j < cost_map.cols(); ++j) {
            cost_map_vec[i][j] = cost_map(i, j);
        }
    }

    visualizeCostmap(cost_map_vec, grid, "costmap.png");

    return 0;
}
