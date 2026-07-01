#include <iostream>
#include <iomanip>
#include <format>
#include <array>
#include <utility>
#include <random>
#include <thread>
#include <stop_token>

int main()
{
    constexpr int rows = 80,
                  columns = 150;

    char grid[rows][columns];

    auto resetGrid = [&grid]() {
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column)
                grid[row][column] = ' ';
    };

    auto moveCursor = [](int row, int column) {
        std::cout << std::format("\033[{};{}H", row + 1, column + 1) << std::flush;
    };

    std::random_device rd;
    std::uniform_real_distribution<double> slopeDist(0.1, 2),
                                           biasDist(0, 5),
                                           xDist(0, columns - 1),
                                           scaleDist(0.6, 1.4);
    constexpr int pointCount = 50;
    constexpr double learningRate = 0.001;

    std::cout << "\033[?25l" << std::flush;
    std::cout << "\033[2J" << std::flush;

    std::jthread thr{[&](std::stop_token st) {
        while (!st.stop_requested())
        {
            const double realSlope = slopeDist(rd),
                         realBias = biasDist(rd);

            std::array<std::pair<double, double>, pointCount> points;
            for (auto& [xRef, yRef] : points)
            {
                xRef = xDist(rd);
                yRef = xRef * realSlope + realBias;
                yRef *= scaleDist(rd);
            }

            double maxX = 0,
                   maxY = 0;
            for (const auto [x, y] : points)
            {
                maxX = std::max(x, maxX);
                maxY = std::max(y, maxY);
            }
            maxX = std::max<double>(maxX, 1);
            maxY = std::max<double>(maxY, 1);

            double predictedSlope = 0,
                   predictedBias = 0;
            for (int iter = 0; iter < 100'000; ++iter)
            {
                double averageSignedError = 0,
                       averageInputWeightedError = 0;

                for (const auto [realX, realY] : points)
                {
                    const double normalizedX = realX / maxX;
                    const double normalizedY = realY / maxY;

                    const double predictedY = normalizedX * predictedSlope + predictedBias;
                    const double error = predictedY - normalizedY;

                    averageSignedError += error / pointCount;
                    averageInputWeightedError += error / pointCount * normalizedX;
                }

                predictedSlope -= learningRate * averageInputWeightedError;
                predictedBias -= learningRate * averageSignedError;

                if (iter % 100 == 0)
                {
                    resetGrid();

                    for (int column = 0; column < columns; ++column)
                    {
                        const double x = column;
                        const double normalizedX = x / maxX;
                        const double normalizedYPrediction = normalizedX * predictedSlope + predictedBias;
                        const double realYPrediction = normalizedYPrediction * maxY;
                        const int row = rows - 1 - static_cast<int>(realYPrediction);

                        if ((row >= 0) && (row < rows))
                            grid[row][column] = (predictedSlope > 0.25) ? '/' : '_';
                    }

                    for (const auto [x, y] : points)
                    {
                        const int row = rows - 1 - static_cast<int>(y),
                                  column = static_cast<int>(x);

                        if ((row >= 0) && (row < rows))
                            grid[row][column] = 'o';
                    }

                    moveCursor(0, 0);
                    for (int row = 0; row < rows; ++row)
                    {
                        for (int column = 0; column < columns; ++column)
                            std::cout << grid[row][column];
                        std::cout << '\n' << std::flush;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }};
    std::cin.get();
}