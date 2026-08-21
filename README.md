# Linear Regression

A real-time terminal visualization of linear regression training via gradient descent.

Random noisy points are generated around a hidden line. The model starts from zero and iteratively fits a line \( y = mx + b \) while the process is rendered live in the console.

## How it works

1. A true slope and bias are sampled randomly.
2. 50 points are generated with noise.
3. The model begins at \( m = 0 \), \( b = 0 \).
4. Gradient descent runs for 100 000 iterations (with visualization every 100 steps).
5. After training finishes, a new random dataset is generated and the process repeats.

### Update rules

```cpp
predictedSlope -= learningRate * averageInputWeightedError;
predictedBias  -= learningRate * averageSignedError;
```

Inputs and targets are normalized before each training run.

## Parameters

```cpp
constexpr int rows = 80;
constexpr int columns = 150;

constexpr int pointCount = 50;
constexpr double learningRate = 0.001;
```

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Requires a C++23-capable compiler.

## Running

```bash
./LinearRegression
```

The simulation starts immediately. Press any key to exit.

## Notes

- Points are drawn as `o`.
- The current prediction is drawn as `/` (positive slope) or `_`.
- Uses ANSI escape sequences for cursor control and clearing.
- New random problems are generated continuously after each training run.
