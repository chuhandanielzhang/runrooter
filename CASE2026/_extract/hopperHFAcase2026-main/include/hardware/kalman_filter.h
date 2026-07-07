#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <vector>
#include <cmath>

/**
 * @brief 1D卡尔曼滤波器 (用于单个关节的位置-速度估计)
 * 
 * 状态向量: [position, velocity]
 * 测量向量: [position_measured, velocity_measured]
 * 
 * 系统模型:
 *   x[k] = F * x[k-1] + w[k]     // 状态转移
 *   z[k] = H * x[k] + v[k]       // 测量方程
 * 
 * 其中:
 *   F = [1  dt]  // 状态转移矩阵
 *       [0  1 ]
 *   H = [1  0 ]  // 测量矩阵
 *       [0  1 ]
 *   Q = process noise covariance  // 过程噪声协方差
 *   R = measurement noise covariance  // 测量噪声协方差
 */
class KalmanFilter1D {
public:
    /**
     * @brief 构造函数
     * @param dt 时间步长 (s)
     * @param process_noise_pos 位置过程噪声标准差
     * @param process_noise_vel 速度过程噪声标准差
     * @param measurement_noise_pos 位置测量噪声标准差
     * @param measurement_noise_vel 速度测量噪声标准差
     */
    KalmanFilter1D(float dt, 
                   float process_noise_pos = 1e-4, 
                   float process_noise_vel = 1e-2,
                   float measurement_noise_pos = 1e-6,
                   float measurement_noise_vel = 1e-1);
    
    /**
     * @brief 预测步骤
     */
    void predict();
    
    /**
     * @brief 更新步骤
     * @param measured_pos 测量的位置
     * @param measured_vel 测量的速度
     */
    void update(float measured_pos, float measured_vel);
    
    /**
     * @brief 获取估计的位置
     */
    float getPosition() const { return x_pos; }
    
    /**
     * @brief 获取估计的速度
     */
    float getVelocity() const { return x_vel; }
    
    /**
     * @brief 重置滤波器状态
     */
    void reset();
    
    /**
     * @brief 设置初始状态
     */
    void setState(float pos, float vel);

private:
    // 时间步长
    float dt;
    
    // 状态向量 [position, velocity]
    float x_pos;  // 位置
    float x_vel;  // 速度
    
    // 误差协方差矩阵 P (2x2对称矩阵)
    float P[2][2];
    
    // 过程噪声协方差矩阵 Q
    float Q[2][2];
    
    // 测量噪声协方差矩阵 R
    float R[2][2];
    
    // 是否已初始化
    bool initialized;
};

/**
 * @brief 3D卡尔曼滤波器向量 (用于3个关节)
 */
class KalmanFilter3D {
public:
    /**
     * @brief 构造函数
     * @param dt 时间步长 (s)
     * @param process_noise_pos 位置过程噪声标准差
     * @param process_noise_vel 速度过程噪声标准差
     * @param measurement_noise_pos 位置测量噪声标准差
     * @param measurement_noise_vel 速度测量噪声标准差
     */
    KalmanFilter3D(float dt,
                   float process_noise_pos = 1e-4,
                   float process_noise_vel = 1e-2,
                   float measurement_noise_pos = 1e-6,
                   float measurement_noise_vel = 1e-1);
    
    /**
     * @brief 析构函数
     */
    ~KalmanFilter3D();
    
    /**
     * @brief 预测和更新所有关节
     * @param measured_pos 测量的位置数组 [3]
     * @param measured_vel 测量的速度数组 [3]
     * @param output_pos 输出的滤波后位置 [3]
     * @param output_vel 输出的滤波后速度 [3]
     */
    void filter(const float measured_pos[3], const float measured_vel[3],
                float output_pos[3], float output_vel[3]);
    
    /**
     * @brief 重置所有滤波器
     */
    void reset();

private:
    // 三个独立的1D卡尔曼滤波器
    KalmanFilter1D* filters[3];
};

#endif // KALMAN_FILTER_H














