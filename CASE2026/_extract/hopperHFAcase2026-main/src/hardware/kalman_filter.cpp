#include "../../include/hardware/kalman_filter.h"
#include <cstring>  // for memset

// ============================================================================
// KalmanFilter1D Implementation
// ============================================================================

KalmanFilter1D::KalmanFilter1D(float dt, 
                               float process_noise_pos, 
                               float process_noise_vel,
                               float measurement_noise_pos,
                               float measurement_noise_vel)
    : dt(dt), x_pos(0.0f), x_vel(0.0f), initialized(false) {
    
    // 初始化误差协方差矩阵 P (初始不确定性较大)
    P[0][0] = 1.0f;  // 位置方差
    P[0][1] = 0.0f;  // 协方差
    P[1][0] = 0.0f;  // 协方差
    P[1][1] = 1.0f;  // 速度方差
    
    // 初始化过程噪声协方差矩阵 Q
    Q[0][0] = process_noise_pos * process_noise_pos;
    Q[0][1] = 0.0f;
    Q[1][0] = 0.0f;
    Q[1][1] = process_noise_vel * process_noise_vel;
    
    // 初始化测量噪声协方差矩阵 R
    R[0][0] = measurement_noise_pos * measurement_noise_pos;
    R[0][1] = 0.0f;
    R[1][0] = 0.0f;
    R[1][1] = measurement_noise_vel * measurement_noise_vel;
}

void KalmanFilter1D::predict() {
    // 状态预测: x_pred = F * x
    // F = [1  dt]
    //     [0  1 ]
    float x_pos_pred = x_pos + dt * x_vel;
    float x_vel_pred = x_vel;
    
    // 误差协方差预测: P_pred = F * P * F^T + Q
    float P00 = P[0][0] + dt * P[1][0] + dt * (P[0][1] + dt * P[1][1]) + Q[0][0];
    float P01 = P[0][1] + dt * P[1][1] + Q[0][1];
    float P10 = P[1][0] + dt * P[1][1] + Q[1][0];
    float P11 = P[1][1] + Q[1][1];
    
    // 更新状态和协方差
    x_pos = x_pos_pred;
    x_vel = x_vel_pred;
    P[0][0] = P00;
    P[0][1] = P01;
    P[1][0] = P10;
    P[1][1] = P11;
}

void KalmanFilter1D::update(float measured_pos, float measured_vel) {
    if (!initialized) {
        // 第一次测量：直接初始化状态
        x_pos = measured_pos;
        x_vel = measured_vel;
        initialized = true;
        return;
    }
    
    // 预测步骤
    predict();
    
    // 测量残差 (innovation)
    // y = z - H * x
    float y_pos = measured_pos - x_pos;
    float y_vel = measured_vel - x_vel;
    
    // 残差协方差
    // S = H * P * H^T + R
    float S00 = P[0][0] + R[0][0];
    float S01 = P[0][1] + R[0][1];
    float S10 = P[1][0] + R[1][0];
    float S11 = P[1][1] + R[1][1];
    
    // 计算S的逆矩阵 (2x2矩阵求逆)
    float det = S00 * S11 - S01 * S10;
    if (fabs(det) < 1e-10f) {
        // 矩阵奇异，跳过更新
        return;
    }
    float S_inv00 = S11 / det;
    float S_inv01 = -S01 / det;
    float S_inv10 = -S10 / det;
    float S_inv11 = S00 / det;
    
    // 卡尔曼增益
    // K = P * H^T * S^(-1)
    float K00 = P[0][0] * S_inv00 + P[0][1] * S_inv10;
    float K01 = P[0][0] * S_inv01 + P[0][1] * S_inv11;
    float K10 = P[1][0] * S_inv00 + P[1][1] * S_inv10;
    float K11 = P[1][0] * S_inv01 + P[1][1] * S_inv11;
    
    // 状态更新
    // x = x + K * y
    x_pos = x_pos + K00 * y_pos + K01 * y_vel;
    x_vel = x_vel + K10 * y_pos + K11 * y_vel;
    
    // 误差协方差更新
    // P = (I - K * H) * P
    float I_KH_00 = 1.0f - K00;
    float I_KH_01 = -K01;
    float I_KH_10 = -K10;
    float I_KH_11 = 1.0f - K11;
    
    float P00_new = I_KH_00 * P[0][0] + I_KH_01 * P[1][0];
    float P01_new = I_KH_00 * P[0][1] + I_KH_01 * P[1][1];
    float P10_new = I_KH_10 * P[0][0] + I_KH_11 * P[1][0];
    float P11_new = I_KH_10 * P[0][1] + I_KH_11 * P[1][1];
    
    P[0][0] = P00_new;
    P[0][1] = P01_new;
    P[1][0] = P10_new;
    P[1][1] = P11_new;
}

void KalmanFilter1D::reset() {
    x_pos = 0.0f;
    x_vel = 0.0f;
    P[0][0] = 1.0f;
    P[0][1] = 0.0f;
    P[1][0] = 0.0f;
    P[1][1] = 1.0f;
    initialized = false;
}

void KalmanFilter1D::setState(float pos, float vel) {
    x_pos = pos;
    x_vel = vel;
    initialized = true;
}

// ============================================================================
// KalmanFilter3D Implementation
// ============================================================================

KalmanFilter3D::KalmanFilter3D(float dt,
                               float process_noise_pos,
                               float process_noise_vel,
                               float measurement_noise_pos,
                               float measurement_noise_vel) {
    // 为3个关节创建独立的卡尔曼滤波器
    for (int i = 0; i < 3; i++) {
        filters[i] = new KalmanFilter1D(dt, 
                                        process_noise_pos, 
                                        process_noise_vel,
                                        measurement_noise_pos,
                                        measurement_noise_vel);
    }
}

KalmanFilter3D::~KalmanFilter3D() {
    for (int i = 0; i < 3; i++) {
        delete filters[i];
    }
}

void KalmanFilter3D::filter(const float measured_pos[3], const float measured_vel[3],
                            float output_pos[3], float output_vel[3]) {
    for (int i = 0; i < 3; i++) {
        filters[i]->update(measured_pos[i], measured_vel[i]);
        output_pos[i] = filters[i]->getPosition();
        output_vel[i] = filters[i]->getVelocity();
    }
}

void KalmanFilter3D::reset() {
    for (int i = 0; i < 3; i++) {
        filters[i]->reset();
    }
}














