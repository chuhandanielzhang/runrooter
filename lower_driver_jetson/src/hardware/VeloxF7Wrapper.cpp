#include "hardware/VeloxF7Wrapper.h"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cmath>
#include <cstring>
#include <errno.h>
#include <chrono>
#include <thread>

VeloxF7Wrapper::VeloxF7Wrapper(const std::string& port, int frequency_hz) 
    : port_(port), frequency_hz_(frequency_hz), connected_(false), running_(false), has_new_data_(false), serial_fd_(-1),
      acc_filter_(0.9f), gyro_filter_(0.8f),
      motor_m1_(1000), motor_m2_(1000), motor_m3_(1000), motor_m4_(1000), motor_m5_(1000), motor_m6_(1000), has_motor_cmd_(false) {
    

    memset(&current_data_, 0, sizeof(current_data_));
    
    std::cout << "VeloxF7Wrapper: Init Velox F7 SE @ " << port_ << " (True Duplex)" << std::endl;
}

VeloxF7Wrapper::~VeloxF7Wrapper() {
    disconnect();
}

bool VeloxF7Wrapper::connect() {
    if (connected_) {
        return true;
    }
    
    if (!openSerial()) {
        std::cerr << "VeloxF7Wrapper: Cannot open port " << port_ << std::endl;
        return false;
    }
    
    connected_ = true;
    running_ = true;
    

    std::cout << "VeloxF7Wrapper: Fast init..." << std::endl;
    

    tcflush(serial_fd_, TCIOFLUSH);
    
    sleep(1);
    

        sendMSPRequest(1);  // MSP_API_VERSION
    usleep(50000);  // 50ms
    

    reader_thread_ = std::thread(&VeloxF7Wrapper::readerThread, this);
    sender_thread_ = std::thread(&VeloxF7Wrapper::senderThread, this);
    
    std::cout << "VeloxF7Wrapper: OK Connected (True Dual-Thread)" << std::endl;
    return true;
}

void VeloxF7Wrapper::disconnect() {
    if (!connected_) {
        return;
    }
    
    running_ = false;
    

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
    
    closeSerial();
    connected_ = false;
    
    std::cout << "VeloxF7Wrapper: Disconnected (Dual-Thread Stopped)" << std::endl;
}

bool VeloxF7Wrapper::hasImuData() {
    return has_new_data_.load();
}

void VeloxF7Wrapper::getImuData(VeloxImuData& data) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    data = current_data_;
    has_new_data_ = false;
}

bool VeloxF7Wrapper::openSerial() {
    serial_fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (serial_fd_ == -1) {
        std::cerr << "VeloxF7Wrapper: Cannot open serial port " << port_ << ": " << strerror(errno) << std::endl;
        return false;
    }
    
    struct termios options;
    tcgetattr(serial_fd_, &options);
    

    cfsetispeed(&options, B921600);
    cfsetospeed(&options, B921600);
    

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    

    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY | ICRNL | INLCR);
    

    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 0;
    
    tcsetattr(serial_fd_, TCSANOW, &options);
    tcflush(serial_fd_, TCIOFLUSH);
    
    std::cout << "VeloxF7Wrapper: Serial config OK (Non-blocking) " << port_ << std::endl;
    return true;
}

void VeloxF7Wrapper::closeSerial() {
    if (serial_fd_ != -1) {
        close(serial_fd_);
        serial_fd_ = -1;
    }
}

uint8_t VeloxF7Wrapper::calculateChecksum(uint8_t length, uint8_t cmd, const uint8_t* data) {
    uint8_t checksum = length ^ cmd;
    for (int i = 0; i < length; i++) {
        checksum ^= data[i];
    }
    return checksum;
}

bool VeloxF7Wrapper::sendMSPRequest(uint8_t cmd) {
    if (serial_fd_ == -1) return false;
    
    uint8_t packet[6];
    packet[0] = '$';
    packet[1] = 'M';
    packet[2] = '<';
    packet[3] = 0;  // length
    packet[4] = cmd;
    packet[5] = calculateChecksum(0, cmd, nullptr);
    
    return write(serial_fd_, packet, 6) == 6;
}

bool VeloxF7Wrapper::readMSPResponse(uint8_t& cmd, uint8_t* data, uint8_t& length) {
    if (serial_fd_ == -1) return false;
    

    usleep(2000);
    
    uint8_t header[3];
    ssize_t bytes_read = read(serial_fd_, header, 3);
    if (bytes_read < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return false;
        }
        return false;
    }
    if (bytes_read != 3) {
        if (bytes_read > 0) {

            for (int retry = 0; retry < 3 && bytes_read < 3; retry++) {
                usleep(1000);  // 1ms
                ssize_t remaining = read(serial_fd_, header + bytes_read, 3 - bytes_read);
                if (remaining > 0) bytes_read += remaining;
            }
            if (bytes_read != 3) {
                return false;
            }
        } else {
            return false;
        }
    }
    
    if (header[0] != '$' || header[1] != 'M' || header[2] != '>') {


        //           << std::hex << (int)header[0] << " " << (int)header[1] << " " << (int)header[2] << std::dec << std::endl;
        return false;
    }
    

    for (int retry = 0; retry < 3; retry++) {
        if (read(serial_fd_, &length, 1) == 1) break;
        usleep(1000);
        if (retry == 2) return false;
    }
    

    for (int retry = 0; retry < 3; retry++) {
        if (read(serial_fd_, &cmd, 1) == 1) break;
        usleep(1000);
        if (retry == 2) return false;
    }
    
    if (length > 0 && length < 32) {
        ssize_t data_read = 0;
        ssize_t total_read = 0;
        

        int retry_count = 0;
        while (total_read < length && retry_count < 10) {
            data_read = read(serial_fd_, data + total_read, length - total_read);
            if (data_read > 0) {
                total_read += data_read;
            } else if (data_read == 0 || (data_read < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) {

                usleep(2000);  // 2ms
                retry_count++;
            } else {

                return false;
            }
        }
        
        if (total_read < length) {

            return false;
        }
    } else if (length >= 32) {
        std::cerr << "VeloxF7Wrapper: Invalid data length: " << (int)length << std::endl;
        return false;
    }
    
    uint8_t checksum;

    bool checksum_read = false;
    for (int retry = 0; retry < 5; retry++) {
        if (read(serial_fd_, &checksum, 1) == 1) {
            checksum_read = true;
            break;
        }
        usleep(1000);  // 1ms
    }
    
    if (!checksum_read) {

        return false;
    }
    
    uint8_t expected = calculateChecksum(length, cmd, data);
    if (checksum != expected) {



        return false;
    }
    
    return true;
}

void VeloxF7Wrapper::updateImuData(const uint8_t* raw_imu, const uint8_t* attitude) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (raw_imu) {

        int16_t* values = (int16_t*)raw_imu;
        



        // ========== 加速度计数据解析 ==========
        // Betaflight使用NED坐标系（North-East-Down）：
        //   - X轴：向前（North）
        //   - Y轴：向右（East）
        //   - Z轴：向下（Down）
        //
        // 代码期望ENU坐标系（East-North-Up，body frame）：
        //   - X轴：向前（forward）
        //   - Y轴：向左（left）
        //   - Z轴：向上（up）
        //
        // IMPORTANT (ModeE convention used by hopper_controller/modee/core.py):
        // - `acc` is treated as a **gravity / down vector** in body frame (not specific force).
        // - When the robot is level and stationary, we expect: acc_b ≈ [0, 0, -9.81] (because +Z is UP).
        //
        // Betaflight IMU body frame is typically **FRD**:
        //   +X forward, +Y right, +Z down.
        // ModeE uses **FLU** body frame:
        //   +X forward, +Y left, +Z up.
        //
        // FRD -> FLU is a 180deg rotation about +X:
        //   [x, y, z]_FLU = [ x, -y, -z ]_FRD
        //
        // Apply this consistently to accel + gyro (and attitude below).
        const float acc_scale = 9.81f / 2048.0f;  // Betaflight加速度计标度：2048 LSB = 1g = 9.81 m/s²
        float acc_raw[3] = {
            values[0] * acc_scale,    // X: forward (keep)
            -values[1] * acc_scale,   // Y: right -> left (flip)
            -values[2] * acc_scale    // Z: down -> up (flip)
        };
        // If your IMU is mounted with a different axis orientation, apply a fixed rotation here
        // (and do the same for gyro + quaternion).
        
        acc_filter_.filter(acc_raw);
        
        current_data_.accCalibrated.data[0] = acc_raw[0];
        current_data_.accCalibrated.data[1] = acc_raw[1];
        current_data_.accCalibrated.data[2] = acc_raw[2];
        


        const float gyro_scale = 1.0f / 32.0f;  // 0.03125 deg/s per LSB
        float gyro_raw[3] = {
            values[3] * gyro_scale,   // wx: about +X forward (keep)
            -values[4] * gyro_scale,  // wy: about +Y right -> +Y left (flip)
            -values[5] * gyro_scale   // wz: about +Z down -> +Z up (flip)
        };
        

        gyro_filter_.filter(gyro_raw);
        
        current_data_.gyroIIBiasCalibrated.data[0] = gyro_raw[0];
        current_data_.gyroIIBiasCalibrated.data[1] = gyro_raw[1];
        current_data_.gyroIIBiasCalibrated.data[2] = gyro_raw[2];
    }
    
    if (attitude) {
        // ========== 解析MSP_ATTITUDE数据 ==========
        int16_t* att_values = (int16_t*)attitude;
        
        // 转换为度（Betaflight使用decidegrees: 1度 = 10 decidegrees）
        // Roll和Pitch范围: -1800到1800 decidegrees (-180°到180°)
        // Yaw范围: -18000到18000 decidegrees (-1800°到1800°)，但yaw是累积的，可能超过±180°
        current_data_.euler.data[0] = att_values[0] / 10.0f;  // roll (decidegrees → degrees)
        current_data_.euler.data[1] = att_values[1] / 10.0f;  // pitch (decidegrees → degrees)
        current_data_.euler.data[2] = att_values[2] / 10.0f;  // yaw (decidegrees → degrees)
        
        // ========== 使用Eigen库计算四元数（改进算法）==========
        // 转换为弧度
        float roll_rad = current_data_.euler.data[0] * M_PI / 180.0f;
        float pitch_rad = current_data_.euler.data[1] * M_PI / 180.0f;
        float yaw_rad = current_data_.euler.data[2] * M_PI / 180.0f;
        
        // 边界检查：防止NaN和Inf
        if (std::isnan(roll_rad) || std::isnan(pitch_rad) || std::isnan(yaw_rad) ||
            std::isinf(roll_rad) || std::isinf(pitch_rad) || std::isinf(yaw_rad)) {
            std::cerr << "VeloxF7Wrapper: Invalid RPY values detected! Using previous quaternion." << std::endl;
            return;  // 保持上一次的有效四元数
        }
        
        // 合理性检查：Roll和Pitch应该在±180°范围内
        // 注意：Yaw是累积角度，可能超过±180°，这是正常的
        if (std::abs(roll_rad) > 2.0f * M_PI || std::abs(pitch_rad) > 2.0f * M_PI) {
            std::cerr << "VeloxF7Wrapper: RPY out of reasonable range! Roll=" 
                      << current_data_.euler.data[0] << "°, Pitch=" 
                      << current_data_.euler.data[1] << "°, Yaw="
                      << current_data_.euler.data[2] << "°" << std::endl;
            // 继续处理，但输出警告
        }
        
        // Yaw 归一化和方向修正：
        // 1. Betaflight 的 yaw 可能是 [0, 2π] 范围，需要转换到 [-π, π]
        // 2. Betaflight 的 yaw 可能是顺时针为正（左手系），ModeE 期望逆时针为正（右手系）
        // 3. 如果 yaw 是 0-2π 且顺时针减小，需要：先转换范围，再取反
        
        // 如果 yaw 在 [0, 2π] 范围，先转换到 [-π, π]
        if (yaw_rad >= 0.0f && yaw_rad <= 2.0f * M_PI) {
            if (yaw_rad > M_PI) {
                yaw_rad = yaw_rad - 2.0f * M_PI;  // [π, 2π] → [-π, 0]
            }
            // 现在 yaw_rad 在 [-π, π] 范围
        } else {
            // 如果已经是 [-π, π] 范围，直接归一化
            while (yaw_rad > M_PI) yaw_rad -= 2.0f * M_PI;
            while (yaw_rad < -M_PI) yaw_rad += 2.0f * M_PI;
        }
        
        // Yaw: ModeE uses right-hand yaw: from +Z looking down, CCW is positive.
        // Betaflight yaw is typically clockwise-positive, so we negate yaw.
        //
        // Per user request: revert pitch sign flip (keep pitch as provided by the FC).
        yaw_rad = -yaw_rad;

        // Keep published Euler yaw consistent with the quaternion (degrees, wrapped).
        current_data_.euler.data[2] = yaw_rad * 180.0f / static_cast<float>(M_PI);
        // Keep published Euler pitch as provided by the FC (no sign flip).
        
        // 使用Eigen库的AngleAxis构建旋转，按XYZ内旋顺序（与scipy的'xyz'一致）
        // 这等价于：R = Rx(roll) * Ry(pitch) * Rz(yaw)
        Eigen::AngleAxisf rollAngle(roll_rad, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch_rad, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw_rad, Eigen::Vector3f::UnitZ());
        
        // 组合旋转（注意顺序：从右到左应用）
        Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
        
        // Eigen自动归一化四元数！但为了安全，我们显式调用normalize()
        q.normalize();
        
        // 验证归一化（仅在异常时输出警告）
        float norm = q.norm();
        if (std::abs(norm - 1.0f) > 1e-4f) {
            std::cerr << "VeloxF7Wrapper: Warning! Quaternion norm = " << norm 
                      << " (should be 1.0)" << std::endl;
        }
        
        // 存储四元数（Eigen的顺序是[w, x, y, z]，与我们的存储格式一致）
        current_data_.quaternion.data[0] = q.w();
        current_data_.quaternion.data[1] = q.x();
        current_data_.quaternion.data[2] = q.y();
        current_data_.quaternion.data[3] = q.z();
        
        // 调试输出已移除（避免控制台噪音）
    }
    
    has_new_data_ = true;
}

void VeloxF7Wrapper::readerThread() {
    std::cout << "VeloxF7Wrapper: RX Thread Started (Max Speed)" << std::endl;
    
    static uint8_t buffer[2048];
    int buffer_pos = 0;
    
    // 保存原始数据用于updateImuData处理
    static uint8_t last_raw_imu[18];
    static uint8_t last_attitude[6];

    tcflush(serial_fd_, TCIFLUSH);
    
    while (running_) {
        try {

            ssize_t bytes_read = read(serial_fd_, buffer + buffer_pos, sizeof(buffer) - buffer_pos - 1);
            if (bytes_read > 0) {
                buffer_pos += bytes_read;
                

                int processed = 0;
                while (buffer_pos - processed >= 6) {

                    uint8_t* header_pos = (uint8_t*)memchr(buffer + processed, '$', buffer_pos - processed - 2);
                    if (!header_pos) {

                        if (buffer_pos > 5) {
                            memmove(buffer, buffer + buffer_pos - 5, 5);
                            buffer_pos = 5;
                        }
                        break;
                    }
                    
                    processed = header_pos - buffer;
                    

                    if (processed + 5 >= buffer_pos) break;
                    if (buffer[processed + 1] != 'M' || buffer[processed + 2] != '>') {
                        processed++;
                        continue;
                    }
                    
                    uint8_t size = buffer[processed + 3];
                    uint8_t cmd = buffer[processed + 4];
                    int total_length = 6 + size;
                    
                    if (processed + total_length > buffer_pos) break;
                    
                    // IMPORTANT:
                    // Only treat RAW_IMU / ATTITUDE packets as "new IMU data".
                    // Previously we latched `has_raw_imu/has_attitude` forever once seen,
                    // then called updateImuData() for *every* MSP response (including motor acks),
                    // which can inflate apparent gyro/acc update rates (filters run on stale data).
                    bool got_raw_imu = false;
                    bool got_attitude = false;

                    if (cmd == MSP_ATTITUDE && size >= 6) {
                        // 保存原始数据
                        memcpy(last_attitude, buffer + processed + 5, 6);
                        got_attitude = true;
                    } else if (cmd == MSP_RAW_IMU && size >= 18) {
                        // 保存原始数据
                        memcpy(last_raw_imu, buffer + processed + 5, 18);
                        got_raw_imu = true;
                    }
                    
                    // 当有新的IMU或姿态数据时，更新current_data_（用于getImuData）
                    if (got_raw_imu || got_attitude) {
                        updateImuData(got_raw_imu ? last_raw_imu : nullptr,
                                     got_attitude ? last_attitude : nullptr);
                    }
                    
                    processed += total_length;
                }
                

                if (processed > 0) {
                    memmove(buffer, buffer + processed, buffer_pos - processed);
                    buffer_pos -= processed;
                }
                

                if (buffer_pos > 1500) {
                    buffer_pos = 0;
                    tcflush(serial_fd_, TCIFLUSH);
                }
            } else {

                usleep(100);
            }
            
        } catch (...) {
            usleep(1000);
        }
    }
    
    std::cout << "VeloxF7Wrapper: RX Thread Ended" << std::endl;
}

void VeloxF7Wrapper::senderThread() {
    std::cout << "VeloxF7Wrapper: TX Thread Started @ " << frequency_hz_ << "Hz" << std::endl;
    
    int send_cycle = 0;
    
    // 计算每次循环的延迟时间（微秒）
    const int64_t period_us = 1000000 / frequency_hz_;  // 例如 200Hz = 5000µs
    auto next_send_time = std::chrono::steady_clock::now();

    tcflush(serial_fd_, TCOFLUSH);
    
    while (running_) {
        try {
            // 使用 frequency_hz_ 控制发送频率
            next_send_time += std::chrono::microseconds(period_us);
            std::this_thread::sleep_until(next_send_time);

            // User request: HIGH-RATE motor PWM updates.
            // Send MSP_SET_MOTOR every TX cycle so the FC sees prop commands at `frequency_hz_`.
            //
            // Note: this increases MSP traffic and FC workload. At 500Hz it is typically still fine on USB.
            // If you observe FC lagging / missed IMU packets, reduce IMU poll rate or lower frequency_hz_.
            if (has_motor_cmd_) {
                // Send motor command (MSP_SET_MOTOR). This is a write-only packet.
                sendMotorCommandDirect(motor_m1_, motor_m2_, motor_m3_, motor_m4_, motor_m5_, motor_m6_);
            }

            // Always poll IMU each cycle: alternate ATTITUDE and RAW_IMU
            if (send_cycle % 2 == 0) {
                sendMSPRequest(MSP_ATTITUDE);
            } else {
                sendMSPRequest(MSP_RAW_IMU);
            }
            
            send_cycle++;
            
            


            
        } catch (...) {

            tcflush(serial_fd_, TCOFLUSH);
            usleep(500);
            // 重新同步时间，避免异常后时间偏移
            next_send_time = std::chrono::steady_clock::now();
        }
    }
    
    std::cout << "VeloxF7Wrapper: TX Thread Ended" << std::endl;
}

bool VeloxF7Wrapper::sendDirectMotorCommands(uint16_t m1, uint16_t m2, uint16_t m3, uint16_t m4, uint16_t m5, uint16_t m6) {
    if (!connected_) {
        return false;
    }
    

    motor_m1_ = m1;
    motor_m2_ = m2;
    motor_m3_ = m3;
    motor_m4_ = m4;
    motor_m5_ = m5;
    motor_m6_ = m6;
    has_motor_cmd_ = true;
    
    return true;
}

void VeloxF7Wrapper::sendMotorCommandDirect(uint16_t m1, uint16_t m2, uint16_t m3, uint16_t m4, uint16_t m5, uint16_t m6) {

    uint8_t packet[22];
    

    packet[0] = '$'; packet[1] = 'M'; packet[2] = '<';
    packet[3] = 16;
    packet[4] = 214; // MSP_SET_MOTOR
    

    // 设置6个电机PWM值，后2个填充为1000
    uint16_t motors[8] = {m1, m2, m3, m4, m5, m6, 1000, 1000};
    for (int i = 0; i < 8; i++) {
        packet[5 + i*2] = motors[i] & 0xFF;
        packet[5 + i*2 + 1] = (motors[i] >> 8) & 0xFF;
    }
    

    uint8_t checksum = 16 ^ 214;
    for (int i = 0; i < 16; i++) {
        checksum ^= packet[5 + i];
    }
    packet[21] = checksum;
    

    write(serial_fd_, packet, 22);
}

