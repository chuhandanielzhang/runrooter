#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <array>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>


struct VeloxImuData {
    struct {
        float data[3];  // x, y, z
    } accCalibrated;
    
    struct {
        float data[3];  // x, y, z (deg/s)
    } gyroIIBiasCalibrated;
    
    struct {
        float data[3];  // roll, pitch, yaw (deg)
    } euler;
    
    struct {
        float data[4];  // w, x, y, z
    } quaternion;
};

class VeloxF7Wrapper {
public:
    VeloxF7Wrapper(const std::string& port = "/dev/ttyACM1", int frequency_hz = 100);
    ~VeloxF7Wrapper();
    
    bool connect();
    void disconnect();
    bool hasImuData();
    void getImuData(VeloxImuData& data);
    bool sendDirectMotorCommands(uint16_t m1, uint16_t m2, uint16_t m3, uint16_t m4 = 1000, uint16_t m5 = 1000, uint16_t m6 = 1000);
    
private:
    std::string port_;
    int frequency_hz_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::atomic<bool> has_new_data_;
    
    std::thread reader_thread_;
    std::thread sender_thread_;
    std::mutex data_mutex_;
    VeloxImuData current_data_;
    

    volatile uint16_t motor_m1_, motor_m2_, motor_m3_, motor_m4_, motor_m5_, motor_m6_;
    volatile bool has_motor_cmd_;
    
    void readerThread();
    void senderThread();
    void updateImuData(const uint8_t* raw_imu, const uint8_t* attitude);
    void sendMotorCommandDirect(uint16_t m1, uint16_t m2, uint16_t m3, uint16_t m4, uint16_t m5, uint16_t m6);
    

    int serial_fd_;
    bool openSerial();
    void closeSerial();
    bool sendMSPRequest(uint8_t cmd);
    bool readMSPResponse(uint8_t& cmd, uint8_t* data, uint8_t& length);
    uint8_t calculateChecksum(uint8_t length, uint8_t cmd, const uint8_t* data);
    

    static const uint8_t MSP_RAW_IMU = 102;
    static const uint8_t MSP_ATTITUDE = 108;
    

    struct SimpleFilter {
        std::array<float, 3> prev_values;
        float alpha;
        
        SimpleFilter(float filter_alpha = 0.8f) : alpha(filter_alpha) {
            prev_values.fill(0.0f);
        }
        
        void filter(float input[3]) {
            for (int i = 0; i < 3; i++) {
                prev_values[i] = alpha * input[i] + (1.0f - alpha) * prev_values[i];
                input[i] = prev_values[i];
            }
        }
    };
    

    SimpleFilter acc_filter_;
    SimpleFilter gyro_filter_;
};




