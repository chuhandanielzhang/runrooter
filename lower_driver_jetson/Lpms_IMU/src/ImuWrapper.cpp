#include "ImuWrapper.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <stdarg.h>

using namespace std;

ImuWrapper::ImuWrapper() 
    : streamRunning_(false)
    , shouldStop_(false)
    , dataCallback_(nullptr) {
    
    // Create sensor instance
    sensor_ = std::unique_ptr<IG1I>(IG1Factory());
    if (sensor_) {
        sensor_->setVerbose(VERBOSE_INFO);
        sensor_->setAutoReconnectStatus(true);
    }
}

ImuWrapper::~ImuWrapper() {
    stopDataStream();
    disconnect();
    
    if (sensor_) {
        sensor_->release();
    }
}

bool ImuWrapper::connect(const std::string& port, int baudrate, int timeout_s) {
    if (!sensor_) {
        log("Error: Sensor not initialized");
        return false;
    }
    
    log("Connecting to " + port + " @ " + std::to_string(baudrate));
    
    if (!sensor_->connect(port, baudrate)) {
        log("Error connecting to sensor");
        return false;
    }
    
    // Wait for connection. A baudrate mismatch can leave the library stuck in
    // CONNECTING/DATA_TIMEOUT forever (autoReconnect keeps retrying), so bound
    // the wait: we probe several baudrates at startup and must fail fast.
    for (int i = 0; i < timeout_s; ++i) {
        log("Waiting for sensor to connect...");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        int st = sensor_->getStatus();
        if (st == STATUS_CONNECTED) {
            log("Sensor connected successfully");
            return true;
        }
        if (st == STATUS_CONNECTION_ERROR) break;
    }
    
    log("Sensor connect failed @ " + std::to_string(baudrate) +
        " (status: " + std::to_string(sensor_->getStatus()) + ")");
    sensor_->disconnect();
    return false;
}

void ImuWrapper::disconnect() {
    if (sensor_) {
        sensor_->disconnect();
        log("Sensor disconnected");
    }
}

bool ImuWrapper::isConnected() const {
    return sensor_ && sensor_->getStatus() == STATUS_CONNECTED;
}

bool ImuWrapper::gotoCommandMode() {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    sensor_->commandGotoCommandMode();
    log("Goto command mode");
    return true;
}

bool ImuWrapper::gotoStreamingMode() {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    sensor_->commandGotoStreamingMode();
    log("Goto streaming mode");
    return true;
}

bool ImuWrapper::resetHeading() {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    sensor_->commandSetOffsetMode(LPMS_OFFSET_MODE_HEADING);
    log("Reset heading");
    return true;
}

bool ImuWrapper::configureStreaming(uint32_t freq_hz, uint32_t tdr_mask) {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    // NOTE: every command* call in LpmsIG1 CLEARS the internal command queue
    // before enqueueing, so calls must be spaced out to let the queue drain.
    log("Set stream frequency " + std::to_string(freq_hz) + " Hz");
    sensor_->commandSetSensorFrequency(freq_hz);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (tdr_mask != 0) {
        log("Set transmit data mask 0x" + std::to_string(tdr_mask));
        sensor_->commandSetTransmitData(tdr_mask);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    log("Save parameters to sensor flash");
    sensor_->commandSaveParameters();
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    sensor_->commandGotoStreamingMode();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    return true;
}

bool ImuWrapper::setUartBaudrate(uint32_t baud) {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    // Order matters: persist everything else BEFORE touching the baudrate
    // (configureStreaming already saved). If the sensor switches its UART
    // immediately, the trailing save may be lost -- the startup probe list
    // covers both outcomes (sensor at new or old baud).
    log("Set sensor UART baudrate " + std::to_string(baud));
    sensor_->commandSetUartBaudRate(baud);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    sensor_->commandSaveParameters();
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    return true;
}

bool ImuWrapper::setAutoReconnect(bool enable) {
    if (!sensor_) {
        return false;
    }
    
    sensor_->setAutoReconnectStatus(enable);
    log("Auto reconnect " + std::string(enable ? "enabled" : "disabled"));
    return true;
}

bool ImuWrapper::getAutoReconnect() const {
    return sensor_ ? sensor_->getAutoReconnectStatus() : false;
}

bool ImuWrapper::hasImuData() const {
    return sensor_ ? sensor_->hasImuData() : false;
}

bool ImuWrapper::getImuData(IG1ImuDataI& data) {
    if (!sensor_) {
        return false;
    }
    
    return sensor_->getImuData(data);
}

float ImuWrapper::getDataFrequency() const {
    return sensor_ ? sensor_->getDataFrequency() : 0.0f;
}

bool ImuWrapper::getSensorInfo(IG1InfoI& info) {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    sensor_->getInfo(info);
    return true;
}

bool ImuWrapper::getSensorSettings(IG1SettingsI& settings) {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    sensor_->getSettings(settings);
    return true;
}

bool ImuWrapper::startDataStream(ImuDataCallback callback) {
    if (!isConnected()) {
        log("Error: Sensor not connected");
        return false;
    }
    
    if (streamRunning_) {
        log("Error: Data stream already running");
        return false;
    }
    
    dataCallback_ = callback;
    shouldStop_ = false;
    streamRunning_ = true;
    
    // Start streaming thread
    streamThread_ = std::make_unique<std::thread>(&ImuWrapper::streamTask, this);
    
    log("Data stream started");
    return true;
}

void ImuWrapper::stopDataStream() {
    if (!streamRunning_) {
        return;
    }
    
    shouldStop_ = true;
    streamRunning_ = false;
    
    if (streamThread_ && streamThread_->joinable()) {
        streamThread_->join();
    }
    
    streamThread_.reset();
    log("Data stream stopped");
}

bool ImuWrapper::isStreaming() const {
    return streamRunning_;
}

int ImuWrapper::getStatus() const {
    return sensor_ ? sensor_->getStatus() : STATUS_CONNECTION_ERROR;
}

std::string ImuWrapper::getStatusString() const {
    return statusToString(getStatus());
}

void ImuWrapper::setVerbose(int level) {
    if (sensor_) {
        sensor_->setVerbose(level);
    }
}

void ImuWrapper::log(const std::string& message) {
    printf("[INFO] [IMU_WRAPPER]: %s\n", message.c_str());
}

void ImuWrapper::streamTask() {
    if (!sensor_ || sensor_->getStatus() != STATUS_CONNECTED) {
        streamRunning_ = false;
        log("Sensor is not connected. Stream task terminated");
        return;
    }
    
    sensor_->commandGotoStreamingMode();
    
    while (sensor_->getStatus() == STATUS_CONNECTED && streamRunning_ && !shouldStop_) {
        IG1ImuDataI sd;
        if (sensor_->hasImuData()) {
            if (sensor_->getImuData(sd)) {
                // Call callback if provided
                if (dataCallback_) {
                    dataCallback_(sd);
                } else {
                    // Default printing behavior
                    printDataTask(sd);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    
    streamRunning_ = false;
    log("Stream task terminated");
}

void ImuWrapper::printDataTask(IG1ImuDataI sd) {
        float freq = sensor_->getDataFrequency();
        printf("[INFO] [IMU_WRAPPER]: t(s): %.3f acc: %+2.2f %+2.2f %+2.2f gyr: %+3.2f %+3.2f %+3.2f euler: %+3.2f %+3.2f %+3.2f Hz:%3.3f \r\n", 
            sd.timestamp*0.002f, 
            sd.accCalibrated.data[0], sd.accCalibrated.data[1], sd.accCalibrated.data[2],
            sd.gyroIIAlignmentCalibrated.data[0], sd.gyroIIAlignmentCalibrated.data[1], sd.gyroIIAlignmentCalibrated.data[2],
            sd.euler.data[0], sd.euler.data[1], sd.euler.data[2], 
            freq);
}

std::string ImuWrapper::statusToString(int status) const {
    switch (status) {
        case STATUS_DISCONNECTED:
            return "DISCONNECTED";
        case STATUS_CONNECTING:
            return "CONNECTING";
        case STATUS_CONNECTED:
            return "CONNECTED";
        case STATUS_CONNECTION_ERROR:
            return "CONNECTION_ERROR";
        default:
            return "UNKNOWN";
    }
} 