#ifndef IMU_WRAPPER_H
#define IMU_WRAPPER_H

#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>
#include "LpmsIG1I.h"
#include "LpmsIG1Registers.h"

class ImuWrapper {
public:
    // Callback function type for IMU data
    using ImuDataCallback = std::function<void(const IG1ImuDataI&)>;
    
    // Constructor
    ImuWrapper();
    
    // Destructor
    ~ImuWrapper();
    
    // Connection management
    bool connect(const std::string& port = "/dev/ttyUSB0", int baudrate = 115200);
    void disconnect();
    bool isConnected() const;
    
    // Mode control
    bool gotoCommandMode();
    bool gotoStreamingMode();
    
    // Sensor configuration
    bool resetHeading();
    bool setAutoReconnect(bool enable);
    bool getAutoReconnect() const;
    
    // Data retrieval
    bool hasImuData() const;
    bool getImuData(IG1ImuDataI& data);
    float getDataFrequency() const;
    
    // Information and settings
    bool getSensorInfo(IG1InfoI& info);
    bool getSensorSettings(IG1SettingsI& settings);
    
    // Data streaming
    bool startDataStream(ImuDataCallback callback = nullptr);
    void stopDataStream();
    bool isStreaming() const;
    
    // Status
    int getStatus() const;
    std::string getStatusString() const;
    
    // Utility
    void setVerbose(int level);
    void log(const std::string& message);

private:
    // Member variables
    std::unique_ptr<IG1I> sensor_;
    std::unique_ptr<std::thread> streamThread_;
    std::atomic<bool> streamRunning_;
    std::atomic<bool> shouldStop_;
    ImuDataCallback dataCallback_;
    
    // Internal methods
    void streamTask();
    void printDataTask(IG1ImuDataI sd);
    std::string statusToString(int status) const;
    
    // Disable copy constructor and assignment operator
    ImuWrapper(const ImuWrapper&) = delete;
    ImuWrapper& operator=(const ImuWrapper&) = delete;
};

#endif // IMU_WRAPPER_H 