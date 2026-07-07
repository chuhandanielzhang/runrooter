#include <iostream>
#include <thread>
#include <chrono>
#include <functional>

#include "ImuWrapper.h"

using namespace std;

// Custom callback function for IMU data
void customImuDataCallback(const IG1ImuDataI& data) {
    printf("[CUSTOM] Timestamp: %.3f, Euler: %+3.2f %+3.2f %+3.2f\n", 
           data.timestamp * 0.002f,
           data.euler.data[0], data.euler.data[1], data.euler.data[2]);
}

void printMenu() {
    cout << endl;
    cout << "===================" << endl;
    cout << "IMU Wrapper Menu" << endl;
    cout << "===================" << endl;
    cout << "[c] Connect sensor" << endl;
    cout << "[d] Disconnect sensor" << endl;
    cout << "[0] Goto command mode" << endl;
    cout << "[1] Goto streaming mode" << endl;
    cout << "[h] Reset sensor heading" << endl;
    cout << "[i] Print sensor info" << endl;
    cout << "[s] Print sensor settings" << endl;
    cout << "[p] Start data stream (default)" << endl;
    cout << "[P] Start data stream (custom callback)" << endl;
    cout << "[x] Stop data stream" << endl;
    cout << "[a] Toggle auto reconnect" << endl;
    cout << "[q] quit" << endl;
    cout << endl;
}

int main(int argc, char** argv) {
    string comportNo = "/dev/ttyUSB0";
    int baudrate = 115200;
    
    if (argc == 2) {
        comportNo = string(argv[1]);
    }
    else if (argc == 3) {
        comportNo = string(argv[1]);
        baudrate = atoi(argv[2]);
    }
    
    cout << "IMU Wrapper Example" << endl;
    cout << "Connecting to " << comportNo << " @ " << baudrate << endl;
    
    // Create IMU wrapper instance
    ImuWrapper imu;
    
    // Connect to sensor
    if (!imu.connect(comportNo, baudrate)) {
        cout << "Failed to connect to sensor. Exiting." << endl;
        return 1;
    }
    
    cout << "Sensor connected successfully!" << endl;
    printMenu();
    
    bool quit = false;
    while (!quit) {
        char cmd = '\0';
        string input;
        getline(cin, input);
        
        if (!input.empty())
            cmd = input.at(0);
            
        switch (cmd) {
        case 'c':
            cout << "Connect sensor" << endl;
            if (!imu.connect(comportNo, baudrate))
                cout << "Error connecting to sensor" << endl;
            break;

        case 'd':
            cout << "Disconnect sensor" << endl;
            imu.disconnect();
            break;

        case '0':
            cout << "Goto command mode" << endl;
            imu.gotoCommandMode();
            break;

        case '1':
            cout << "Goto streaming mode" << endl;
            imu.gotoStreamingMode();
            break;

        case 'h':
            cout << "Reset heading" << endl;
            imu.resetHeading();
            break;

        case 'i': {
            cout << "Print sensor info" << endl;
            IG1InfoI info;
            if (imu.getSensorInfo(info)) {
                cout << info.toString() << endl;
            } else {
                cout << "Failed to get sensor info" << endl;
            }
            break;
        }

        case 's': {
            cout << "Print sensor settings" << endl;
            IG1SettingsI settings;
            if (imu.getSensorSettings(settings)) {
                cout << settings.toString() << endl;
            } else {
                cout << "Failed to get sensor settings" << endl;
            }
            break;
        }

        case 'p': {     
            cout << "Start data stream (default)" << endl;       
            if (!imu.startDataStream()) {
                cout << "Failed to start data stream" << endl;
            }
            break;
        }

        case 'P': {     
            cout << "Start data stream (custom callback)" << endl;       
            if (!imu.startDataStream(customImuDataCallback)) {
                cout << "Failed to start data stream" << endl;
            }
            break;
        }

        case 'x': {
            cout << "Stop data stream" << endl;
            imu.stopDataStream();
            break;
        }

        case 'a': {
            cout << "Toggle auto reconnect" << endl;
            bool current = imu.getAutoReconnect();
            imu.setAutoReconnect(!current);
            cout << "Auto reconnect " << (!current ? "enabled" : "disabled") << endl;
            break;
        }

        case 'q':
            cout << "Quitting..." << endl;
            imu.disconnect();
            quit = true;
            break;

        default:
            printMenu();
            break;
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    cout << "Bye!" << endl;
    return 0;
} 