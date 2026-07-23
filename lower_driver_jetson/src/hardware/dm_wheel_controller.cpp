#include "dm_wheel_controller.h"

#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
constexpr uint16_t kVelModeIdOffset = 0x200;   // velocity-mode control frames
constexpr uint16_t kRegFrameId = 0x7FF;        // parameter read/write frames
constexpr uint8_t kRidCtrlMode = 0x0A;         // CTRL_MODE register
constexpr uint32_t kCtrlModeVelocity = 3;

float uint_to_float(int x_int, float x_min, float x_max, int bits) {
    const float span = x_max - x_min;
    return static_cast<float>(x_int) * span
           / static_cast<float>((1 << bits) - 1) + x_min;
}
}  // namespace

DmWheelController::~DmWheelController() { close_bus(); }

bool DmWheelController::init(const char* ifname) {
    char cmd[160];
    snprintf(cmd, sizeof(cmd),
             "sudo ip link set %s type can bitrate 1000000 2>/dev/null", ifname);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "sudo ifconfig %s up 2>/dev/null", ifname);
    system(cmd);
    snprintf(cmd, sizeof(cmd), "sudo ifconfig %s txqueuelen 65536 2>/dev/null",
             ifname);
    system(cmd);

    fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (fd_ < 0) {
        fprintf(stderr, "WARN: wheel CAN socket failed -- MOBILE wheels off\n");
        return false;
    }
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    if (ioctl(fd_, SIOCGIFINDEX, &ifr) < 0) {
        fprintf(stderr,
                "WARN: wheel CAN interface %s not found -- MOBILE wheels off\n",
                ifname);
        close(fd_);
        fd_ = -1;
        return false;
    }
    struct sockaddr_can addr;
    memset(&addr, 0, sizeof(addr));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        fprintf(stderr, "WARN: wheel CAN bind(%s) failed -- MOBILE wheels off\n",
                ifname);
        close(fd_);
        fd_ = -1;
        return false;
    }
    fcntl(fd_, F_SETFL, O_NONBLOCK);

    // Start from a known state: everything disabled (freewheel).
    for (int i = 0; i < kNumWheels; i++) {
        send_ff_tail(kFirstEscId + i, 0xFD);
        usleep(2000);
    }
    printf("Wheels: 3x DaMiao DM-H6215 (%s, VELOCITY mode, IDs %d-%d)\n",
           ifname, kFirstEscId, kFirstEscId + kNumWheels - 1);
    return true;
}

void DmWheelController::close_bus() {
    if (fd_ < 0) return;
    for (int i = 0; i < kNumWheels; i++) {
        send_speed(kFirstEscId + i, 0.0f);
    }
    usleep(2000);
    for (int i = 0; i < kNumWheels; i++) {
        send_ff_tail(kFirstEscId + i, 0xFD);
    }
    usleep(10000);
    close(fd_);
    fd_ = -1;
}

void DmWheelController::update(bool armed, const float* w_des_rad_s) {
    if (fd_ < 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (armed) {
        const float age_s = armed_prev_
            ? std::chrono::duration<float>(now - last_enable_t_).count()
            : 1e9f;
        if (age_s > 0.5f) {
            // Rising edge + 2 Hz refresh: force the volatile CTRL_MODE
            // register to velocity, then enable. Harmless when already
            // enabled; re-arms a motor that self-disabled (fault/power dip)
            // without operator intervention.
            for (int i = 0; i < kNumWheels; i++) {
                send_ctrl_mode_velocity(kFirstEscId + i);
            }
            for (int i = 0; i < kNumWheels; i++) {
                send_ff_tail(kFirstEscId + i, 0xFC);
            }
            last_enable_t_ = now;
        }
        for (int i = 0; i < kNumWheels; i++) {
            send_speed(kFirstEscId + i, w_des_rad_s[i]);
        }
    } else if (armed_prev_) {
        // Falling edge: stop, then drop out of Enable Mode (freewheel).
        for (int i = 0; i < kNumWheels; i++) {
            send_speed(kFirstEscId + i, 0.0f);
        }
        for (int i = 0; i < kNumWheels; i++) {
            send_ff_tail(kFirstEscId + i, 0xFD);
        }
    }
    armed_prev_ = armed;
    receive();
}

void DmWheelController::send_ff_tail(int esc_id, uint8_t tail) {
    struct can_frame f;
    memset(&f, 0, sizeof(f));
    f.can_id = kVelModeIdOffset + esc_id;
    f.can_dlc = 8;
    memset(f.data, 0xFF, 7);
    f.data[7] = tail;
    write(fd_, &f, sizeof(f));
}

void DmWheelController::send_ctrl_mode_velocity(int esc_id) {
    struct can_frame f;
    memset(&f, 0, sizeof(f));
    f.can_id = kRegFrameId;
    f.can_dlc = 8;
    f.data[0] = static_cast<uint8_t>(esc_id & 0xFF);        // CANID_L
    f.data[1] = static_cast<uint8_t>((esc_id >> 8) & 0xFF); // CANID_H
    f.data[2] = 0x55;                                       // write parameter
    f.data[3] = kRidCtrlMode;
    const uint32_t mode = kCtrlModeVelocity;                // uint32 LE
    memcpy(&f.data[4], &mode, 4);
    write(fd_, &f, sizeof(f));
}

void DmWheelController::send_speed(int esc_id, float v_rad_s) {
    struct can_frame f;
    memset(&f, 0, sizeof(f));
    f.can_id = kVelModeIdOffset + esc_id;
    f.can_dlc = 4;
    memcpy(f.data, &v_rad_s, 4);   // float32, little-endian on this platform
    write(fd_, &f, sizeof(f));
}

void DmWheelController::receive() {
    struct can_frame f;
    while (read(fd_, &f, sizeof(f)) > 0) {
        if (f.can_dlc < 8) continue;                 // register echoes etc.
        const int esc_id = f.data[0] & 0x0F;
        const uint8_t err = f.data[0] >> 4;
        const int idx = esc_id - kFirstEscId;
        if (idx < 0 || idx >= kNumWheels) continue;
        status[idx] = err;
        if (err >= 0x8) {
            static int fault_print_cnt = 0;
            if ((fault_print_cnt++ % 500) == 0) {
                fprintf(stderr, "WHEEL W%d FAULT status=0x%X\n", esc_id, err);
            }
        }
        const int v_int = (f.data[3] << 4) | (f.data[4] >> 4);
        vel_rad_s[idx] = uint_to_float(v_int, -kFbVMax, kFbVMax, 12);
    }
}
