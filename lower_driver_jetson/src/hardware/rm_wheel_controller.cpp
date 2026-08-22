#include "rm_wheel_controller.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <math.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
constexpr uint16_t kCmdId = 0x200;          // C610 current cmd for IDs 1..4
constexpr float kPi = 3.14159265358979323846f;

inline float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

inline int16_t be_i16(const uint8_t* p) {
    return static_cast<int16_t>((static_cast<uint16_t>(p[0]) << 8)
                                | static_cast<uint16_t>(p[1]));
}

inline void put_be_i16(uint8_t* p, int16_t v) {
    p[0] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[1] = static_cast<uint8_t>(v & 0xFF);
}
}  // namespace

RmWheelController::~RmWheelController() { close_bus(); }

bool RmWheelController::init(const char* ifname) {
    strncpy(ifname_, ifname, sizeof(ifname_) - 1);
    ifname_[sizeof(ifname_) - 1] = '\0';
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

    // Coast on bring-up.
    float zero[kNumWheels] = {0.0f, 0.0f, 0.0f};
    send_currents(zero);
    printf("Wheels: 3x RM M2006/C610 (%s, speed-PI current, IDs %d-%d)\n",
           ifname, kFirstEscId, kFirstEscId + kNumWheels - 1);
    init_logged_ok_ = true;
    return true;
}

void RmWheelController::try_reinit() {
    const auto now = std::chrono::steady_clock::now();
    if (reinit_t_valid_ &&
        std::chrono::duration<float>(now - last_reinit_t_).count()
            < kReinitPeriodS) {
        return;
    }
    last_reinit_t_ = now;
    reinit_t_valid_ = true;
    // Cheap presence probe first so we don't spawn `sudo ip link` every 2 s
    // while the adapter is unplugged.
    if (if_nametoindex(ifname_) == 0) return;
    for (int i = 0; i < kNumWheels; i++) {
        online[i] = false;
        iq_i_[i] = 0.0f;
    }
    if (init(ifname_)) {
        printf("Wheels: %s recovered -- MOBILE wheels back online\n", ifname_);
    }
}

void RmWheelController::close_bus() {
    if (fd_ < 0) return;
    float zero[kNumWheels] = {0.0f, 0.0f, 0.0f};
    send_currents(zero);
    usleep(5000);
    close(fd_);
    fd_ = -1;
}

void RmWheelController::update(bool armed, const float* w_des_rad_s) {
    if (fd_ < 0) {
        try_reinit();
        if (fd_ < 0) return;
    }
    receive();

    const auto now = std::chrono::steady_clock::now();
    float dt = kDtNom;
    if (tick_inited_) {
        dt = std::chrono::duration<float>(now - last_tick_t_).count();
        dt = clampf(dt, 0.0005f, 0.02f);
    }
    last_tick_t_ = now;
    tick_inited_ = true;

    float iq_cmd[kNumWheels] = {0.0f, 0.0f, 0.0f};
    if (armed) {
        for (int i = 0; i < kNumWheels; i++) {
            const float e = w_des_rad_s[i] - vel_rad_s[i];
            iq_i_[i] = clampf(
                iq_i_[i] + kKiAPerRadS2 * e * dt, -kIqMaxA, kIqMaxA);
            // Freeze integrator when no fresh feedback (avoid windup into
            // a dead ESC).
            if (!online[i]) iq_i_[i] = 0.0f;
            iq_cmd[i] = clampf(
                kKpAPerRadS * e + iq_i_[i], -kIqMaxA, kIqMaxA);
        }
    } else {
        for (int i = 0; i < kNumWheels; i++) iq_i_[i] = 0.0f;
        if (armed_prev_) {
            // Falling edge: one explicit zero frame, then keep streaming 0.
            send_currents(iq_cmd);
        }
    }
    armed_prev_ = armed;
    send_currents(iq_cmd);
}

void RmWheelController::send_currents(const float* iq_a_cmd) {
    struct can_frame f;
    memset(&f, 0, sizeof(f));
    f.can_id = kCmdId;
    f.can_dlc = 8;
    for (int i = 0; i < kNumWheels; i++) {
        // A -> C610 raw (±10000 = ±10 A), then big-endian packing.
        float raw = clampf(iq_a_cmd[i], -10.0f, 10.0f) * 1000.0f;
        put_be_i16(&f.data[2 * i], static_cast<int16_t>(raw));
    }
    // ID 4 slot left at 0 (unused).
    if (write(fd_, &f, sizeof(f)) < 0 &&
        (errno == ENODEV || errno == ENXIO || errno == EBADF)) {
        // Netdev died or was re-created with a new ifindex (USB drop /
        // replug): this socket is deaf forever. Drop it; update() rebinds.
        fprintf(stderr,
                "WARN: wheel CAN %s vanished (errno %d) -- wheels off, "
                "will auto-rebind\n", ifname_, errno);
        close(fd_);
        fd_ = -1;
        init_logged_ok_ = false;
        for (int i = 0; i < kNumWheels; i++) online[i] = false;
    }
}

void RmWheelController::receive() {
    struct can_frame f;
    while (read(fd_, &f, sizeof(f)) > 0) {
        if (f.can_dlc < 6) continue;
        const int esc_id = static_cast<int>(f.can_id) - 0x200;
        const int idx = esc_id - kFirstEscId;
        if (idx < 0 || idx >= kNumWheels) continue;
        // DATA[2:3] rotor RPM (BE int16); DATA[4:5] torque current raw.
        const int16_t rpm = be_i16(&f.data[2]);
        const int16_t iq_raw = be_i16(&f.data[4]);
        // Output shaft (after 36:1): rad/s = rpm * 2pi/60 / 36.
        vel_rad_s[idx] = (static_cast<float>(rpm) * (2.0f * kPi) / 60.0f)
                         / kGear;
        iq_a[idx] = static_cast<float>(iq_raw) / 1000.0f;
        online[idx] = true;
    }
}
