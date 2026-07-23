#include "hopper_hardware.hpp"

#include <cstdlib>
#include <string>
#include <thread>

static inline float wrap_to_pi(float a) {
    a = std::fmod(a + static_cast<float>(M_PI), 2.0f * static_cast<float>(M_PI));
    if (a < 0.0f) a += 2.0f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
}

// ===== Delta leg motor ordering (REAL ROBOT) =====
// We publish/consume the delta leg motors on LCM as q[0],q[1],q[2] and tau_ff[0],tau_ff[1],tau_ff[2].
// This array maps that **LCM joint index** -> **AK60 motor_id** (0..2).
//
// IMPORTANT:
// - Keep this mapping consistent for BOTH: sending commands AND reading states.
// - If your wiring / CAN IDs differ, change this array (do not scatter swaps across the file).
//
// LCM joint index rotation (2026-06-24), applied twice at LCM source:
//   1st: q0<-q2, q1<-q0, q2<-q1  => {2,0,1}
//   2nd: same rule again           => {1,2,0}
//   (2026-06-27) identity mapping {0,1,2}: LCM q0->motor 0, q1->motor 1, q2->motor 2
static constexpr int kDeltaAk60MotorIdFromJointIdx[3] = {0, 1, 2};
// LCM q = -motor_pos + offset. Full extension -> q_lcm = 0 when motor_pos = offset.
static constexpr float kAk60LcmQOffsetRad = 1.4835f;  // +85 deg (reverted from +90 per user 2026-07-10)

HopperHardware::HopperHardware(bool is_publish_lcm_data):
        Controller2Robot("udpm://239.255.76.67:7667?ttl=255"),
        Robot2Controller("udpm://239.255.76.67:7667?ttl=255"),
    lcmreceiveTask(&task_manager_, 0.001, "lcmrecv_task", &HopperHardware::_thread_lcmrec_run, this),  // 1000Hz receive (drains ALL pending msgs per tick)
    lcmsendTask   (&task_manager_, 0.001, "lcmsend_task", &HopperHardware::_thread_lcmsen_run, this),  // 1000Hz send
    imuTask       (&task_manager_, 0.001, "imu_task", &HopperHardware::_thread_imu_run, this),           // 1000Hz IMU publish
    gamepadTask   (&task_manager_, 0.005, "gamepad_task", &HopperHardware::_thread_xbox_run, this),   // 200Hz gamepad
        motor_pos(3, 0.0),
        motor_vel(3, 0.0),
        motor_tau(3, 0.0),
        motor_vel_diff(3, 0.0),
        motor_pos_prev(3, 0.0)
{
    this->is_publish_lcm_data = is_publish_lcm_data;
    set_value_zero(); // clear all the value
    // initialize the leg motor controller (3x AK60 over SocketCAN / can0)
    ak60_controller_ptr_ = new AK60Controller();
    rot_offset_ = coordinateRotation(CoordinateAxis::Z, 0.0f)
                * coordinateRotation(CoordinateAxis::Y, 0.0f)
                * coordinateRotation(CoordinateAxis::X, 0.0f);
    // initialize the Lpms IG1 IMU (same as CASE hopper_driver-master)
    // 2026-07-09: run the IG1 at 500 Hz (factory default streams 100 Hz, which
    // held every gyro/rpy sample for ~5 control ticks). 500 Hz does not fit in
    // 115200 baud, so the sensor UART is converted to 921600 and the payload
    // trimmed to the fields this driver consumes.
    imu_wrapper_ptr_ = new ImuWrapper();
    {
        const uint32_t kImuTdrMask =
            TDR_ACC_CALIBRATED_OUTPUT_ENABLED |
            TDR_GYR1_BIAS_CALIBRATED_OUTPUT_ENABLED |   // parsed as gyroIIBiasCalibrated
            TDR_QUAT_OUTPUT_ENABLED |
            TDR_EULER_OUTPUT_ENABLED;
        const int kImuTargetBaud = LPMS_UART_BAUDRATE_921600;
        // Probe target baud first (sensor already converted on a previous run),
        // then factory default, then intermediates in case a conversion half-landed.
        const int bauds[] = {kImuTargetBaud, LPMS_UART_BAUDRATE_115200,
                             LPMS_UART_BAUDRATE_460800, LPMS_UART_BAUDRATE_230400};
        const char* env_port = std::getenv("HOPPER_IMU_PORT");
        std::vector<std::string> ports;
        if (env_port != nullptr && env_port[0] != '\0') ports.push_back(env_port);
        ports.push_back("/dev/ttyUSB0");
        ports.push_back("/dev/ttyUSB1");
        // NO ttyACM* fallback: since 2026-07-23 /dev/ttyACM2 is the wheel-bus
        // CANable2 (see 99-canable.rules), and the IG1 baud probe wedges its
        // CDC endpoint (write -> EIO until USB re-enumeration). If the IG1 is
        // ever attached over ACM again, point HOPPER_IMU_PORT at it.

        std::string imu_port;
        int imu_baud = 0;
        for (const std::string& p : ports) {
            for (int b : bauds) {
                if (imu_wrapper_ptr_->connect(p, b)) {
                    imu_port = p;
                    imu_baud = b;
                    break;
                }
            }
            if (imu_baud != 0) break;
        }

        if (imu_baud == 0) {
            std::cerr << "WARN: Lpms IMU connect failed (tried HOPPER_IMU_PORT, ttyUSB0/1, ttyACM2)."
                      << std::endl;
        } else {
            std::cout << "Lpms IMU connected on " << imu_port << " @ " << imu_baud << std::endl;
            if (imu_baud != kImuTargetBaud) {
                // Convert the sensor UART BEFORE raising the stream rate, so we
                // never stream 500 Hz into a saturated 115200 link.
                std::cout << "IMU: converting sensor UART " << imu_baud << " -> "
                          << kImuTargetBaud << std::endl;
                imu_wrapper_ptr_->setUartBaudrate(static_cast<uint32_t>(kImuTargetBaud));
                imu_wrapper_ptr_->disconnect();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (imu_wrapper_ptr_->connect(imu_port, kImuTargetBaud)) {
                    imu_baud = kImuTargetBaud;
                } else if (imu_wrapper_ptr_->connect(imu_port, imu_baud)) {
                    // Sensor kept the old baud (some firmwares apply it only after
                    // a power cycle); stay at a stream rate the link can carry.
                    std::cerr << "WARN: IMU stayed at " << imu_baud
                              << " baud; will retry conversion next boot." << std::endl;
                } else {
                    std::cerr << "WARN: IMU reconnect after baud conversion failed." << std::endl;
                    imu_baud = 0;
                }
            }
            if (imu_baud != 0) {
                // Pick the highest stream rate the serial link can carry
                // (~70 B/packet with the trimmed TDR incl. LP-BUS framing).
                const uint32_t hz = (imu_baud >= LPMS_UART_BAUDRATE_460800) ? 500u
                                  : (imu_baud >= LPMS_UART_BAUDRATE_230400) ? 250u : 100u;
                imu_wrapper_ptr_->configureStreaming(hz, kImuTdrMask);
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::cout << "IMU stream rate: " << imu_wrapper_ptr_->getDataFrequency()
                          << " Hz (target " << hz << " Hz @ " << imu_baud << " baud)" << std::endl;
            }
        }
    }
    // MOBILE kiwi wheels: 3x DaMiao DM-H6215, velocity mode, dedicated can1.
    // init() degrades gracefully (prints a WARN and no-ops) when the bus is
    // not attached, so hop-only operation is unaffected.
    wheel_controller_ptr_ = new DmWheelController();
    wheel_controller_ptr_->init("can1");
    std::cout << "Driver: 3x DaMiao DM4310 (can0, MIT, IDs 1-3) + Lpms IMU -> hopper_data_lcmt / hopper_imu_lcmt @ 500Hz" << std::endl;
    std::cout << "Propellers remain on Pixhawk (px4_bridge); disable px4-dds-bridge to avoid duplicate IMU." << std::endl;
    // initialize the xbox controller
    xbox_controller_ptr_ = new XboxController();

    start_threads();

}
XboxController::XboxMap HopperHardware::get_xbox_map(){
    return xbox_controller_ptr_->getMap();
}

void HopperHardware::_store_motor_state(const float* temp_pos, const float* temp_vel, const float* temp_tau){
    // 2026-07-07: qd published on LCM = the AK60 CAN velocity report (temp_vel
    // -> motor_vel). The position-difference velocity (motor_vel_diff) is still
    // computed here but only kept for local debugging/comparison.
    const auto now = std::chrono::steady_clock::now();
    float vel_diff[3] = {0.0f, 0.0f, 0.0f};
    if (motor_diff_init_) {
        float dt = std::chrono::duration<float>(now - motor_diff_prev_t_).count();
        // Nominal tick is 2ms; clamp dt so a scheduling hiccup or duplicate
        // call cannot blow up the derivative.
        if (dt < 5e-4f) dt = 5e-4f;
        if (dt > 2e-2f) dt = 2e-2f;
        for (int j = 0; j < 3; j++) {
            vel_diff[j] = (temp_pos[j] - motor_pos_prev[j]) / dt;
        }
    }
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];        // raw CAN report, debug only
            motor_tau[j] = temp_tau[j];
            motor_vel_diff[j] = vel_diff[j];   // -> hopper_data_lcmt.qd
        }
    }
    for (int j = 0; j < 3; j++) motor_pos_prev[j] = temp_pos[j];
    motor_diff_prev_t_ = now;
    motor_diff_init_ = true;
}

int HopperHardware::get_motor_pwm_control_mode() {
    std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
    return int(motor_pwm_lcmt_.control_mode);
}

void HopperHardware::clear_motor_pwm_control_mode() {
    std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
    motor_pwm_lcmt_.control_mode = 0;
}
void HopperHardware::step_with_only_receiving(){
    if(step_counter== 0)
    {
        ak60_controller_ptr_->enableMotors();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for(int i = 0; i < NUM_MOTORS; i++){
        ak60_controller_ptr_->setMotorParams(i, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();


    float temp_pos[3], temp_vel[3], temp_tau[3];
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        ak60_controller_ptr_->getMotorState(motor_id, temp_pos[j], temp_vel[j], temp_tau[j]);
    }
    _store_motor_state(temp_pos, temp_vel, temp_tau);
    _publish_rm_cmd(false);   // OFF: M2006 zero current (leg-class gating)
    _update_wheels(false);    // OFF: wheels disabled (freewheel)

    step_counter++;
}

void HopperHardware::step_with_damping(){
    if(step_counter== 0)
    {
        ak60_controller_ptr_->enableMotors();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for(int i = 0; i < NUM_MOTORS; i++){
        ak60_controller_ptr_->setMotorParams(i, 0.0, 0.0, 0.0, 0.0, 0.8);
    }
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();


    float temp_pos[3], temp_vel[3], temp_tau[3];
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        ak60_controller_ptr_->getMotorState(motor_id, temp_pos[j], temp_vel[j], temp_tau[j]);
    }
    _store_motor_state(temp_pos, temp_vel, temp_tau);
    _publish_rm_cmd(false);   // DAMP (B): M2006 zero current, same as legs stopping
    _update_wheels(false);    // DAMP (B): wheels disabled (freewheel)

    step_counter++;
}

void HopperHardware::step_with_pd_control(){
    step_counter++;
    lcm_cmd_mutex.lock();

    // 关节PD控制（始终执行）
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        // LCM <-> motor sign convention MUST match `_fill_in_motor_data_to_lcm()`.
        // We publish:
        //   q_lcm  = -motor_pos + offset
        //   qd_lcm = -motor_vel
        // Therefore commands coming from the PC in LCM convention must be converted back:
        //   motor_pos_des = -q_des + offset
        //   motor_vel_des = -qd_des
        //   motor_tau_ff  = -tau_ff
        const float offset = kAk60LcmQOffsetRad;
        const float q_des_motor = -hopper_cmd_lcmt_.q_des[j] + offset;
        const float qd_des_motor = -hopper_cmd_lcmt_.qd_des[j];
        const float tau_ff_motor = -hopper_cmd_lcmt_.tau_ff[j];
        ak60_controller_ptr_->setMotorParams(
            motor_id,
            q_des_motor,
            qd_des_motor,
            tau_ff_motor,
            hopper_cmd_lcmt_.kp_joint[j],
            hopper_cmd_lcmt_.kd_joint[j]
        );
    }
    lcm_cmd_mutex.unlock();
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();

    // 读取关节状态
    float temp_pos[3], temp_vel[3], temp_tau[3];
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        ak60_controller_ptr_->getMotorState(motor_id, temp_pos[j], temp_vel[j], temp_tau[j]);
    }
    _store_motor_state(temp_pos, temp_vel, temp_tau);
    _publish_rm_cmd(true);    // PD (X): M2006 armed, forward rm_iq_des
    _update_wheels(true);     // PD (X): wheels armed (msg.enable still gates)
}

void HopperHardware::step_with_pd_pwm_control(){
    // Legs-only build: legs follow PD exactly like step_with_pd_control().
    // Propeller PWM is no longer emitted here; the Pixhawk (px4_bridge) consumes
    // motor_pwm_lcmt directly and drives the props via DShot.
    step_counter++;

    // Copy LCM commands under lock (avoid races between recv thread and control loop)
    lcm_cmd_mutex.lock();
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        // Same sign/offset conversion as in `step_with_pd_control()`.
        const float offset = kAk60LcmQOffsetRad;
        const float q_des_motor = -hopper_cmd_lcmt_.q_des[j] + offset;
        const float qd_des_motor = -hopper_cmd_lcmt_.qd_des[j];
        const float tau_ff_motor = -hopper_cmd_lcmt_.tau_ff[j];
        ak60_controller_ptr_->setMotorParams(
            motor_id,
            q_des_motor,
            qd_des_motor,
            tau_ff_motor,
            hopper_cmd_lcmt_.kp_joint[j],
            hopper_cmd_lcmt_.kd_joint[j]
        );
    }
    lcm_cmd_mutex.unlock();
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();

    float temp_pos[3], temp_vel[3], temp_tau[3];
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        ak60_controller_ptr_->getMotorState(motor_id, temp_pos[j], temp_vel[j], temp_tau[j]);
    }
    _store_motor_state(temp_pos, temp_vel, temp_tau);
    _publish_rm_cmd(true);    // PWMPD: M2006 armed, forward rm_iq_des
    _update_wheels(true);     // PWMPD: wheels armed (msg.enable still gates)
}

void HopperHardware::step_with_pwm_only(){
    // Legs-only build: leg motors are left FREE (zero torque, kp=kd=0).
    // Propellers used to be driven here; they are now produced on the Pixhawk
    // side (px4_bridge consumes motor_pwm_lcmt). This mode therefore just keeps
    // the legs free while the props are commanded independently.
    if(step_counter == 0)
    {
        ak60_controller_ptr_->enableMotors();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Free legs: zero everything. Same as step_with_only_receiving() for the legs.
    for(int i = 0; i < NUM_MOTORS; i++){
        ak60_controller_ptr_->setMotorParams(i, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();

    // Read joint state (still publish to LCM so the PC controller can monitor)
    float temp_pos[3], temp_vel[3], temp_tau[3];
    for (int j = 0; j < 3; j++) {
        const int motor_id = kDeltaAk60MotorIdFromJointIdx[j];
        ak60_controller_ptr_->getMotorState(motor_id, temp_pos[j], temp_vel[j], temp_tau[j]);
    }
    _store_motor_state(temp_pos, temp_vel, temp_tau);
    _publish_rm_cmd(false);   // PWM_ONLY: legs free -> M2006 zero current too
    _update_wheels(false);    // PWM_ONLY: wheels disabled (freewheel)

    step_counter++;
}

    void HopperHardware::start_threads(){
    lcmsendTask.start();
    imuTask.start();
    gamepadTask.start();
    Controller2Robot.subscribe("hopper_cmd_lcmt", &HopperHardware::handleController2RobotLCM, this);
    Controller2Robot.subscribe("motor_pwm_lcmt", &HopperHardware::handleMotorPwmLCM, this);
    Controller2Robot.subscribe("rm_esc_data_lcmt", &HopperHardware::handleRmEscDataLCM, this);
    Controller2Robot.subscribe("wheel_cmd_lcmt", &HopperHardware::handleWheelCmdLCM, this);
    // NOTE: do NOT subscribe gamepad_lcmt here. This driver is the PUBLISHER of
    // gamepad_lcmt; subscribing to it (empty handler) made our own 200Hz publication
    // loop back into the receive queue and waste dispatch slots ahead of hopper_cmd.
    lcmreceiveTask.start();
}

void HopperHardware::step_with_set_zero_mode(){
    std::cout<<"setting value zero, please hold still and wait..."<<std::endl;
    ak60_controller_ptr_->disableMotors();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ak60_controller_ptr_->setZero();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // Also zero the M2006 output-shaft angles: latch the current multi-turn
    // angle as the new origin (rm_q reads 0 at this pose from now on).
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int i = 0; i < 3; i++) {
            rm_q_offset_[i] = rm_esc_data_lcmt_.shaft_angle_rad[i];
        }
    }
    std::cout<<"set zero done (AK60 + M2006)"<<std::endl;
    step_counter = 0;
}
void HopperHardware::handleController2RobotLCM(const lcm::ReceiveBuffer* rbuf,
                                      const std::string& chan,
                                      const hopper_cmd_lcmt* msg){
                                    // std::cout<<"111";
    (void)rbuf;
    (void)chan;
    {
        std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
        memcpy(&hopper_cmd_lcmt_, msg, sizeof(hopper_cmd_lcmt));
    }
    // RM logical-position initialization (0 -> nonzero edge). Choose the
    // offset so the current physical shaft position reads rm_zero_at_rad:
    //   rm_q = shaft - offset  =>  offset = shaft - requested_q.
    // This is coordinate-only and never energizes the M2006s, so it is safe
    // in OFF/DAMP as required by the whole-robot actuator interlock.
    const bool rz_now = (msg->rm_set_zero != 0);
    if (rz_now && !rm_set_zero_prev_) {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        const float requested_q = msg->rm_zero_at_rad;
        for (int i = 0; i < 3; i++) {
            rm_q_offset_[i] =
                rm_esc_data_lcmt_.shaft_angle_rad[i] - requested_q;
        }
        std::cout << "RM logical init (LCM): current position -> "
                  << requested_q << " rad" << std::endl;
    }
    rm_set_zero_prev_ = rz_now;
}

void HopperHardware::handleGamepadLCM(const lcm::ReceiveBuffer* rbuf,
                                      const std::string& chan,
                                      const gamepad_lcmt* msg) {
    (void)rbuf;
    (void)chan;

}

void HopperHardware::handleMotorPwmLCM(const lcm::ReceiveBuffer* rbuf,
                                       const std::string& chan,
                                       const motor_pwm_lcmt* msg) {
    (void)rbuf;
    (void)chan;
    std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
    memcpy(&motor_pwm_lcmt_, msg, sizeof(motor_pwm_lcmt));
}

void HopperHardware::handleRmEscDataLCM(const lcm::ReceiveBuffer* rbuf,
                                        const std::string& chan,
                                        const rm_esc_data_lcmt* msg) {
    (void)rbuf;
    (void)chan;
    std::lock_guard<std::mutex> lock(motor_state_mutex);
    rm_esc_data_lcmt_ = *msg;
    rm_data_rx_t_ = std::chrono::steady_clock::now();
    rm_data_seen_ = true;
}

void HopperHardware::handleWheelCmdLCM(const lcm::ReceiveBuffer* rbuf,
                                       const std::string& chan,
                                       const wheel_cmd_lcmt* msg) {
    (void)rbuf;
    (void)chan;
    std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
    wheel_cmd_lcmt_ = *msg;
    wheel_cmd_rx_t_ = std::chrono::steady_clock::now();
    wheel_cmd_seen_ = true;
}

void HopperHardware::_update_wheels(bool enabled) {
    // Leg-class gate (same policy as _publish_rm_cmd) + msg.enable + a
    // 200 ms freshness watchdog: if the PC controller dies mid-drive the
    // wheels stop here, and the motor's own comm-loss protection is the
    // final layer below that.
    if (wheel_controller_ptr_ == nullptr) return;
    float w_des[3] = {0.0f, 0.0f, 0.0f};
    bool armed = false;
    {
        std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
        const bool fresh = wheel_cmd_seen_ &&
            std::chrono::duration<float>(
                std::chrono::steady_clock::now() - wheel_cmd_rx_t_).count()
            < 0.2f;
        armed = enabled && fresh && (wheel_cmd_lcmt_.enable != 0);
        if (armed) {
            for (int i = 0; i < 3; i++) {
                w_des[i] = wheel_cmd_lcmt_.speed_des_rad_s[i];
            }
        }
    }
    wheel_controller_ptr_->update(armed, w_des);
}

void HopperHardware::_publish_rm_cmd(bool enabled) {
    // hopper_cmd_lcmt.rm_iq_des is in AMPS; the bridge/PX4 protocol wants raw
    // C610 units (-10000..10000 = -10..+10 A). Gate: anything but PD/PWMPD
    // streams 0 A (coast). Downstream safety stays layered on top of this:
    // px4_dds_bridge zeroes after 0.2s without this message, and the PX4
    // rm_c610 module zeroes after 100ms without DDS input.
    rm_esc_cmd_lcmt cmd;
    cmd.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    float iq_des[3] = {0.0f, 0.0f, 0.0f};
    if (enabled) {
        std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
        for (int i = 0; i < 3; i++) iq_des[i] = hopper_cmd_lcmt_.rm_iq_des[i];
    }
    for (int i = 0; i < 3; i++) {
        float raw = iq_des[i] * 1000.0f;               // A -> raw (10A = 10000)
        if (raw > 10000.0f) raw = 10000.0f;
        if (raw < -10000.0f) raw = -10000.0f;
        cmd.current_raw[i] = static_cast<int16_t>(raw);
    }
    Robot2Controller.publish("rm_esc_cmd_lcmt", &cmd);
}

void HopperHardware::set_value_zero(){
    step_counter = 0;
    // ak60_controller_ptr_->set_value_zero();
    for(int i = 0; i < NUM_MOTORS; i++){
        hopper_cmd_lcmt_.tau_ff[i] = 0.0;
        hopper_cmd_lcmt_.kp_joint[i] = 0.0;
        hopper_cmd_lcmt_.kd_joint[i] = 0.0;
        hopper_cmd_lcmt_.q_des[i] = 0.0;
        hopper_cmd_lcmt_.qd_des[i] = 0.0;
        
        hopper_data_lcmt_.q[i] = 0.0;
        hopper_data_lcmt_.qd[i] = 0.0;
        hopper_data_lcmt_.tauIq[i] = 0.0;

        hopper_cmd_lcmt_.rm_iq_des[i] = 0.0;
        hopper_data_lcmt_.rm_q[i] = 0.0;
        hopper_data_lcmt_.rm_qd[i] = 0.0;
        hopper_data_lcmt_.rm_iq[i] = 0.0;
        rm_esc_data_lcmt_.shaft_angle_rad[i] = 0.0;
        rm_esc_data_lcmt_.shaft_speed_rad_s[i] = 0.0;
        rm_esc_data_lcmt_.current_raw[i] = 0;
        rm_esc_data_lcmt_.rpm[i] = 0;
        rm_esc_data_lcmt_.angle_raw[i] = 0;
    }
    hopper_cmd_lcmt_.rm_zero_at_rad = 0.0f;
    hopper_cmd_lcmt_.rm_set_zero = 0;
    for (int i = 0; i < 3; i++) wheel_cmd_lcmt_.speed_des_rad_s[i] = 0.0f;
    wheel_cmd_lcmt_.enable = 0;
    hopper_data_lcmt_.rm_online = 0;
    rm_esc_data_lcmt_.online_mask = 0;

}

void HopperHardware::_thread_lcmrec_run(){
    // Drain the ENTIRE receive queue every tick. handleTimeout() dispatches only
    // ONE message per call, so a single call per tick caps throughput at the task
    // rate and lets hopper_cmd_lcmt queue up behind other traffic (stale commands).
    // handleTimeout(0) returns 0 immediately when the queue is empty.
    while (Controller2Robot.handleTimeout(0) > 0) {
    }
}

void HopperHardware::_thread_lcmsen_run(){
    if(is_publish_lcm_data){
        _fill_in_motor_data_to_lcm();
        Robot2Controller.publish("hopper_data_lcmt", &hopper_data_lcmt_);
    }
}

void HopperHardware::_thread_imu_run(){
    if (imu_wrapper_ptr_ != nullptr && imu_wrapper_ptr_->hasImuData()) {
        imu_wrapper_ptr_->getImuData(imu_raw_data_);
        if(is_publish_lcm_data){
            _fill_in_imu_data_to_lcm();
            Robot2Controller.publish("hopper_imu_lcmt", &hopper_imu_lcmt_);
        }
    }
}

void HopperHardware::_thread_xbox_run(){

    xbox_controller_ptr_->processInput();
    // const auto& map = xbox_controller_ptr_->getMap();
    if(is_publish_lcm_data){
        _fill_in_gamepad_data_to_lcm();
        Robot2Controller.publish("gamepad_lcmt", &gamepad_cmd_lcmt_);
    }
}

void HopperHardware::imu_compensation(IG1ImuDataI* source, IG1ImuDataI* final)
{
    eigen_gyro_(0) = source->gyroIIBiasCalibrated.data[0] * static_cast<float>(M_PI) / 180.0f;
    eigen_gyro_(1) = source->gyroIIBiasCalibrated.data[1] * static_cast<float>(M_PI) / 180.0f;
    eigen_gyro_(2) = source->gyroIIBiasCalibrated.data[2] * static_cast<float>(M_PI) / 180.0f;

    eigen_acc_(0) = source->accCalibrated.data[0];
    eigen_acc_(1) = source->accCalibrated.data[1];
    eigen_acc_(2) = source->accCalibrated.data[2];

    eigen_quat_(0) = source->quaternion.data[0];
    eigen_quat_(1) = source->quaternion.data[1];
    eigen_quat_(2) = source->quaternion.data[2];
    eigen_quat_(3) = source->quaternion.data[3];

    eigen_gyro_rotated_ = rot_offset_ * eigen_gyro_;
    eigen_acc_rotated_ = rot_offset_ * eigen_acc_;
    eigen_quat_rotated_ = rotationMatrixToQuaternion(
        rot_offset_ * quaternionToRotationMatrix(eigen_quat_));
    eigen_rpy_ = quatToRPY(eigen_quat_rotated_);

    final->gyroIIBiasCalibrated.data[0] = eigen_gyro_rotated_(0);
    final->gyroIIBiasCalibrated.data[1] = eigen_gyro_rotated_(1);
    final->gyroIIBiasCalibrated.data[2] = eigen_gyro_rotated_(2);

    final->accCalibrated.data[0] = eigen_acc_rotated_(0);
    final->accCalibrated.data[1] = eigen_acc_rotated_(1);
    final->accCalibrated.data[2] = eigen_acc_rotated_(2);

    final->quaternion.data[0] = eigen_quat_rotated_(0);
    final->quaternion.data[1] = eigen_quat_rotated_(1);
    final->quaternion.data[2] = eigen_quat_rotated_(2);
    final->quaternion.data[3] = eigen_quat_rotated_(3);
    final->euler.data[0] = eigen_rpy_(0);
    final->euler.data[1] = eigen_rpy_(1);
    final->euler.data[2] = eigen_rpy_(2);
}

void HopperHardware::_fill_in_imu_data_to_lcm(){
    imu_compensation(&imu_raw_data_, &imu_raw_data_compensated_);

    for(int i = 0; i < 3; i++){
        hopper_imu_lcmt_.acc[i] = imu_raw_data_compensated_.accCalibrated.data[i];
        hopper_imu_lcmt_.gyro[i] = imu_raw_data_compensated_.gyroIIBiasCalibrated.data[i];
        hopper_imu_lcmt_.rpy[i] = imu_raw_data_compensated_.euler.data[i];
    }
    // order: w,x,y,z (same as CASE; quat from raw sensor, acc/gyro/rpy compensated)
    for(int i = 0; i < 4; i++){
        hopper_imu_lcmt_.quat[i] = imu_raw_data_.quaternion.data[i];
    }
}

void HopperHardware::_fill_in_motor_data_to_lcm(){

    const float offset = kAk60LcmQOffsetRad;
    // Copy motor state under lock to ensure a consistent snapshot.
    // qd on LCM = the AK60 CAN-REPORTED velocity (motor internal estimate).
    // 2026-07-07 user decision (reversed from 07-06): publish the CAN qd and
    // let ALL upper layers consume it; no q-differentiation on the PC side.
    // (motor_vel_diff is still computed in _store_motor_state for local debug.)
    // Same sign map as position: q_lcm = -motor_pos + offset => qd_lcm = -qd_can.
    float q_lcm[3] = {0.0f, 0.0f, 0.0f};
    float qd_lcm[3] = {0.0f, 0.0f, 0.0f};
    float tau_lcm[3] = {0.0f, 0.0f, 0.0f};
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int i = 0; i < 3; i++) {
            q_lcm[i] = -motor_pos[i] + offset;
            qd_lcm[i] = -motor_vel[i];
            tau_lcm[i] = -motor_tau[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        hopper_data_lcmt_.q[i] = q_lcm[i];
        hopper_data_lcmt_.qd[i] = qd_lcm[i];
        hopper_data_lcmt_.tauIq[i] = tau_lcm[i];
    }

    // ---- RM M2006/C610 state (relayed from px4_dds_bridge's rm_esc_data_lcmt) ----
    // rm_online reflects BOTH links: the PX4-side per-ESC freshness bits AND the
    // LCM relay freshness (0 if no rm_esc_data_lcmt for >200ms, e.g. Pixhawk off).
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        bool fresh = false;
        if (rm_data_seen_) {
            const float age = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - rm_data_rx_t_).count();
            fresh = (age < 0.2f);
        }
        for (int i = 0; i < 3; i++) {
            hopper_data_lcmt_.rm_q[i]  = rm_esc_data_lcmt_.shaft_angle_rad[i] - rm_q_offset_[i];
            hopper_data_lcmt_.rm_qd[i] = rm_esc_data_lcmt_.shaft_speed_rad_s[i];
            hopper_data_lcmt_.rm_iq[i] = rm_esc_data_lcmt_.current_raw[i] / 1000.0f;  // raw -> A
        }
        hopper_data_lcmt_.rm_online = fresh ? rm_esc_data_lcmt_.online_mask : 0;
    }
}
void HopperHardware::_fill_in_gamepad_data_to_lcm(){
    const auto& map = xbox_controller_ptr_->getMap();
    gamepad_cmd_lcmt_.a = map.a;
    gamepad_cmd_lcmt_.b = map.b;
    gamepad_cmd_lcmt_.x = map.x;
    gamepad_cmd_lcmt_.y = map.y;
    gamepad_cmd_lcmt_.leftBumper = map.lb;
    gamepad_cmd_lcmt_.rightBumper = map.rb;
    gamepad_cmd_lcmt_.thumbl = map.thumbl;
    gamepad_cmd_lcmt_.thumbr = map.thumbr;
    gamepad_cmd_lcmt_.home = map.home;
    gamepad_cmd_lcmt_.start = map.start;
    gamepad_cmd_lcmt_.select = map.select;
    gamepad_cmd_lcmt_.point = map.point;
    //axis scaled to [-1,1]
    gamepad_cmd_lcmt_.leftStickAnalog[0] = map.lx / 32767.0;
    gamepad_cmd_lcmt_.leftStickAnalog[1] = map.ly / 32767.0;
    gamepad_cmd_lcmt_.rightStickAnalog[0] = map.rx / 32767.0;
    gamepad_cmd_lcmt_.rightStickAnalog[1] = map.ry / 32767.0;
    gamepad_cmd_lcmt_.leftTriggerAnalog = map.lt / 255.0;
    gamepad_cmd_lcmt_.rightTriggerAnalog = map.rt / 255.0;

}

HopperHardware::~HopperHardware()
{
    task_manager_.stopAll();
    // Allow a short time for threads to exit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    delete ak60_controller_ptr_;
    delete wheel_controller_ptr_;   // sends zero speed + disable on close
    delete imu_wrapper_ptr_;
    delete xbox_controller_ptr_;
}
