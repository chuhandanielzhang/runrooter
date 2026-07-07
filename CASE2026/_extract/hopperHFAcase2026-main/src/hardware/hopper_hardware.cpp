#include "hopper_hardware.hpp"

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
// Identity mapping (DO NOT remap motors here):
//   LCM joint idx 0 -> motor_id 0
//   LCM joint idx 1 -> motor_id 1
//   LCM joint idx 2 -> motor_id 2
//
// If the delta kinematics motor numbering differs from your physical motors, fix it in
// `hopper_controller/forward_kinematics.py` (motor index permutation), NOT here.
static constexpr int kDeltaAk60MotorIdFromJointIdx[3] = {0, 1, 2};

HopperHardware::HopperHardware(bool is_publish_lcm_data):
        Controller2Robot("udpm://239.255.76.67:7667?ttl=255"),
        Robot2Controller("udpm://239.255.76.67:7667?ttl=255"),
    lcmreceiveTask(&task_manager_, 0.00001, "lcmrecv_task", &HopperHardware::_thread_lcmrec_run, this),  // 1000Hz receive
    lcmsendTask   (&task_manager_, 0.001, "lcmsend_task", &HopperHardware::_thread_lcmsen_run, this),  // 500Hz send
    gamepadTask(&task_manager_, 0.005, "gamepad_task", &HopperHardware::_thread_xbox_run, this),       // 200Hz
        motor_pos(3, 0.0),
        motor_vel(3, 0.0),
        motor_tau(3, 0.0)
{
    this->is_publish_lcm_data = is_publish_lcm_data;
    set_value_zero(); // clear all the value
    // initialize the leg motor controller (3x AK60 over SocketCAN / can0)
    ak60_controller_ptr_ = new AK60Controller();
    std::cout << "Driver: LEGS-ONLY (3x AK60 via SocketCAN can0)" << std::endl;
    std::cout << "IMU + propellers are handled by the Pixhawk (px4_bridge), not this driver." << std::endl;
    // initialize the xbox controller
    xbox_controller_ptr_ = new XboxController();

    start_threads();

}
XboxController::XboxMap HopperHardware::get_xbox_map(){
    return xbox_controller_ptr_->getMap();
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
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];
            motor_tau[j] = temp_tau[j];
        }
    }

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
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];
            motor_tau[j] = temp_tau[j];
        }
    }

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
        const float offset = -1.047f;
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
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];
            motor_tau[j] = temp_tau[j];
        }
    }
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
        const float offset = -1.047f;
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
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];
            motor_tau[j] = temp_tau[j];
        }
    }
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
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int j = 0; j < 3; j++) {
            motor_pos[j] = temp_pos[j];
            motor_vel[j] = temp_vel[j];
            motor_tau[j] = temp_tau[j];
        }
    }

    step_counter++;
}

    void HopperHardware::start_threads(){
    lcmsendTask.start();
    gamepadTask.start();
    Controller2Robot.subscribe("hopper_cmd_lcmt", &HopperHardware::handleController2RobotLCM, this);
    Controller2Robot.subscribe("motor_pwm_lcmt", &HopperHardware::handleMotorPwmLCM, this);
    Controller2Robot.subscribe("gamepad_lcmt", &HopperHardware::handleGamepadLCM, this);
    lcmreceiveTask.start();
}

void HopperHardware::step_with_set_zero_mode(){
    std::cout<<"setting value zero, please hold still and wait..."<<std::endl;
    ak60_controller_ptr_->disableMotors();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ak60_controller_ptr_->setZero();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout<<"set zero done"<<std::endl;
    step_counter = 0;
}
void HopperHardware::handleController2RobotLCM(const lcm::ReceiveBuffer* rbuf,
                                      const std::string& chan,
                                      const hopper_cmd_lcmt* msg){
                                    // std::cout<<"111";
    (void)rbuf;
    (void)chan;
    std::lock_guard<std::mutex> lock(lcm_cmd_mutex);
    memcpy(&hopper_cmd_lcmt_, msg, sizeof(hopper_cmd_lcmt));
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
    }

    // Reset qd-from-position estimator state (published in hopper_data_lcmt_.qd)
    qd_est_inited_ = false;
    qd_last_ts_inited_ = false;
    qd_lcm_ema_ = {{0.0f, 0.0f, 0.0f}};
}

void HopperHardware::_thread_lcmrec_run(){
    // std::cout<<"222";
    Controller2Robot.handleTimeout(10);
}

void HopperHardware::_thread_lcmsen_run(){
    if(is_publish_lcm_data){
        _fill_in_motor_data_to_lcm();
        Robot2Controller.publish("hopper_data_lcmt", &hopper_data_lcmt_);
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

void HopperHardware::_fill_in_motor_data_to_lcm(){

    const float offset = -1.047f;
    // Copy motor state under lock to ensure a consistent snapshot
    float q_lcm[3] = {0.0f, 0.0f, 0.0f};
    float tau_lcm[3] = {0.0f, 0.0f, 0.0f};
    {
        std::lock_guard<std::mutex> lock(motor_state_mutex);
        for (int i = 0; i < 3; i++) {
            q_lcm[i] = -motor_pos[i] + offset;
            tau_lcm[i] = -motor_tau[i];
        }
    }

    // User request: publish hopper_data_lcmt.qd computed ONLY from position (q) using EWMA.
    // qd_k = a*qd_{k-1} + (1-a)*(q_k - q_{k-1})/dt
    constexpr float a = 0.4f;        // forgetting factor (user requested)
    constexpr float dt_min = 5e-4f;  // 0.5 ms
    constexpr float dt_max = 5e-2f;  // 50 ms
    constexpr float v_max = 45.0f;   // rad/s (match AK60 clamp)

    const auto now = std::chrono::steady_clock::now();
    float dt = 0.002f;  // fallback (500Hz)
    if (qd_last_ts_inited_) {
        dt = std::chrono::duration<float>(now - qd_last_ts_).count();
        if (dt < dt_min) dt = dt_min;
        if (dt > dt_max) dt = dt_max;
    } else {
        qd_last_ts_inited_ = true;
    }
    qd_last_ts_ = now;

    if (!qd_est_inited_) {
        for (int i = 0; i < 3; i++) {
            q_lcm_prev_[i] = q_lcm[i];
            qd_lcm_ema_[i] = 0.0f;
        }
        qd_est_inited_ = true;
    } else {
        for (int i = 0; i < 3; i++) {
            float raw = (q_lcm[i] - q_lcm_prev_[i]) / dt;
            if (raw > v_max) raw = v_max;
            if (raw < -v_max) raw = -v_max;
            qd_lcm_ema_[i] = a * qd_lcm_ema_[i] + (1.0f - a) * raw;
            q_lcm_prev_[i] = q_lcm[i];
        }
    }

    for (int i = 0; i < 3; i++) {
        hopper_data_lcmt_.q[i] = q_lcm[i];
        hopper_data_lcmt_.qd[i] = qd_lcm_ema_[i];
        hopper_data_lcmt_.tauIq[i] = tau_lcm[i];
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
    delete xbox_controller_ptr_;
}
