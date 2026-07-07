#include "hopper_hardware.hpp"

HopperHardware::HopperHardware(bool is_publish_lcm_data):
        Controller2Robot("udpm://239.255.76.67:7667?ttl=255"),
        Robot2Controller("udpm://239.255.76.67:7667?ttl=255"),
        lcmreceiveTask(&task_manager_, 0.001, "lcmrecv_task", &HopperHardware::_thread_lcmrec_run, this),
        lcmsendTask   (&task_manager_, 0.001, "lcmsend_task", &HopperHardware::_thread_lcmsen_run, this),
        imuTask    (&task_manager_, 0.001, "imu_task", &HopperHardware::_thread_imu_run, this),
        gamepadTask(&task_manager_, 0.005, "gamepad_task", &HopperHardware::_thread_xbox_run, this),
        motor_pos(6, 0.0),
        motor_vel(6, 0.0),
        motor_tau(6, 0.0)
{
    this->is_publish_lcm_data = is_publish_lcm_data;
    set_value_zero(); // clear all the value
    // initialize the motor controller
    ak60_controller_ptr_ = new AK60Controller();
    rot_offset = coordinateRotation(CoordinateAxis::Z, 0.0f)*coordinateRotation(CoordinateAxis::Y,0.0f)*coordinateRotation(CoordinateAxis::X,0.0f);
    // initialize the imu
    imu_wrapper_ptr = new ImuWrapper();
    imu_wrapper_ptr->connect();
    // initialize the xbox controller
    xbox_controller_ptr_ = new XboxController();

    start_threads();
    
}
XboxController::XboxMap HopperHardware::get_xbox_map(){
    return xbox_controller_ptr_->getMap();
}
void HopperHardware::step_with_only_receiving(){
    if(step_counter== 0)
    {
        ak60_controller_ptr_->enableMotors();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for(int i = 0; i < 6; i++){
        ak60_controller_ptr_->setMotorParams(i, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();
    for (int i=0;i<6;i++)
    {
        ak60_controller_ptr_->getMotorState(i,motor_pos[i], motor_vel[i], motor_tau[i]);
    }
    step_counter++;
}

void HopperHardware::step_with_damping(){
    if(step_counter== 0)
    {
        ak60_controller_ptr_->enableMotors();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for(int i = 0; i < 6; i++){
        ak60_controller_ptr_->setMotorParams(i, 0.0, 0.0, 0.0, 0.0, 0.8);
    }
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();
    for (int i=0;i<6;i++)
    {
        ak60_controller_ptr_->getMotorState(i,motor_pos[i], motor_vel[i], motor_tau[i]);
    }
    step_counter++;
}

void HopperHardware::step_with_pd_control(){
    step_counter++;
    lcm_cmd_mutex.lock();
    for(int i = 0; i < 2; i++){
        ak60_controller_ptr_->setMotorParams(i, hopper_cmd_lcmt_.q_des[i], 
                                                hopper_cmd_lcmt_.qd_des[i], 
                                                hopper_cmd_lcmt_.tau_ff[i], 
                                                hopper_cmd_lcmt_.kp_joint[i], 
                                                hopper_cmd_lcmt_.kd_joint[i]);
    }
    ak60_controller_ptr_->setMotorParams(3, hopper_cmd_lcmt_.q_des[2], 
                                            hopper_cmd_lcmt_.qd_des[2], 
                                            hopper_cmd_lcmt_.tau_ff[2], 
                                            hopper_cmd_lcmt_.kp_joint[2], 
                                            hopper_cmd_lcmt_.kd_joint[2]);
    lcm_cmd_mutex.unlock();
    ak60_controller_ptr_->sendMotorCommands();
    ak60_controller_ptr_->updateMotorStates();
    for (int i=0;i<6;i++)
    {
        ak60_controller_ptr_->getMotorState(i,motor_pos[i], motor_vel[i], motor_tau[i]);
    }
}

    void HopperHardware::start_threads(){
    lcmsendTask.start();
    imuTask.start();
    gamepadTask.start();
    Controller2Robot.subscribe("hopper_cmd_lcmt", &HopperHardware::handleController2RobotLCM, this);
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
    (void)rbuf;
    (void)chan;
    memcpy(&hopper_cmd_lcmt_, msg, sizeof(hopper_cmd_lcmt));
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

}

void HopperHardware::_thread_lcmrec_run(){
    Controller2Robot.handle();
}

void HopperHardware::_thread_lcmsen_run(){
    if(is_publish_lcm_data){
        _fill_in_motor_data_to_lcm();
        Robot2Controller.publish("hopper_data_lcmt", &hopper_data_lcmt_);
    }
}

void HopperHardware::_thread_imu_run(){
    if (imu_wrapper_ptr->hasImuData()) {
        imu_wrapper_ptr->getImuData(imu_raw_data);
    }
    if(is_publish_lcm_data){
        _fill_in_imu_data_to_lcm();
        Robot2Controller.publish("hopper_imu_lcmt", &hopper_imu_lcmt_);
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
    for(int i = 0; i < 2; i++){
        hopper_data_lcmt_.q[i] = motor_pos[i];
        hopper_data_lcmt_.qd[i] = motor_vel[i];
        hopper_data_lcmt_.tauIq[i] = motor_tau[i];
    }
        hopper_data_lcmt_.q[2] = motor_pos[3];
        hopper_data_lcmt_.qd[2] = motor_vel[3];
        hopper_data_lcmt_.tauIq[2] = motor_tau[3];
}
 void HopperHardware::imu_compensation(IG1ImuDataI* source, IG1ImuDataI* final)
{
    eigen_gyro(0) = source->gyroIIBiasCalibrated.data[0]*M_PI/180;
    eigen_gyro(1) = source->gyroIIBiasCalibrated.data[1]*M_PI/180;
    eigen_gyro(2) = source->gyroIIBiasCalibrated.data[2]*M_PI/180;

    eigen_acc(0) = source->accCalibrated.data[0];
    eigen_acc(1) = source->accCalibrated.data[1];
    eigen_acc(2) = source->accCalibrated.data[2];

    eigen_quat(0) = source->quaternion.data[0];
    eigen_quat(1) = source->quaternion.data[1];
    eigen_quat(2) = source->quaternion.data[2];
    eigen_quat(3) = source->quaternion.data[3];

    //rotate by eigen calculation.
    eigen_gyro_rotated = rot_offset*eigen_gyro;
    eigen_acc_rotated = rot_offset*eigen_acc;
    eigen_quat_rotated = rotationMatrixToQuaternion(rot_offset*quaternionToRotationMatrix(eigen_quat));
    eigen_rpy = quatToRPY(eigen_quat_rotated);
    // write
    final->gyroIIBiasCalibrated.data[0] = eigen_gyro_rotated(0);
    final->gyroIIBiasCalibrated.data[1] = eigen_gyro_rotated(1);
    final->gyroIIBiasCalibrated.data[2] = eigen_gyro_rotated(2);

    final->accCalibrated.data[0] = eigen_acc_rotated(0);
    final->accCalibrated.data[1] = eigen_acc_rotated(1);
    final->accCalibrated.data[2] = eigen_acc_rotated(2);

    final->quaternion.data[0] = eigen_quat_rotated(0);
    final->quaternion.data[1] = eigen_quat_rotated(1);
    final->quaternion.data[2] = eigen_quat_rotated(2);
    final->quaternion.data[3] = eigen_quat_rotated(3);
    final->euler.data[0] = eigen_rpy(0);
    final->euler.data[1] = eigen_rpy(1);
    final->euler.data[2] = eigen_rpy(2);
}

void HopperHardware::_fill_in_imu_data_to_lcm(){
    imu_compensation(&imu_raw_data, &imu_raw_data_compensated);
    
    for(int i = 0; i < 3; i++){
        hopper_imu_lcmt_.acc[i] = imu_raw_data_compensated.accCalibrated.data[i];
        hopper_imu_lcmt_.gyro[i] = imu_raw_data_compensated.gyroIIBiasCalibrated.data[i];
        hopper_imu_lcmt_.rpy[i] = imu_raw_data_compensated.euler.data[i];
    }
    // order: w,x,y,z
    for(int i = 0; i < 4; i++){
        hopper_imu_lcmt_.quat[i] = imu_raw_data.quaternion.data[i];
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
    delete ak60_controller_ptr_;
    delete imu_wrapper_ptr;
    delete xbox_controller_ptr_;
}
