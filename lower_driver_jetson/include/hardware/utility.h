#ifndef UTILITY_H
#define UTILITY_H

// ===== Leg motor protocol =====
// 1 = DaMiao DM-J4310 (MIT-compatible CAN). Ranges below MUST equal the
//     motor's own registers (read from the actual units, 2026-07-18):
//     PMAX=12.5, VMAX=45, TMAX=10, CTRL_MODE=1 (MIT).
//     P/V happen to match the old AK60 constants; only torque differs.
//     Feedback frame data[0] = (status<<4)|motor_id, status: 0=disabled,
//     1=enabled, >=0x8 fault (overvolt/undervolt/overcurr/overtemp/...).
//     Feedback arbitration ID = Master ID (0x10|motor_id), payload id nibble
//     is what we key on.
// 0 = legacy CubeMars AK60-6 (plain id in data[0], torque +/-15).
#define MOTOR_PROTOCOL_DAMIAO 1

#define P_MIN -12.5f
#define P_MAX 12.5f
#define V_MIN -45.0f
#define V_MAX 45.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#if MOTOR_PROTOCOL_DAMIAO
#define T_MIN -10.0f
#define T_MAX 10.0f
#else
#define T_MIN -15.0f
#define T_MAX 15.0f
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <fcntl.h>
#include <sys/time.h>
#include <chrono>

class canChannel
{
public:
	canChannel();

    // time
    //static std::chrono::high_resolution_clock::time_point start_time;
    //static std::chrono::high_resolution_clock::time_point end_time;
    //auto start_time;
    //auto end_time;
    uint64_t execution_timer(int state);
    //void start_execution(void);
    //void end_execution(void);
    //uint64_t get_execution_time(void);

	void comm_init(uint8_t portNum, uint8_t device_num); // FIRST FUNCTION TO CALL
	void comm_close(void); // LAST FUNCTION TO CALL
	int fd;	// File Descriptor

    // send custom can frame
    void send_EnterMotorMode(int device_id);
    void send_ExitMotorMode(int device_id);
    void send_SetZero(int device_id);
    void send_dummy(int device_id);
    void send_motor_cmd(int device_id, const float * motor_cmd);

    // motor state
    float motor_state[255][3];
    // receive can frame, custom processing
    void receive_process_frame();

private:
	// Private variables
    uint8_t channelNum;
    uint8_t numOfDevice;
    struct can_frame tx_frame;    
    struct can_frame rx_frame;
    float zero_cmd[5];

    //Private functions
    void send_frame(int device_id);
    void pack_cmd(const float * motor_cmd);
    void unpack_reply();
    int float_to_uint(float x, float x_min, float x_max, unsigned int bits);
    float uint_to_float(int x_int, float x_min, float x_max, int bits);

};
/*
float fmaxf(float x, float y);
float fminf(float x, float y);
float fmaxf3(float x, float y, float z);
float fminf3(float x, float y, float z);
float roundf(float x);
void limit_norm(float *x, float *y, float limit);
void limit(float *x, float min, float max);
int float_to_uint(float x, float x_min, float x_max, int bits);
float uint_to_float(int x_int, float x_min, float x_max, int bits);
*/
//newly added functions
/*
void pack_cmd(can_frame * msg, float * motor_cmd);
void SetZero(can_frame * msg);
void EnterMotorMode(can_frame * msg);
void ExitMotorMode(can_frame * msg);
void unpack_reply(can_frame msg, float * motor_state);
*/

#endif