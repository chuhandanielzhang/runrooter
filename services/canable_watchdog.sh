#!/bin/bash
# canable-watchdog (gs_usb edition, 2026-08-12): recovery for the candleLight
# CAN adapters and the driver's stale sockets.
#
# History: the slcan edition babysat slcand zombies (deleted ttys, ldisc
# detach...). With candleLight/gs_usb firmware none of that exists -- the
# kernel driver removes the netdev on USB loss and udev + can*-gsusb.service
# recreate/configure it on return. What is STILL needed:
#
#   d) STALE DRIVER SOCKET: hopper_driver binds CAN_RAW sockets to an ifindex
#      ONCE at init. Any netdev re-creation (USB reset, hub reset, replug)
#      yields a NEW ifindex; the old socket is deaf/ENODEV forever. So every
#      time can0/can1's ifindex changes, restart hopper-driver.
#   e) WEDGED ADAPTER (2026-07-06): adapter alive on USB, TX counting, RX
#      pinned at zero -> internal CAN core stuck; only a full USB
#      re-enumeration resets it. RX also freezes when the motor battery is
#      simply OFF, so this is rate-limited and harmless in that case.
#      can0 ONLY (wheel battery is routinely off while the legs hop).
#   c) MISSED SETUP: adapter enumerated but canX missing or down (missed udev
#      event, setup race lost) -> re-run the setup script.
#
# Serials are the candleLight USB serials (filled in at deploy time).
LEGS_SERIAL="004D00663945501620303651"
WHEEL_SERIAL="0067005F3945501620303651"

PERIOD=2
RX_STALL_TRIPS=5        # e) consecutive checks with TX alive, RX frozen
REENUM_COOLDOWN=30      # e) min seconds between USB re-enumerations
idx_prev=""
rx_prev=""; tx_prev=""; stall=0; last_reenum=0

note() { logger -t canable-watchdog "$*"; }

usb_present() {  # $1=serial -> 0 if a USB device with that serial exists
    local d
    for d in /sys/bus/usb/devices/*/serial; do
        [ "$(cat "$d" 2>/dev/null)" = "$1" ] && return 0
    done
    return 1
}

while sleep "$PERIOD"; do
    # c) adapter on USB but netdev missing/down -> re-run setup
    if usb_present "$LEGS_SERIAL"; then
        state=$(cat /sys/class/net/can0/operstate 2>/dev/null)
        if [ -z "$state" ] || [ "$state" = "down" ]; then
            note "legs adapter present but can0 ${state:-missing} -> re-run setup"
            /usr/local/bin/can_gsusb_setup.sh "$LEGS_SERIAL" can0 1000000
        fi
    fi
    if usb_present "$WHEEL_SERIAL"; then
        state=$(cat /sys/class/net/can1/operstate 2>/dev/null)
        if [ -z "$state" ] || [ "$state" = "down" ]; then
            note "wheel adapter present but can1 ${state:-missing} -> re-run setup"
            /usr/local/bin/can_gsusb_setup.sh "$WHEEL_SERIAL" can1 1000000
        fi
    fi

    # e) wedged adapter (can0 only): TX counts up, RX frozen -> USB re-enum
    rx=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null)
    tx=$(cat /sys/class/net/can0/statistics/tx_packets 2>/dev/null)
    if [ -n "$rx" ] && [ -n "$rx_prev" ]; then
        if [ "$rx" = "$rx_prev" ] && [ "$tx" != "$tx_prev" ]; then
            stall=$((stall + 1))
        else
            stall=0
        fi
        now=$(date +%s)
        if [ "$stall" -ge "$RX_STALL_TRIPS" ] && \
           [ $((now - last_reenum)) -ge "$REENUM_COOLDOWN" ]; then
            note "can0 TX alive but RX frozen ${stall}x${PERIOD}s -> USB re-enumerate legs adapter"
            last_reenum=$now; stall=0
            usbdev=$(readlink -f /sys/class/net/can0/device/../.. 2>/dev/null)
            if [ -n "$usbdev" ] && [ -e "$usbdev/serial" ]; then
                systemctl stop hopper-driver.service
                echo "$(basename "$usbdev")" > /sys/bus/usb/drivers/usb/unbind 2>/dev/null
                sleep 2
                echo "$(basename "$usbdev")" > /sys/bus/usb/drivers/usb/bind 2>/dev/null
                # udev + can0-gsusb.service reconfigure the fresh netdev;
                # (d) below sees the new ifindex and restarts the driver.
            else
                note "cannot resolve can0 USB device -> re-run setup only"
                /usr/local/bin/can_gsusb_setup.sh "$LEGS_SERIAL" can0 1000000
            fi
        fi
    fi
    rx_prev="$rx"; tx_prev="$tx"

    # d) ifindex changed (netdev re-created) OR newly appeared (hot-plug after
    #    driver start) -> hopper-driver must rebind. "none" is a real state so
    #    the none->N transition (adapter plugged in later) also triggers;
    #    N->none (adapter gone) does not -- nothing to rebind until it returns.
    #    can0 (legs) ONLY: since 2026-08-13 RmWheelController self-heals can1
    #    in-process (lazy rebind + ENODEV detect), so a wheel-adapter replug
    #    no longer justifies killing the leg driver mid-operation.
    idx=$(cat /sys/class/net/can0/ifindex 2>/dev/null || echo none)
    changed=""
    [ -n "$idx_prev" ]  && [ "$idx" != "none" ]  && [ "$idx" != "$idx_prev" ]   && changed="can0"
    if [ -n "$changed" ]; then
        sleep 1
        if systemctl is-active -q hopper-driver.service; then
            note "$changed re-created (new ifindex) -> restart hopper-driver (stale CAN socket)"
            systemctl restart hopper-driver.service
        else
            note "$changed re-created -> start hopper-driver"
            systemctl start hopper-driver.service
        fi
    fi
    idx_prev="$idx"
done
