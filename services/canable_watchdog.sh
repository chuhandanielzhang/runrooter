#!/bin/bash
# canable-watchdog: belt-and-braces recovery for the slcan CAN bridges.
#
# Covers BOTH adapters since 2026-08-08:
#   canable  / can0 / /dev/canable   (legs AK60/DM4310)
#   canable2 / can1 / /dev/canable2  (RM M2006 wheel bus, optional hot-plug)
#
# Layer 1 (udev + BindsTo in canable*.service) already handles clean USB
# unplug/replug. This watchdog covers what udev can NOT see:
#   a) zombie slcand: tty re-enumerated but slcand still holds the DELETED old
#      tty -> canX stays "up" carrying zero frames forever
#   b) slcand alive but the canX netdev is gone (ldisc detached)
#   c) missed udev add event: /dev/canableX exists but the bridge is down
#   d) hopper_driver's CAN_RAW sockets are bound to the OLD ifindex after ANY
#      bridge restart -> that bus stays deaf forever, so every canable OR
#      canable2 recovery must be followed by a hopper-driver restart.
#      (2026-08-08: canable2 re-enumerated 8 min after boot; can1 looked UP,
#      ESC feedback flowed, but the driver's stale socket write()s ENODEV ->
#      "wheels connected but never move" until a manual driver restart.)
#   e) WEDGED ADAPTER (2026-07-06): slcand alive, can0 up, TX counting, but RX
#      pinned at zero -> the CANable's internal CAN core is stuck. A slcand
#      restart does NOT fix this; only a full USB re-enumeration (unbind/bind)
#      resets the adapter. NOTE: RX also stays zero when the motor battery is
#      simply OFF, so this recovery is rate-limited (once per REENUM_COOLDOWN)
#      and harmless in that case.  can0 ONLY: on can1 the wheel battery is
#      routinely off while the legs hop, and the recovery path stops
#      hopper-driver -- re-enumerating can1 on a mere battery-off would bounce
#      the LEG driver mid-run.
#
# Runs as root from canable-watchdog.service. Logs to journal via `logger`.

PERIOD=2
RX_STALL_TRIPS=5        # e) consecutive checks (PERIOD each) with TX alive, RX frozen
REENUM_COOLDOWN=30      # e) min seconds between USB re-enumerations
prev_ts=""; prev_ts2=""
rx_prev=""; tx_prev=""; stall=0; last_reenum=0

note() { logger -t canable-watchdog "$*"; }

# Checks a)-c) for one bridge: $1=service $2=can netdev $3=stable dev node.
check_bridge() {
    local svc="$1" canif="$2" dev="$3" pid
    # a) slcand holding a deleted tty (USB re-enumerated under it)
    pid=$(systemctl show -p MainPID --value "$svc" 2>/dev/null)
    if [ -n "$pid" ] && [ "$pid" != "0" ] && \
       ls -l "/proc/$pid/fd" 2>/dev/null | grep -q 'deleted'; then
        note "slcand ($svc) pid=$pid holds a deleted tty -> restart $svc"
        systemctl restart "$svc"
    fi
    # b) service active but the CAN netdev is missing
    if systemctl is-active -q "$svc" && \
       ! ip link show "$canif" >/dev/null 2>&1; then
        note "$svc active but $canif missing -> restart $svc"
        systemctl restart "$svc"
    fi
    # c) adapter present but bridge down (missed udev event / start-limit)
    if [ -e "$dev" ] && ! systemctl is-active -q "$svc"; then
        note "$dev present but $svc down -> start $svc"
        systemctl start "$svc"
    fi
}

while sleep "$PERIOD"; do
    check_bridge canable.service  can0 /dev/canable
    check_bridge canable2.service can1 /dev/canable2

    # e) wedged adapter (can0 only, see header): driver transmitting but
    #    nothing ever comes back. (motor battery off looks identical ->
    #    cooldown keeps this harmless)
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
            note "can0 TX alive but RX frozen ${stall}x${PERIOD}s -> USB re-enumerate canable"
            last_reenum=$now; stall=0
            usbdev=$(readlink -f /sys/class/tty/"$(basename "$(readlink /dev/canable)")"/device/../.. 2>/dev/null)
            if [ -n "$usbdev" ] && [ -e "$usbdev" ]; then
                systemctl stop hopper-driver.service canable.service
                echo "$(basename "$usbdev")" > /sys/bus/usb/drivers/usb/unbind 2>/dev/null
                sleep 2
                echo "$(basename "$usbdev")" > /sys/bus/usb/drivers/usb/bind 2>/dev/null
                sleep 3
                systemctl start canable.service
                # (d) below sees the fresh ActiveEnterTimestamp and restarts the driver)
            else
                note "cannot resolve canable USB path -> plain canable restart"
                systemctl restart canable.service
            fi
        fi
    fi
    rx_prev="$rx"; tx_prev="$tx"

    # d) EITHER bridge (re)started since last check -> hopper-driver must
    #    rebind its CAN_RAW sockets (bound to an ifindex ONCE at init; a
    #    recreated can0/can1 has a NEW ifindex and the old socket is deaf).
    ts=$(systemctl show -p ActiveEnterTimestampMonotonic --value canable.service 2>/dev/null)
    ts2=$(systemctl show -p ActiveEnterTimestampMonotonic --value canable2.service 2>/dev/null)
    changed=""
    if [ -n "$prev_ts" ] && [ -n "$ts" ] && [ "$ts" != "$prev_ts" ] && \
       systemctl is-active -q canable.service; then
        changed="canable"
    fi
    if [ -n "$prev_ts2" ] && [ -n "$ts2" ] && [ "$ts2" != "$prev_ts2" ] && \
       systemctl is-active -q canable2.service; then
        changed="${changed:+${changed}+}canable2"
    fi
    if [ -n "$changed" ]; then
        sleep 1
        if systemctl is-active -q hopper-driver.service; then
            note "$changed recovered -> restart hopper-driver (stale CAN socket)"
            systemctl restart hopper-driver.service
        else
            note "$changed recovered -> start hopper-driver"
            systemctl start hopper-driver.service
        fi
    fi
    [ -n "$ts" ] && prev_ts="$ts"
    [ -n "$ts2" ] && prev_ts2="$ts2"
done
