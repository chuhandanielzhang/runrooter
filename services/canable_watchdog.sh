#!/bin/bash
# canable-watchdog: belt-and-braces recovery for the slcan CAN bridge.
#
# Layer 1 (udev + BindsTo in canable.service) already handles clean USB
# unplug/replug. This watchdog covers what udev can NOT see:
#   a) zombie slcand: tty re-enumerated but slcand still holds the DELETED old
#      tty -> can0 stays "up" carrying zero frames forever
#   b) slcand alive but the can0 netdev is gone (ldisc detached)
#   c) missed udev add event: /dev/canable exists but the bridge is down
#   d) hopper_driver's CAN_RAW socket is bound to the OLD can0 ifindex after any
#      bridge restart -> the driver would stay deaf forever, so every canable
#      recovery must be followed by a hopper-driver restart
#   e) WEDGED ADAPTER (2026-07-06): slcand alive, can0 up, TX counting, but RX
#      pinned at zero -> the CANable's internal CAN core is stuck. A slcand
#      restart does NOT fix this; only a full USB re-enumeration (unbind/bind)
#      resets the adapter. NOTE: RX also stays zero when the motor battery is
#      simply OFF, so this recovery is rate-limited (once per REENUM_COOLDOWN)
#      and harmless in that case.
#
# Runs as root from canable-watchdog.service. Logs to journal via `logger`.

PERIOD=2
RX_STALL_TRIPS=5        # e) consecutive checks (PERIOD each) with TX alive, RX frozen
REENUM_COOLDOWN=30      # e) min seconds between USB re-enumerations
prev_ts=""
rx_prev=""; tx_prev=""; stall=0; last_reenum=0

note() { logger -t canable-watchdog "$*"; }

while sleep "$PERIOD"; do
    # a) slcand holding a deleted tty (USB re-enumerated under it)
    pid=$(systemctl show -p MainPID --value canable.service 2>/dev/null)
    if [ -n "$pid" ] && [ "$pid" != "0" ] && \
       ls -l "/proc/$pid/fd" 2>/dev/null | grep -q 'deleted'; then
        note "slcand pid=$pid holds a deleted tty -> restart canable"
        systemctl restart canable.service
    fi

    # b) service active but can0 netdev missing
    if systemctl is-active -q canable.service && \
       ! ip link show can0 >/dev/null 2>&1; then
        note "canable active but can0 missing -> restart canable"
        systemctl restart canable.service
    fi

    # c) adapter present but bridge down (missed udev event / start-limit hiccup)
    if [ -e /dev/canable ] && ! systemctl is-active -q canable.service; then
        note "/dev/canable present but bridge down -> start canable"
        systemctl start canable.service
    fi

    # e) wedged adapter: driver is transmitting but nothing ever comes back.
    #    (motor battery off looks identical -> cooldown keeps this harmless)
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

    # d) canable (re)started since last check -> hopper-driver must rebind can0.
    #    Requires=canable.service already STOPPED the driver when the bridge
    #    dropped; bring it back (or bounce it if it somehow survived).
    ts=$(systemctl show -p ActiveEnterTimestampMonotonic --value canable.service 2>/dev/null)
    if [ -n "$prev_ts" ] && [ -n "$ts" ] && [ "$ts" != "$prev_ts" ] && \
       systemctl is-active -q canable.service; then
        sleep 1
        if systemctl is-active -q hopper-driver.service; then
            note "canable recovered -> restart hopper-driver (stale can0 socket)"
            systemctl restart hopper-driver.service
        else
            note "canable recovered -> start hopper-driver"
            systemctl start hopper-driver.service
        fi
    fi
    [ -n "$ts" ] && prev_ts="$ts"
done
