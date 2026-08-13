#!/bin/bash
# can_gsusb_setup.sh <usb-serial> <ifname> [bitrate]
#
# Companion to 99-canable.rules for CANable2 adapters running candleLight
# (gs_usb) firmware. Called from can0-gsusb.service / can1-gsusb.service each
# time the matching USB device (re)appears.
#
# Why a script instead of udev NAME=: the kernel hands out "can0"/"can1" in
# probe order (and Tegra's onboard mttcan grabs "can0" at boot before udev can
# rename it). A one-shot rename inside udev fails whenever the target name is
# transiently held by the other adapter. This script retries until the name
# frees up, then configures bitrate and brings the link up.
#
# 2026-08-12: replaces slcand entirely. gs_usb is a native kernel driver: on
# USB reset the netdev vanishes and reappears (no zombie slcand holding a
# deleted tty), udev re-triggers us, and the bus is back in <1s instead of the
# 10s watchdog path.

serial="$1"; target="$2"; bitrate="${3:-1000000}"
log() { logger -t can-gsusb-setup "$target: $*"; }

# Find the CAN netdev whose parent USB device has the wanted serial.
# gs_usb net device sysfs: /sys/class/net/<if>/device = USB interface,
# its parent = the USB device carrying the "serial" attribute.
find_if() {
    local n s
    for n in /sys/class/net/*; do
        s=$(cat "$n/device/../serial" 2>/dev/null) || continue
        if [ "$s" = "$serial" ]; then basename "$n"; return 0; fi
    done
    return 1
}

cur=""
for _ in $(seq 1 25); do
    cur=$(find_if) || { sleep 0.2; continue; }
    [ "$cur" = "$target" ] && break
    # Target name held by ANOTHER netdev (kernel probe-order collision, e.g.
    # both adapters swapped, or mttcan). Move the squatter to a temp name so
    # both setup services can't deadlock waiting on each other; the squatter's
    # own service finds it again by serial and gives it its proper name.
    if [ -d "/sys/class/net/$target" ]; then
        tmp="${target}sq$$"
        ip link set "$target" down 2>/dev/null
        if ip link set "$target" name "$tmp" 2>/dev/null; then
            log "moved squatter $target -> $tmp"
        fi
    fi
    ip link set "$cur" down 2>/dev/null
    ip link set "$cur" name "$target" 2>/dev/null && { cur="$target"; break; }
    sleep 0.2
done

if [ "$cur" != "$target" ]; then
    log "FAILED: netdev for usb serial $serial not renamed (last seen: '$cur')"
    exit 1
fi

ip link set "$target" down 2>/dev/null
ip link set "$target" type can bitrate "$bitrate" || { log "FAILED to set bitrate"; exit 1; }
ip link set "$target" txqueuelen 65536
ip link set "$target" up || { log "FAILED to bring up"; exit 1; }
log "up @ ${bitrate}bps (usb serial $serial)"
