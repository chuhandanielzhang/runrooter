#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: config_network_lcm.sh -I [interface]"
    echo "or config_network_lcm.sh [computer]"
    echo "interface: network interface to configure"
    echo "computer: use stored interface for computer"
    echo "current computers:"
    echo " name       interface    description"
    echo " ----       --------     -----------"
    echo " opi          eth0            opi network"
fi

if [ "$1" == "-I" ]; then
    sudo ifconfig $2 multicast
    sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev $2
fi


if [ "$1" == "opi" ]; then
    sudo ifconfig eth0 multicast
    sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0
fi
