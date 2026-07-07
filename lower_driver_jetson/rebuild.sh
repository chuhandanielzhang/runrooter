#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "hopper_lcm_types/lcm_types/cpp/gamepad_lcmt.hpp" ]; then
    if ! command -v lcm-gen >/dev/null 2>&1; then
        echo "ERROR: missing LCM types and lcm-gen not found."
        echo "Install LCM first:  sudo apt-get install liblcm-dev"
        exit 1
    fi
    echo "Generating LCM types..."
    (cd hopper_lcm_types/scripts && chmod +x make_types.sh && ./make_types.sh)
fi

rm -rf build
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
echo "OK: $(pwd)/hopper_driver"
