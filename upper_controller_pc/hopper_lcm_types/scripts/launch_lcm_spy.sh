#!/bin/bash
set -e

# launch_lcm_spy.sh
# Sets up environment and launches lcm-spy with custom types (if available)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../" && pwd)"
cd "${ROOT_DIR}"

# Set LCM URL
export LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=255"

# Ensure types are built (optional, for decoding fields like q/qd/tau)
if [ ! -f "hopper_lcm_types/lcm_types/java/my_types.jar" ]; then
  echo "[launch_lcm_spy] Building LCM types (optional)..."
  (cd hopper_lcm_types/scripts && chmod +x make_types.sh && ./make_types.sh) || true
fi

MY_TYPES="hopper_lcm_types/lcm_types/java/my_types.jar"
LCM_JAR="hopper_lcm_types/lcm_types/java/lcm.jar"

if [ -f "${MY_TYPES}" ] && [ -f "${LCM_JAR}" ]; then
  echo "[launch_lcm_spy] Using Java types: ${MY_TYPES}"
  CLASSPATH="${MY_TYPES}:${LCM_JAR}" exec lcm-spy
elif [ -f "${MY_TYPES}" ]; then
  echo "[launch_lcm_spy] Using Java types (no lcm.jar found in repo): ${MY_TYPES}"
  CLASSPATH="${MY_TYPES}" exec lcm-spy
else
  echo "[launch_lcm_spy] WARNING: Java types not available. lcm-spy may not decode fields."
  exec lcm-spy
fi

