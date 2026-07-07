#!/bin/bash
set -e

CURRENT_FILE=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="$( cd "$( dirname "${CURRENT_FILE}" )" && pwd )"

OUT_DIR="${SCRIPT_DIR}/../lcm_types"
mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

rm -rf cpp python java lcmtypes
rm -f *.hpp *.py *.java *.class my_types.jar lcm.jar

if ! command -v lcm-gen >/dev/null 2>&1; then
    echo "ERROR: lcm-gen not found. Install LCM first."
    exit 1
fi

lcm-gen -x ${SCRIPT_DIR}/../*.lcm
mkdir -p cpp
mv -f *.hpp cpp/

# Optional: generate Python types (not required for C++ build)
if lcm-gen -p ${SCRIPT_DIR}/../*.lcm >/dev/null 2>&1; then
    mkdir -p python
    mv -f *.py python/
fi

# Optional: generate Java types for lcm-spy decoding
if command -v javac >/dev/null 2>&1 && command -v jar >/dev/null 2>&1; then
    LCM_JAR=""
    for p in /usr/local/share/java/lcm.jar /usr/share/java/lcm.jar; do
        if [ -f "$p" ]; then
            LCM_JAR="$p"
            break
        fi
    done

    if [ -n "$LCM_JAR" ]; then
        lcm-gen -j ${SCRIPT_DIR}/../*.lcm
        javac -cp "$LCM_JAR" lcmtypes/*.java
        jar cf my_types.jar lcmtypes/*.class
        mkdir -p java
        cp -f "$LCM_JAR" java/lcm.jar
        mv -f my_types.jar java/
    fi
fi

echo "LCM C++ headers: ${OUT_DIR}/cpp"