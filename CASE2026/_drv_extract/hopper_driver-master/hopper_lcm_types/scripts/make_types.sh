#!/bin/bash


GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting LCM type generation...${NC}"
CURRENT_FILE=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR="$( cd "$( dirname "${CURRENT_FILE}" )" && pwd )"
echo "Script directory: ${SCRIPT_DIR}"
echo "Generating LCM types in ${SCRIPT_DIR}/../lcm_types"

mkdir -p ${SCRIPT_DIR}/../lcm_types
cd ${SCRIPT_DIR}/../lcm_types
# Clean
rm */*.jar
rm */*.java
rm */*.hpp
rm */*.class
rm */*.py
rm */*.pyc

# Make
lcm-gen -jxp ${SCRIPT_DIR}/../*.lcm
cp /usr/local/share/java/lcm.jar .
javac -cp lcm.jar */*.java
jar cf my_types.jar */*.class
mkdir -p java
mv my_types.jar java
mv lcm.jar java
mkdir -p cpp
mv *.hpp cpp

mkdir -p python
mv *.py python
echo -e "${GREEN} Done with LCM type generation${NC}"