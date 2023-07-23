#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "$SCRIPT_DIR/src"
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"
