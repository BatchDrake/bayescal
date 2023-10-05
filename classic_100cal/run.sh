#!/bin/bash

path=`readlink -f $0`
dir=`dirname "$path"`
testname=`basename "$dir"`
testroot="$dir/.."
testdest="$testroot/test_$testname.py"

cp "$dir"/main.py "$testdest"
cd "$dir"
rm -f priors.hf5
python3 "$testdest"
