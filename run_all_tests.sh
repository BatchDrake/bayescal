#!/bin/bash

for i in */; do
    test_name=`basename $i`
    if [ -f "$i/run.sh" ]; then
	echo "Launching test ${test_name}..."
	# bash $test_name/run.sh
    fi
done
    
