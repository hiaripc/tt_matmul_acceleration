#!/bin/bash


BURNSCRIPT=torch_baseline.py
ROUNDS=1

starttime="$(date +%H:%M:%S)"

for np in 2 4 8; do
	echo "│   ┌── test with $np process $(date +%H:%M:%S)"

	# Settin process number				
	export OMP_NUM_THREADS=$np
	python $BURNSCRIPT
					
	sleep 2
	echo "│   └── ending test $np $(date +%H:%M:%S)"
done

echo "└── process $0 has finished: $starttime - $(date +%H:%M:%S)"

ctrl_c
exit 0
