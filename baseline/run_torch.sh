#!/bin/bash
source /usr/etc/profile.d/conda.sh
conda activate py-intel

if [ $# -lt 1 ]; then
	echo "$0 name"
	exit 1
fi	

source .support.sh

BASE_DIR=~/pmu_pub/tests_results
BASE_SCRIPTS=~/pmu_pub
TPI_DIR=~/tpi
BASENAME="$1"

if [ ! -d $BASE_DIR/$BASENAME ]; then
	mkdir $BASE_DIR/$BASENAME
fi

#POWERVIRUS
PVBASE=/home/unibo
#ARGUMENTS
CPU_NUMBER=2 		# Number of CPUs, not cores!
SAMPLING_TIME=0.1	# Dont think so...
POWERVIRUS_PATH=$TPI_DIR/trace-generation/scripts/powervirus/burnP6
CONF_PATH=$TPI_DIR/trace-generation/scripts/burn_cpu_mqtt.conf
BURNSCRIPT=hpizzini/intel_pytorch_example/run_llm.py
ROUNDS=1

starttime="$(date +%H:%M:%S)"

for np in 14 28 56 112; do
	echo "│   ┌── test with $np process $(date +%H:%M:%S)"
	for modelname in "llama2_7b" ; do
		echo "│   │   ┌── starting $modelname $(date +%H:%M:%S)"
		for quantization in "bf16" "f32" "int8"; do
			for	((round=0; round<$ROUNDS; round++)); do
				echo -e "│   │   │   ┌── starting round $round $(date +%H:%M:%S)"
				# Prepare for subfolder parameters
				running_dir="$BASE_DIR/$BASENAME/$np/$modelname/$quantization"
				if [ ! -d $running_dir ]; then
					mkdir -p $running_dir
				fi
				$BASE_SCRIPTS/publishers/pmu_pub/pmu_pub -P 1 run &>> $running_dir/round$round\_pub.mqtt &
				PUB=$!

				mosquitto_sub -v -t "#" &>> $running_dir/round$round\_sub.mqtt &
				SUB=$!

				sleep 1

				echo "$np] starting_tmps  $(date +%s)" >> $running_dir/round$round\_time.mqtt
				# Settin process number				
				export OMP_NUM_THREADS=$np
				python $PVBASE/$BURNSCRIPT --model $modelname --dtype $quantization &>> $running_dir/round$round\_burn.mqtt
							
				echo "$np] ending_tmps $(date +%s)" >> $running_dir/round$round\_time.mqtt
				echo -e "│   │   │   └── round $round finished $(date +%H:%M:%S)"
				
				stop_pmu
				sleep 5
			done
		done
		echo "│   │   └── $modelname finished $(date +%H:%M:%S)"
	done
	echo "│   └── ending test $np $(date +%H:%M:%S)"
done

echo "└── process $0 has finished: $starttime - $(date +%H:%M:%S)"

ctrl_c
exit 0
