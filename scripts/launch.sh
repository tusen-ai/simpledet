#!/bin/bash

if [ $# -ne 2 ]; then
echo "usage: $0 config_path comma_separated_worker_hostnames"
exit -1
fi

conffile=$1
hosts=$2

# extract worker and check reachablity
IFS=, read -r -a host_array <<< $hosts
for host in ${host_array[@]}; do
    # check reachability
    echo "check reachability of $host"
    ssh -q $host exit
    if [ $? -ne 0 ]; then
        echo "$host is not reachable"
	exit -1
    fi

    # check availablity (retreat if remote host is in use)
    echo "check availability of $host"
    for x in $(ssh $host nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader); do 
	x="${x//[$'\t\r\n ']}"  # remove trailing whitespace
	if [ $x -gt 10 ]; then 
	    echo "$host has gpu utilization of $x%"; 
	    exit -1
        fi;  
    done
    
    # cleanup potentially dead python process (march since we checked it)
    ssh -q $host pkill python
done

gpucount=8
num_node=${#host_array[@]}
num_servers=${num_node}
root_dir="/mnt/tscpfs/yuntao.chen/simpledet/simpledet_open"
sync_dir="/tmp/simpledet_sync"
singularity_image=/mnt/tscpfs/yuntao.chen/simpledet.img

# check existence of config file
if [ ! -f ${conffile} ]; then
echo "${conffile} does not exsit"
exit -1
fi

# dump hosts in a hostfile for launch.py
IFS=,
output=""
for id in $hosts 
do output+="${id}\n"
done
unset IFS
echo -e ${output::-2} > scripts/hosts.txt
sleep 1

logfile=${conffile#config/}
logfile=${logfile%.py}

export DMLC_INTERFACE=eth0
python -u /mnt/tscpfs/yuntao.chen/dist-mxnet/tools/launch.py \
    -n ${num_node} \
    -s ${num_servers} \
    --launcher ssh \
    -H scripts/hosts.txt \
    scripts/dist_worker.sh ${root_dir} ${singularity_image} ${conffile} \
    2>&1 | tee -a ${root_dir}/log/${logfile}.log
