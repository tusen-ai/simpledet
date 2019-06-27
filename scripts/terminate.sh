#!/bin/bash

if [ $# -ne 1 ]; then
echo "usage: $0 comma_separated_worker_hostnames"
exit -1
fi

hosts=$1

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
	if [ $x -gt 5 ]; then 
	    echo "$host has gpu utilization of $x%"; 
        fi;  
    done
    
    # cleanup potentially dead python process (march since we checked it)
    ssh $host ps aux | grep python
    echo -e "\n"
    echo "Terminate tasks on $host in 5s"
    sleep 5
    ssh -q $host pkill python
done
