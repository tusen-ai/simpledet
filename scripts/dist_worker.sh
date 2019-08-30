root_dir=$1
singularity_image=$2
conffile=$3

if test $(which singularity); then
    singularity exec -B /mnt:/mnt -s /usr/bin/zsh --no-home --nv ${singularity_image} zsh -ic "MXNET_UPDATE_ON_KVSTORE=0 MXNET_OPTIMIZER_AGGREGATION_SIZE=20 python -u detection_train.py --config ${conffile}"
else
    singularity exec -B /mnt:/mnt -s /usr/bin/zsh --no-home --nv ${singularity_image} zsh -ic "python -u detection_train.py"
fi
