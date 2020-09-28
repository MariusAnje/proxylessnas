#!/bin/bash 
#JSUB -J zheyu_RLDR
#JSUB -q tensorflow_sub
#JSUB -gpgpu "num=1" 
#JSUB -R "span[ptile=1]"
#JSUB -n 1
#JSUB -o output.%J
#JSUB -e err.%J 
 
##########################Cluster environment variable###################### 
if [ -z "$LSB_HOSTS" -a -n "$JH_HOSTS" ]
then
        for var in ${JH_HOSTS}
        do
                if ((++i%2==1))
                then
                        hostnode="${var}"
                else
                        ncpu="$(($ncpu + $var))"
                        hostlist="$hostlist $(for node in $(seq 1 $var);do printf "%s " $hostnode;done)"
                fi
        done
        export LSB_MCPU_HOSTS="$JH_HOSTS"
        export LSB_HOSTS="$(echo $hostlist|tr ' ' '\n')"
fi

nodelist=.hostfile.$$
for i in `echo $LSB_HOSTS`
do
    echo "${i}" >> $nodelist
done

ncpu=`echo $LSB_HOSTS |wc -w`
##########################Software environment variable##################### 
module load cuda/cuda10.1 
module load pytorch/pytorch1.5.1 
##########################Software run command############################## 
# python3 cifar10_arch_search.py --train_batch_size 128 --n_worker 0 --warmup_epochs 20 --n_epochs 200 --print_frequency 100
python3 imagenet_arch_search.py --warmup_epochs 0 --arch_algo rl
