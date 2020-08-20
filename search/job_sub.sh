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
##python3 /data/users/xiaowei/pytorch_test/pytorch_mnist.py 
# python3 -u main.py nas  -ns -e 30 -ep 51 --train_epochs 30 --augment
# python3 -u main.py joint  -ns -e 30 -ep 500 --train_epochs 30 --augment
# python3 -u main.py quantization -ns -e 30 -ep 10 --train_epochs 10 --rollout_filename ./experiment/rollout_record0728_2337_46 --method comp --wsSize 9 -b 128
python3 -u finetune.py
##>> /data/users/xiaowei/pytorch_test/xiaowei_test.log 
