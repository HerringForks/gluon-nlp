import argparse
import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNet
from sagemaker.inputs import FileSystemInput
import string
import random

def get_job_name(size=8, chars=string.ascii_uppercase + string.digits):
    return 'MXNet-test-'+''.join(random.choice(chars) for _ in range(size))

if __name__ == '__main__':
    role = get_execution_role()
    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']
    session = boto3.session.Session()
    sagemaker_session = sagemaker.session.Session(boto_session=session)

    subnets = ['subnet-02da219d37e84a7af'] # us-east-1b
    security_group_ids = ['sg-01a3bc0056722f294']
    fsx_id = 'fs-0136b25df4e8b2abd'

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-type', type=str, default='ml.p3dn.24xlarge')
    parser.add_argument('--instance-count', type=int, default=8)
    parser.add_argument('--mode', type=str, default='perf')
    args, _ = parser.parse_known_args()

    # copy of 578276202366.dkr.ecr.us-west-2.amazonaws.com/karjar-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-2020-12-22-02-25-33
    docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/muziy-mx1.8-smd:base'

    SM_DATA_ROOT = '/opt/ml/input/data/train'

    if args.mode == 'perf':
        log_interval = 1
        lr = 0.00176
        num_steps = 2000
        warmup_ratio = 1
        timeout = 5400
    elif args.mode = 'full':
        # 8 node full run config from LAMB paper https://arxiv.org/pdf/1904.00962.pdf
        log_interval = 250
        lr = 0.00176
        num_steps = 112500
        warmup_ratio = 0.025
        timeout = 172800 # 2 day

    hyperparameters={
        "data": '/'.join([SM_DATA_ROOT, 'bert/train']),
        "data_eval": '/'.join([SM_DATA_ROOT, 'bert/eval']),
        "ckpt_dir": '/'.join([SM_DATA_ROOT, 'ckpt_dir']),
        "comm_backend": "smddp",
        "model": "bert_24_1024_16",
        "total_batch_size": args.instance_count * 512,
        "total_batch_size_eval": args.instance_count * 512,
        "max_seq_length": 128,
        "max_predictions_per_seq": 20,
        'log_interval': log_interval,
        "lr": lr,
        "num_steps": num_steps,
        'warmup_ratio': warmup_ratio,
        "raw": '',
        "skip_save_states": '',
        "seed": 987
    }

    distribution = {
        'smdistributed':{
            'dataparallel':{
                'enabled': True
            }
        }
    }

    estimator = MXNet(entry_point='run_pretraining_sm.py',
                        max_run=timeout,
                        role=role,
                        image_uri=docker_image,
                        source_dir='.',
                        train_instance_count=instance_count,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_session,
                        hyperparameters=hyperparameters,
                        distribution=distribution,
                        subnets=subnets,
                        security_group_ids=security_group_ids,
                        debugger_hook_config=False)

    train_fs = FileSystemInput(file_system_id=fsx_id,
                            file_system_type='FSxLustre',
                            directory_path='/fsx',
                            file_system_access_mode='rw')
    data={"train": train_fs}

    estimator.fit(inputs=data, job_name=get_job_name())
