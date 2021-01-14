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

    # copy of 578276202366.dkr.ecr.us-west-2.amazonaws.com/karjar-mxnet-herring-sagemaker:1.8.0-gpu-py37-cu110-ubuntu16.04-2020-12-22-02-25-33
    docker_image='578276202366.dkr.ecr.us-east-1.amazonaws.com/muziy-mx1.8-smd:base'

    instance_count = 8
    instance_type = "ml.p3dn.24xlarge"

    SM_DATA_ROOT = '/opt/ml/input/data/train'

    hyperparameters={
        "data": '/'.join([SM_DATA_ROOT, 'bert/train']),
        "data_eval": '/'.join([SM_DATA_ROOT, 'bert/eval']),
        "ckpt_dir": '/'.join([SM_DATA_ROOT, 'ckpt_dir']),
        "comm_backend": "horovod",
        "model": "bert_24_1024_16",
        "total_batch_size": instance_count * 256,
        "total_batch_size_eval": instance_count * 256,
        "max_seq_length": 128,
        "max_predictions_per_seq": 20,
        'log_interval': 10,
        "lr": 0.0001,
        "num_steps": 2000,
        'warmup_ratio': 1,
        "raw": '',
        "skip_save_states": ''
    }

    distribution = {'mpi': {'enabled': True, "custom_mpi_options": "-verbose --NCCL_DEBUG=INFO"}}

    estimator = MXNet(entry_point='run_pretraining_sm.py',
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
