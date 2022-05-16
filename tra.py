import subprocess
import os

command = f'/opt/conda/bin/python -u train.py --feature_path /workspace/pre_processing/vcdb_mobilenet_avg_pca1024.hdf5 ' \
          f'--model_path /mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_mobilenet_avg_batch64_padding64_negnum16_momentum0999'
print("command :", command)
subprocess.call(command, shell=True)

command = f'/opt/conda/bin/python -u train.py --feature_path /workspace/pre_processing/vcdb_resnet50_imac_pca1024.hdf5 ' \
          f'--model_path /mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_resnet50_imac_batch64_padding64_negnum16_momentum0999'
print("command :", command)
subprocess.call(command, shell=True)

command = f'/opt/conda/bin/python -u train.py --feature_path /workspace/pre_processing/vcdb_resnet50_rmac_pca1024.hdf5 ' \
          f'--model_path /mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_resnet50_rmac_batch64_padding64_negnum16_momentum0999'
print("command :", command)
subprocess.call(command, shell=True)