import subprocess
import os

# epochs = ['epoch5', 'epoch10', 'epoch15', 'epoch20', 'epoch25', 'epoch30', 'epoch35', 'epoch40']
# # epochs = ['epoch40']
# root_path = '/mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_mobilenet_avg_batch64_padding64_negnum16_momentum0999/'
#
# for epoch in epochs:
#     epoch_path = root_path + epoch + '.pth'
#     command = f'/opt/conda/bin/python -u evaluation.py --model_path {epoch_path} --feature_path /workspace/pre_processing/fivr_mobilenet_avg_pca1024.hdf5'
#     print("mobilenet avg, epoch :", epoch)
#
#     subprocess.call(command, shell=True)
#     print()

epochs = ['epoch5', 'epoch10', 'epoch15', 'epoch20', 'epoch25', 'epoch30', 'epoch35', 'epoch40']
# epochs = ['epoch40']
root_path = '/mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_resnet50_imac_batch64_padding64_negnum16_momentum0999/'

for epoch in epochs:
    epoch_path = root_path + epoch + '.pth'
    command = f'/opt/conda/bin/python -u evaluation.py --model_path {epoch_path} --feature_path /workspace/pre_processing/fivr_resnet50_imac_pca1024.hdf5'
    print("resnet50 imac, epoch :", epoch)

    subprocess.call(command, shell=True)
    print()


epochs = ['epoch5', 'epoch10', 'epoch15', 'epoch20', 'epoch25', 'epoch30', 'epoch35', 'epoch40']
# epochs = ['epoch40']
root_path = '/mldisk/nfs_shared_/js/contrastive_learning/model/vcdb_resnet50_rmac_batch64_padding64_negnum16_momentum0999/'

for epoch in epochs:
    epoch_path = root_path + epoch + '.pth'
    command = f'/opt/conda/bin/python -u evaluation.py --model_path {epoch_path} --feature_path /workspace/pre_processing/fivr_resnet50_rmac_pca1024.hdf5'
    print("resnet50 rmac, epoch :", epoch)

    subprocess.call(command, shell=True)
    print()
