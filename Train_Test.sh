################
### kaist
################
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.008_Origin_KL_ir --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types ir
#CUDA_VISIBLE_DEVICES=1 python test_net.py ResNet101_lr0.008_Origin_KL_ir --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types ir #--vis 1884
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.008_Origin_KL_rgb --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types rgb
#CUDA_VISIBLE_DEVICES=1 python test_net.py ResNet101_lr0.008_Origin_KL_rgb --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types rgb #--vis
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.008_Origin_KL_all --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all
#CUDA_VISIBLE_DEVICES=1 python test_net.py ResNet101_lr0.008_Origin_KL_all --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types all #--vis

#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.008_Origin_KL_all_0525 --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all
#CUDA_VISIBLE_DEVICES=1 python test_net.py ResNet101_lr0.008_Origin_KL_all_0525 --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types all #--vis

#CUDA_VISIBLE_DEVICES=0 python trainval_net.py ResNet101_lr0.008_Origin_KL_ir --dataset kaist --cuda --bs 1 --cag --s 1 --types all

######### KAIST Dataset NeurIPS2020 Experiments ########## 
########## ResNet101 Fusion ##########
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.008_fus_ep3 --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res101 --UKLoss OFF --epochs 3 --lr_decay_step 2

CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet101_lr0.008_fus_ep3 --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --checksession 2 --types all --UKLoss OFF --net res101 
######################################

########## ResNet101 Uncer KL ##########
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_lr0.004_Uncer_KL_ep3 --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res101 --UKLoss ON --epochs 3 --lr_decay_step 2

#CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet101_lr0.004_Uncer_KL_ep3 --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --checksession 2 --types all --UKLoss ON --net res101
#CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet101_lr0.008_fus_ep3 --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --checksession 2 --types all --UKLoss OFF --net res101


########################################

########## ResNet50 Fusion ##########
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet50_lr0.008_fus --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res50 --UKLoss OFF

#CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet50_lr0.008_fus --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types all --UKLoss OFF --net res50
#####################################

########## ResNet50 Uncer KL ##########
#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet50_lr0.008_Uncer_KL --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res50 --UKLoss ON

#CUDA_VISIBLE_DEVICES=2 python test_net.py ResNet50_lr0.008_Uncer_KL --dataset kaist --cuda --cag --checkepoch 2 --checkpoint 3769 --checksession 2 --types all --UKLoss ON --net res50
#######################################


###################################################################################################

#CUDA_VISIBLE_DEVICES=0,1,2,3 python trainval_net.py ResNet101_to_extract_feat --dataset kaist --cuda --mGPUs --bs 4 --cag --s 2 --types all --net res101 --UKLoss OFF --epochs 3 --lr_decay_step 2
#CUDA_VISIBLE_DEVICES=0 python test_net.py ResNet101_to_extract_feat --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --checksession 2 --types all --UKLoss OFF --net res101
###################################################################################################

## for visualization histogram
#CUDA_VISIBLE_DEVICES=2 python extract_train_feat.py ResNet101_to_extract_feat --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --checksession 2 --types all --UKLoss OFF --net res101 --r True --checkepoch 3 --checkpoint 3769 --checksession 2

#CUDA_VISIBLE_DEVICES=0 python extract_train_feat.py ResNet101_lr0.008_Uncer_KL_ep3 --dataset kaist --cuda --cag --checkepoch 3 --checkpoint 3769 --types all --UKLoss ON --net res101 --r True --checkepoch 3 --checkpoint 3769 --checksession 2


