python main.py --n_GPUs=1  \
               --data_train='RefMRI' \
               --name_train='T2_val' \
               --data_test='RefMRI' \
               --name_test='T2_newtest' \
               --dir_data='/ssd/dataset/pyw/ixi'  \
               --loss="1*L1"  \
               --model="dualref"  \
               --save="cartesian25"  \
               --batch_size=1 \
               --patch_size=128 \
               --resume=0  \
               --n_color=2 \
               --rgb_range=1 \
               --ref_mat='PD_ref' \
               --ref_list='matching.txt' \
               --pre_train=None