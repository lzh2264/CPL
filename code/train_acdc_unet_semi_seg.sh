CUDA_VISIBLE_DEVICES=0 python train_fully_supervised_2D.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python train_mean_teacher_2D.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher --num_classes 4 --labeled_num 7 && \
