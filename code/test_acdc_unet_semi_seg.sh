CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Mean_Teacher --num_classes 4 --labeled_num 7 && \
CUDA_VISIBLE_DEVICES=0 python test_2D_fully.py --root_path ../data/ACDC --exp ACDC/Fully_Supervised --num_classes 4 --labeled_num 140
python test_2D_fully_LA.py --root_path ../data/LA --exp LA/Mean_Teacher --num_classes 2 --labeled_num 8
