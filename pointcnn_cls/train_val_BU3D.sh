#!/usr/bin/env bash

gpu=
setting=

usage() { echo "train/val pointcnn_cls with -g gpu_id -x setting -p prm_i options"; }

gpu_flag=0
setting_flag=0
while getopts g:x:p:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  p)
    prm_flag=1;
    prm=${OPTARG}
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ $prm_flag -eq 0 ]
then
  echo "-p option is not presented!"
  usage; exit;
fi

#echo "Train/Val prm_$prm with setting $setting on GPU $gpu!"
CUDA_VISIBLE_DEVICES=$gpu python -u ../train_val_cls.py -t ../../data/BU3D/prm_train_$prm.txt -v ../../data/BU3D/prm_valid_$prm.txt --save_folder_chenzhixing ../../models/BU3D_cls -m pointcnn_cls -x $setting > ../../models/BU3D_cls/pointcnn_cls_$setting.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -u ../train_val_cls.py -t ../../data/BU3D/small_train.txt -v ../../data/BU3D/small_valid.txt -s ../../models/BU3D_cls -m pointcnn_cls -x BU3D > ../../models/BU3D_cls/pointcnn_cls_BU3D.txt 2>&1 &
