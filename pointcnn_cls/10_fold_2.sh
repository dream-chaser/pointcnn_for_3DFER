i=8
while [ $i -ge 0 ]
do
  count=`ps -ef | grep "save_folder_chenzhixing_2" | grep -v "grep" | wc -l`
  if [ 0 -eq $count ]; then
    `CUDA_VISIBLE_DEVICES=1 nohup python -u ../train_val_cls_2.py -t ../../data/BU3D/prm_train_$i.txt -v ../../data/BU3D/prm_valid_$i.txt --save_folder_chenzhixing_2 ../../models/BU3D_cls -m pointcnn_cls -x BU3D_multiscale -n prm_2_$i > ../../models/BU3D_cls/pointcnn_2.txt 2>&1 &`
    i=$(($i-1))
  fi
  sleep 1
done
