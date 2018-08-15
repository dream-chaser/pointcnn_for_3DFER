i=0
while [ $i -lt 10 ]
do
  count=`ps -ef | grep "save_folder_chenzhixing_1" | grep -v "grep" | wc -l`
  if [ 0 -eq $count ]; then
    `CUDA_VISIBLE_DEVICES=0 nohup python -u ../train_val_cls.py -t ../../data/BU3D/prm_train_$i.txt -v ../../data/BU3D/prm_valid_$i.txt --save_folder_chenzhixing_1 ../../models/BU3D_cls -m pointcnn_cls -x BU3D_multiscale -n prm_1_$i > ../../models/BU3D_cls/pointcnn_1.txt 2>&1 &`
    i=$(($i+1))
  fi
  sleep 10
done
