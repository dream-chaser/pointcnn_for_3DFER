i=0
while [ $i -lt 10 ]
do
  count=`ps -ef | grep "save_folder_chenzhixing" | grep -v "grep" | wc -l`
  if [ 0 -eq $count ]; then
    #`CUDA_VISIBLE_DEVICES=0 python -u train_val_cls.py --save_folder_chenzhixing ../models/BU3D_cls -m pointcnn_cls -x BU3D_predict -l /home/chenzhixing/code/tensorflow_3D_DNN/models/BU3D_cls/pointcnn_cls_BU3D_train_prm_$i/ckpts/best_model -t ../data/BU3D/prm_train_$i.txt -v ../data/BU3D/prm_valid_$i.txt -n prm_$i`
    `CUDA_VISIBLE_DEVICES=0 python -u train_val_cls.py --save_folder_chenzhixing ../models/BU3D_cls -m pointcnn_cls -x BU3D -t ../data/BU3D/prm_train_$i.txt -v ../data/BU3D/prm_valid_$i.txt -n 4_layer_train_prm_$i`
    i=$(($i+1))
  fi
  sleep 2
done
