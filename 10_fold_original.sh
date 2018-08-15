i=0
while [ $i -lt 10 ]
do
  count=`ps -ef | grep "save_folder_chenzhixing_original" | grep -v "grep" | wc -l`
  if [ 0 -eq $count ]; then
    # predict
    #`CUDA_VISIBLE_DEVICES=0 python -u train_val_cls.py --save_folder_chenzhixing ../models/BU3D_cls -m pointcnn_cls -x BU3D_predict -l /home/chenzhixing/code/tensorflow_3D_DNN/models/BU3D_cls/pointcnn_cls_BU3D_train_prm_$i/ckpts/best_model -t ../data/BU3D/prm_train_$i.txt -v ../data/BU3D/prm_valid_$i.txt -n prm_$i`
    # train
    `CUDA_VISIBLE_DEVICES=0 nohup python -u train_val_cls_original.py --save_folder_chenzhixing_original ../models/BU3D_cls -m pointcnn_cls_original -x BU3D_original -t ../data/BU3D/prm_train_$i.txt -v ../data/BU3D/prm_valid_$i.txt -n original_2_prm_$i > ../models/BU3D_cls/pointcnn_original_prm_$i.txt 2>&1 &`
    i=$(($i+1))
  fi
  sleep 2
done

#CUDA_VISIBLE_DEVICES=2 python -u train_val_cls_predict.py --save_folder_chenzhixing_original ../models/BU3D_cls_predict -m pointcnn_cls_original -x BU3D_original_predict -l ../models/BU3D_cls/pointcnn_cls_original_BU3D_original_original_prm_0/ckpts/best_model -t ../data/BU3D/prm_train_0.txt -v ../data/BU3D/prm_test_0.txt -n original_predict_prm_0
