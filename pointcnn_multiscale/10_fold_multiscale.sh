i=0
while [ $i -lt 10 ]
do
  count=`ps -ef | grep "save_folder_chenzhixing_multiscale" | grep -v "grep" | wc -l`
  if [ 0 -eq $count ]; then
    # predict
    #`CUDA_VISIBLE_DEVICES=0 python -u train_val_cls.py --save_folder_chenzhixing ../models/BU3D_cls -m pointcnn_cls -x BU3D_predict -l /home/chenzhixing/code/tensorflow_3D_DNN/models/BU3D_cls/pointcnn_cls_BU3D_train_prm_$i/ckpts/best_model -t ../data/BU3D/prm_train_$i.txt -v ../data/BU3D/prm_valid_$i.txt -n prm_$i`
    # train
    `CUDA_VISIBLE_DEVICES=0 nohup python -u train_val_cls_multiscale.py --save_folder_chenzhixing_multiscale ../../models/BU3D_cls -m pointcnn_cls_multiscale -x BU3D_multiscale -t ../../data/BU3D/prm_train_$i.txt -v ../../data/BU3D/prm_valid_$i.txt -n multiscale_prm_$i > ../../models/BU3D_cls/pointcnn_multiscale.txt 2>&1 &`
    i=$(($i+1))
  fi
  sleep 2
done

CUDA_VISIBLE_DEVICES=2 python -u train_val_cls_multiscale_predict.py --save_folder_chenzhixing_multiscale ../../models/BU3D_cls_predict -m pointcnn_cls_multiscale -x BU3D_multiscale_predict -l ../../models/BU3D_cls/pointcnn_cls_multiscale_BU3D_multiscale_multiscale_prm_0/ckpts/best_model -t ../../data/BU3D/prm_train_0.txt -v ../../data/BU3D/prm_test_0.txt -n multiscale_predict_prm_0

CUDA_VISIBLE_DEVICES=2 python -u train_val_cls_multiscale_predict.py --save_folder_chenzhixing_multiscale ../../models/BU3D_cls_predict -m pointcnn_cls_multiscale -x BU3D_multiscale_predict -l ../../models/BU3D_cls/pointcnn_cls_multiscale_BU3D_multiscale_multiscale_2_prm_0/ckpts/best_model -t ../../data/BU3D/prm_train_0.txt -v ../../data/BU3D/prm_test_0.txt -n multiscale_2_predict_prm_0
