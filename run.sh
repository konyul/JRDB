#python3 tools/train.py configs/Faster_config/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py --work-dir Fast/

CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/dist_train.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py 4 --work-dir logs/

CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/dist_test.sh configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py logs0.0001/epoch_9.pth 4 --eval mAP

python3 alarm.py

#python3 tools/test.py configs/Faster_config/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-037c8118.pth --eval mAP --gpu-collect

