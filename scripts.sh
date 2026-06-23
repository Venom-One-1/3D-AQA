# 单视频对
python run_3d_aqa.py \
  --student-video /home/sqw/VisualSearch/aqa/ActionSegments/student/2_1_qishi.mp4 \
  --teacher-video /home/sqw/VisualSearch/aqa/ActionSegments/teach/QxVvRcRn2TA_1_qishi.mp4 \
  --student-tracking /home/sqw/VisualSearch/aqa/Tracking/student/2_1_qishi/results/demo_2_1_qishi.pkl \
  --teacher-tracking /home/sqw/VisualSearch/aqa/Tracking/teach/QxVvRcRn2TA_1_qishi/results/demo_QxVvRcRn2TA_1_qishi.pkl \
  --output-dir results/2_1_qishi \
  --device cuda:2 --yolo-batch-size 8


# 检查 tracking 结果
python inspect_tracking_pkl.py \
  /home/sqw/VisualSearch/aqa/Tracking/teach/QxVvRcRn2TA_1_qishi/results/demo_QxVvRcRn2TA_1_qishi.pkl \
  --frame 1

