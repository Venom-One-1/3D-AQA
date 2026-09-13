import os
import cv2
import logging
import re
import shutil
import tempfile
from tqdm import tqdm

# 输入输出路径
video_dir = "/home/sqw/VisualSearch/aqa/teach"
output_dir = "/home/sqw/VisualSearch/aqa/FrameData/teach"
log_file = "/home/sqw/VisualSearch/aqa/FrameData/teach/extract_teach.log"
frame_info_file = "/home/sqw/VisualSearch/aqa/FrameData/teach/frame.txt"
video_info_file = "/home/sqw/VisualSearch/aqa/FrameData/teach/video.txt"

# 日志配置
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

FRAME_NAME_RE = re.compile(r"^\d{5}\.jpg$")


def clear_existing_frames(output_path):
    """Remove frames created by this script so reruns do not leave stale files."""
    if not os.path.isdir(output_path):
        return

    for name in os.listdir(output_path):
        if FRAME_NAME_RE.match(name):
            os.remove(os.path.join(output_path, name))


def get_frame_timestamp(cap, frame_count, fps):
    timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp_msec and timestamp_msec > 0:
        return timestamp_msec / 1000.0
    if fps and fps > 0:
        return frame_count / fps
    return None


def extract_frames(video_path, output_path, urlid, file_handle=None, sample_fps=1.0):
    cap = None
    temp_output_path = None
    pending_frame_info = []

    try:
        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {sample_fps}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {video_path}")

        output_parent = os.path.dirname(output_path)
        os.makedirs(output_parent, exist_ok=True)
        temp_output_path = tempfile.mkdtemp(
            prefix=f".{os.path.basename(output_path)}_",
            suffix=".tmp",
            dir=output_parent
        )

        fps = cap.get(cv2.CAP_PROP_FPS)  # 原视频帧率，仅作为时间戳不可用时的备用值
        sample_interval = 1.0 / sample_fps
        next_sample_time = 0.0
        frame_count, saved_count = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = get_frame_timestamp(cap, frame_count, fps)
            if timestamp is None:
                raise RuntimeError(f"无法获取视频时间戳且 FPS 无效: {video_path}")

            if timestamp + 1e-6 >= next_sample_time:
                # save frame 
                frame_name = f"{saved_count:05d}.jpg"
                save_path = os.path.join(temp_output_path, frame_name)
                if not cv2.imwrite(save_path, frame):
                    raise IOError(f"无法保存视频帧: {save_path}")

                saved_count += 1

                # 写入帧信息
                if file_handle is not None:
                    pending_frame_info.append(f"{urlid},{frame_name},{timestamp:.1f}\n")

                while next_sample_time <= timestamp + 1e-6:
                    next_sample_time += sample_interval
            
            frame_count += 1

        if saved_count == 0:
            raise RuntimeError(f"未从视频中提取到任何帧: {video_path}")

        os.makedirs(output_path, exist_ok=True)
        clear_existing_frames(output_path)
        for frame_name in sorted(os.listdir(temp_output_path)):
            os.replace(
                os.path.join(temp_output_path, frame_name),
                os.path.join(output_path, frame_name)
            )

        if file_handle is not None:
            file_handle.writelines(pending_frame_info)

    except Exception:
        logging.exception(f"处理视频 {video_path} 时出错")

    finally:
        if cap is not None:
            cap.release()
        if temp_output_path is not None and os.path.isdir(temp_output_path):
            shutil.rmtree(temp_output_path)

def main():
    # 找到所有 mp4 文件
    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    videos = sorted(videos)
    # with open(video_info_file, "a", encoding='utf-8') as f:
    #     vids = sorted([f"{v[:-4]},{v}\n" for v in videos])
    #     f.writelines(vids)
    
    os.makedirs(os.path.dirname(frame_info_file), exist_ok=True)
    # with open(frame_info_file, "w", encoding="utf-8") as frame_file:
    with open(frame_info_file, "a", encoding="utf-8") as frame_file:

        for video in tqdm(videos, desc="Processing videos"):
            video_path = os.path.join(video_dir, video)
            urlid = os.path.splitext(video)[0]
            output_path = os.path.join(output_dir, urlid)
            os.makedirs(output_path, exist_ok=True)

            # frame_file = None
            extract_frames(video_path, output_path, urlid, frame_file, sample_fps=1.0)

if __name__ == "__main__":
    main()
