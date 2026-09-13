import os
import pandas as pd
from moviepy import VideoFileClip
from tqdm import tqdm

tagid2str = {
    "1": "qishi", 
    "2": "yemafenzong",
    "3": "baiheliangchi",
    "4": "louxiaobu"
}

def video2segments():
    # 输入路径
    csv_file = "/home/sqw/Projects/annotation-tool/annotations/instruction_teach.txt"
    video_dir = "/home/sqw/VisualSearch/aqa/teach"
    output_dir = "/home/sqw/VisualSearch/aqa/ActionSegments/teach"

    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV
    df = pd.read_csv(csv_file)

    # 去除可能的空行
    df = df.dropna(subset=["URL"])

    # 将 Start 和 End 转为浮点数，并四舍五入为整数
    df["Start"] = df["Start"].astype(float).round().astype(int)
    df["End"] = df["End"].astype(float).round().astype(int)

    # 按视频与动作分组
    groups = df.groupby(["URLID", "URL", "TagID"])

    for (urlid, url, tagid), group in tqdm(groups, desc="Processing videos"):
        if urlid != "QxVvRcRn2TA":
            continue
        
        video_path = os.path.join(video_dir, url)
        if not os.path.exists(video_path):
            print(f"找不到视频: {video_path}")
            continue

        # 获取该动作的时间范围
        start_time = group["Start"].min()
        end_time = group["End"].max()

        if str(tagid) not in tagid2str: # 只划分前三式
            continue

        # 输出文件名
        str_tag = tagid2str[str(tagid)]
        output_name = f"{urlid}_{tagid}_{str_tag}.mp4"
        output_path = os.path.join(output_dir, output_name)
    
        # 截取视频片段
        try:
            clip = VideoFileClip(video_path).subclipped(start_time, end_time)
            clip.write_videofile(output_path)
        except Exception as e:
            print(f"处理 {video_path} 时出错: {e}")

if __name__ == "__main__":
    video2segments()
