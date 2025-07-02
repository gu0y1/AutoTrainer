import cv2
import mediapipe as mp
import csv
import os

# ========= 配置部分 =========
video_dir = "videos"                 # 视频主目录（可含子目录）
output_dir = "video_frames"         # 所有帧图像统一保存目录
csv_path = "video_labels.csv"       # CSV 输出路径
skip_rate = 1                       # 每 N 帧采一次
# ===========================

# 初始化输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 初始化 MediaPipe（最多检测两只手）
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ==== 判断 CSV 是否首次写入 ====
is_first_write = not os.path.exists(csv_path)
write_mode = "w" if is_first_write else "a"

# 打开 CSV 文件
with open(csv_path, mode=write_mode, newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # 写入表头
    if is_first_write:
        header = ["frame_filename"] + \
                 [f"hand{i}_{pt}_{axis}" for i in range(2) for pt in range(21) for axis in ['x', 'y', 'z']] + \
                 ["is_two_hands", "label"]
        writer.writerow(header)

    # 遍历所有子文件夹和视频
    for root, dirs, files in os.walk(video_dir):
        for filename in files:
            if not filename.endswith(".mp4"):
                continue

            video_path = os.path.join(root, filename)
            label = os.path.basename(root)  # 使用上级目录作为标签

            print(f"\n🚀 开始处理视频：{filename}，标签：{label}")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"❌ 无法打开视频：{video_path}")
                continue

            frame_index = 0
            valid_frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_rate != 0:
                    frame_index += 1
                    continue

                frame_filename = f"{label}_{os.path.splitext(filename)[0]}_frame_{frame_index:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                success = cv2.imwrite(frame_path, frame)
                if not success:
                    print(f"⚠️ 无法保存帧 {frame_index} 到 {frame_path}")
                    frame_index += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                row = [frame_filename]

                if results.multi_hand_landmarks:
                    hand_num = len(results.multi_hand_landmarks)
                    landmarks_all = []

                    for i in range(min(hand_num, 2)):
                        lm = results.multi_hand_landmarks[i].landmark
                        landmarks_all.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

                    if hand_num == 1:
                        landmarks_all.extend([-1] * (21 * 3))  # 另一只手补齐

                    row.extend(landmarks_all)
                    row.append(1 if hand_num == 2 else 0)
                    valid_frame_count += 1
                    print(f"✅ 第 {frame_index:04d} 帧：检测到 {hand_num} 手")
                else:
                    row.extend([-1] * (21 * 3 * 2))  # 两手都补齐
                    row.append(0)
                    print(f"⛔ 第 {frame_index:04d} 帧：未检测到手")

                row.append(label)
                writer.writerow(row)
                frame_index += 1

            cap.release()
            print(f"📊 完成：{filename}，总帧：{frame_index}，有效帧：{valid_frame_count}")

hands.close()
print("\n✅ 所有视频处理完毕，数据写入：", csv_path)