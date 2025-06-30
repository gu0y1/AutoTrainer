import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image
import os

# === 模型路径配置 ===
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.joblib")  # 替换为 MLP 模型路径
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# === 字体路径（支持中文）===
font_path = "C:/Windows/Fonts/msyh.ttc"  # 可替换为其他系统字体
font = ImageFont.truetype(font_path, 36)

# === 加载模型组件 ===
try:
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit(1)

# === 初始化 MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

# === 摄像头启动 ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败")
    exit(1)

print("📷 摄像头已启动，开始手语识别（按 ESC 退出）")

# === 主循环 ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # ===== 提取关键点特征 =====
    if results.multi_hand_landmarks:
        hand_num = len(results.multi_hand_landmarks)
        features = []

        # 最多提取两只手
        for i in range(min(hand_num, 2)):
            lm = results.multi_hand_landmarks[i].landmark
            features.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

        # 如果只有一只手，补齐另一只手为 -1
        if hand_num == 1:
            features.extend([-1] * (21 * 3))

        # 添加是否为双手的维度
        features.append(1 if hand_num == 2 else 0)

        # 特征标准化与推理
        features_scaled = scaler.transform([features])
        pred = clf.predict(features_scaled)[0]
        label = encoder.inverse_transform([pred])[0]

        # 计算置信度（可选）
        if hasattr(clf, "predict_proba"):
            confidence = np.max(clf.predict_proba(features_scaled)) * 100
        else:
            confidence = 100.0  # 若模型不支持 probas，则默认显示 100%

        # ===== 使用 PIL 绘制中文识别结果 =====
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 50), f"识别结果：{label} ({confidence:.1f}%)", font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 显示图像
    cv2.imshow("中文手语识别", frame)

    # ESC 键退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === 释放资源 ===
cap.release()
cv2.destroyAllWindows()
