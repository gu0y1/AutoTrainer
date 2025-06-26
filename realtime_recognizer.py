import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# === 加载模型组件 ===
clf = joblib.load("models/svm_model.joblib")
scaler = joblib.load("models/scaler.joblib")
encoder = joblib.load("models/label_encoder.joblib")

# === 加载支持中文的字体 ===
font_path = "C:/Windows/Fonts/msyh.ttc"  # 替换为你系统的中文字体路径
font = ImageFont.truetype(font_path, 36)

# === 初始化 MediaPipe Hands（最多检测 2 只手） ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(0)
print("📷 摄像头启动，请做出手语...按 ESC 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_num = len(results.multi_hand_landmarks)
        features = []

        # 最多处理两只手
        for i in range(min(hand_num, 2)):
            lm = results.multi_hand_landmarks[i].landmark
            features.extend([coord for pt in lm for coord in (pt.x, pt.y, pt.z)])

        # 如果只有一只手，补齐另一只手的关键点为 -1
        if hand_num == 1:
            features.extend([-1] * (21 * 3))

        # 添加是否为双手的维度
        features.append(1 if hand_num == 2 else 0)

        # 特征标准化并预测
        features_scaled = scaler.transform([features])
        pred = clf.predict(features_scaled)[0]
        label = encoder.inverse_transform([pred])[0]

        # ===== 使用 PIL 绘制中文文本 =====
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 50), f"识别结果：{label}", font=font, fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("中文手语识别", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
