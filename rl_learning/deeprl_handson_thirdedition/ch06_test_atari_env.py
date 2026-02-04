import gymnasium as gym
import ale_py
import cv2
import numpy as np
import time

# 创建环境，不直接渲染到窗口
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
state, _ = env.reset()

# 创建OpenCV窗口
window_name = "Pong Game (Resized)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 400, 300)  # 设置窗口初始大小

# 可以调整的比例
scale_percent = 50  # 缩小到50%

try:
    for _ in range(1000):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        # 获取当前帧
        frame = env.render()
        
        # 调整帧大小
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # 显示调整后的帧
        cv2.imshow(window_name, resized_frame[:, :, ::-1])  # BGR转换为RGB
        
        # 检测按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('+'):  # 按+放大
            scale_percent = min(200, scale_percent + 10)
        elif key == ord('-'):  # 按-缩小
            scale_percent = max(20, scale_percent - 10)
        elif key == ord('r'):  # 按r重置
            state, _ = env.reset()
        
        if terminated or truncated:
            state, _ = env.reset()
            
        time.sleep(0.01)
        
finally:
    cv2.destroyAllWindows()
    env.close()