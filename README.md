# Stretch VLA ROS 2 Workspace

這個工作區聚合 Hello Robot Stretch 的 VLA（Visual Language Acting）相關套件，涵蓋控制、GUI、感測器以及自訂介面。透過 ROS 2 Humble，可在本機環境直接開發、整合與測試。

## 先決條件
- Ubuntu 22.04 或其他支援 ROS 2 Humble 的系統
- 已安裝 ROS 2 Humble 基礎套件 (`ros-humble-desktop` 或等效)
- `colcon`, `rosdep`, `python3-vcstool`
- 依需求安裝 RealSense SDK、Hello Robot Stretch SDK

## 初始化環境
```bash
sudo apt update
sudo rosdep init    # 若從未執行過
rosdep update
```

## 安裝依賴並建置
```bash
cd ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
```

建議將最後兩行加入殼層啟動腳本（例如 `~/.bashrc`），以便每次開啟終端時都能載入 ROS 2 與工作區環境。

## 常用啟動指令
- 控制節點：`ros2 launch stretch_vla_control control_launch.py`
- GUI 節點：`ros2 run stretch_gui gui_node`
- RealSense D435i 節點：`ros2 launch d435i_camera camera_launch.py`

啟動前請確認相機、機器人等裝置權限已配置（如將使用者加入 `dialout`, `video` 群組），並視需要於多個終端分別載入 setup 檔案。

## 專案結構
- `stretch_vla_control/`：負責 VLA 控制邏輯與硬體命令
- `stretch_gui/`：操作介面與顯示
- `stretch_interfaces/`：自訂訊息與服務定義
- `d435i_camera/`：RealSense D435i 相機整合
- `llava_server/`：語言模型服務（若需要可於需求時啟用）

## 開發與除錯提示
- 使用 `ros2 topic list`, `ros2 service list` 觀察節點間溝通
- `colcon test` 可執行套件自帶測試（若已提供）
- 確保程式碼遵循 ROS 2 標準，利於後續整合與部署

若有額外流程（如容器化、CI 管線），歡迎依需求擴充本文件。
