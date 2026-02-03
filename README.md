# 灯光音乐播放器

本项目是一个基于 **Quectel Pi H1 智能主控板** 的音乐灯光可视化示例程序。  
程序通过播放本地音乐文件并进行实时音频分析，驱动 **16×16 WS2812 RGB LED 矩阵** 显示与音乐节奏同步变化的灯光效果。
项目采用 Python 实现，使用 SPI 接口模拟 WS2812 时序，在 Linux 系统下无需额外 MCU 即可稳定驱动 LED 灯板。

  ## 功能特性

  - 播放本地 MP3 / WAV 音乐文件  
  - 实时音量与频谱分析 
  - 支持 16×16 WS2812 RGB LED 矩阵显示  
  - 提供两种灯光效果模式：
    - 频谱推进效果
    - 雨滴灯光效果
  - 支持键盘切歌与退出控制  
  - 使用 SPI 模拟 WS2812 协议，支持亮度控制与 Gamma 校正  


  ## 硬件要求

  - Quectel Pi H1 智能主控板  
  - 16×16 WS2812 RGB LED 矩阵
  - 音频输出设备（扬声器或耳机）  


  ## 软件环境

  - 操作系统：Debian 13（Quectel Pi H1 默认系统）  
  - Python：Python 3.13  
  - 依赖组件：
    - `pulseaudio-utils`
    - `ffmpeg`
    - `python3-spidev`
    - `python3-numpy`（可选，用于频谱分析）

  ### 依赖安装

  ```bash
sudo apt update && sudo apt install -y pulseaudio-utils python3-numpy python3-spidev ffmpeg
  ```

###  运行示例

```
sudo python3 light_music_player.py        //默认模式1
sudo python3 light_music_player.py mode1  //模式1
sudo python3 light_music_player.py mode2  //模式2
```

##  目录结构

```
light_music_player/
├──assets                       # 静态资源文件
├── src/                        # 源代码目录
│   ├── light_music_player.py   
├── README.md                   # 项目说明文档
└── README_en.md                 
```

## 贡献

我们欢迎对本项目的改进做出贡献！请按照以下步骤进行贡献：

1. Fork 此仓库。
2. 创建一个新分支（`git checkout -b feature/your-feature`）。
3. 提交您的更改（`git commit -m 'Add your feature'`）。
4. 推送到分支（`git push origin feature/your-feature`）。
5. 打开一个 Pull Request。

## 支持

如果您有任何问题或需要支持，请参阅 [Quectel Pi 文档](https://developer.quectel.com/doc/sbc/) 或在本仓库中打开一个 issue。

