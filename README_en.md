# Light Music Player

This project is a music-and-light visualization demo program based on the **Quectel Pi H1 Smart Single-Board Computer**. 
By playing local music files and performing real-time audio analysis, it drives a **16×16 WS2812 RGB LED matrix** to display lighting effects that change in sync with the music rhythm. 
Implemented in Python, the project uses the SPI interface to emulate WS2812 timing, enabling stable LED panel driving on Linux without an additional MCU.

## Features

- Plays local MP3 / WAV audio files 
- Real-time volume and spectrum analysis 
- Supports 16×16 WS2812 RGB LED matrix display 
- Provides two lighting effect modes:
  - Spectrum push effect
  - Raindrop lighting effect
- Supports keyboard controls for track switching and exit 
- Uses SPI to emulate the WS2812 protocol, supporting brightness control and gamma correction 

## Hardware Requirements

- Quectel Pi H1 Smart Single-Board Computer 
- 16×16 WS2812 RGB LED matrix 
- Audio output device (speaker or headphones) 

## Software Environment

- OS: Debian 13 (default system on Quectel Pi H1) 
- Python: Python 3.13 
- Dependencies:
  - `pulseaudio-utils`
  - `ffmpeg`
  - `python3-spidev`
  - `python3-numpy` (optional, for spectrum analysis)

### Installing Dependencies

```bash
sudo apt update && sudo apt install -y pulseaudio-utils python3-numpy python3-spidev ffmpeg
```

### Run Examples

```bash
sudo python3 light_music_player.py        # Default: mode1
sudo python3 light_music_player.py mode1  # Mode 1
sudo python3 light_music_player.py mode2  # Mode 2
```

## Project Structure

```text
light_music_player/
├── assets                      # Static resource files
├── src/                        # Source code directory
│   ├── light_music_player.py
├── README.md                   # Project documentation (Chinese)
└── README_en.md                # Project documentation (English)
```

## Contributing

Contributions to improve this project are welcome. Please follow these steps:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## Support

If you have any questions or need support, please refer to [Quectel Pi Documentation](https://developer.quectel.com/doc/sbc/) or open an issue in this repository.