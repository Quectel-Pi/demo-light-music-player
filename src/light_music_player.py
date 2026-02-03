#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import random
import signal
import threading
import subprocess
import select
from dataclasses import dataclass
from array import array

# ==========================
# 固定配置：不再支持 argparse 选项
#   通过启动参数选择模式：
#     sudo python3 light_music_player.py 1   -> 模式1 SpectrumPush
#     sudo python3 light_music_player.py 2   -> 模式2 Rain
# ==========================

MUSIC_DIR = "/home/pi/Music"

SPI_BUS = 10
SPI_DEV = 0
SPI_SPEED_HZ = 3200000
BRIGHTNESS = 0.30
RESET_US = 300

FPS = 60

# 默认启动模式（当命令行不带 1/2 时使用）
# 0: SpectrumPush（模式1）
# 1: Rain（模式2）
START_EFFECT = 0

SAMPLE_RATE = 44100
CHANNELS = 2
SHUFFLE_PLAYLIST = True

BG_COLOR = (0, 0, 0)

# ====== 16x16 映射默认：origin=br row=bottom_to_top col=right_to_left serpentine=True ======
MATRIX_W = 16
MATRIX_H = 16
ORIGIN = "br"
ROW_ORDER = "bottom_to_top"
COL_ORDER = "right_to_left"
SERPENTINE = True

# ====== Rain 参数对齐你的运行参数 ======
ROW2_HSV = (0.55, 0.40, 0.30)
TOP_ROW = 0
TOP_HSV = (0.55, 0.40, 0.30)
AUDIO_V_GAIN = 0.70
AUDIO_CADENCE_MIN = 0.65
AUDIO_CADENCE_MAX = 1.30


try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def _parse_start_effect_from_argv() -> int:
    """
    argv:
      (no arg) -> START_EFFECT
      1 -> mode1 -> effect_idx=0
      2 -> mode2 -> effect_idx=1
    """
    if len(sys.argv) >= 2:
        a = (sys.argv[1] or "").strip()
        if a == "mode1":
            return 0
        if a == "mode2":
            return 1
        if a in ("-h", "--help", "help"):
            print("Usage:")
            print(f"  sudo python3 {os.path.basename(sys.argv[0])} 1   # Mode1: SpectrumPush")
            print(f"  sudo python3 {os.path.basename(sys.argv[0])} 2   # Mode2: Rain")
            print(f"  sudo python3 {os.path.basename(sys.argv[0])}     # Default: START_EFFECT={START_EFFECT + 1}")
            raise SystemExit(0)
    return int(START_EFFECT)


# ==========================
# WS2812 over SPI (4-bit encoding) + brightness + gamma
# ==========================
class WS2812_SPI:
    def __init__(self, led_count, bus=10, device=0, spi_speed=3200000, brightness=0.42, reset_us=300):
        import spidev
        self.led_count = int(led_count)
        self.brightness = max(0.0, min(1.0, float(brightness)))

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = int(spi_speed)
        self.spi.mode = 0

        self._lut4 = [self._enc(v) for v in range(256)]
        self._gamma = [int(((i / 255.0) ** 2.2) * 255 + 0.5) for i in range(256)]
        self._bg = [self._gamma[int(max(0, min(255, i * self.brightness)))] for i in range(256)]

        bytes_per_us = self.spi.max_speed_hz / 8.0 / 1_000_000.0
        self.reset_bytes = max(1, int(math.ceil(reset_us * bytes_per_us)))

        self.clear()

    @staticmethod
    def _enc(byte_val: int):
        acc = 0
        for i in range(8):
            bit = (byte_val >> (7 - i)) & 1
            acc = (acc << 4) | (0b1110 if bit else 0b1000)
        return ((acc >> 24) & 0xFF, (acc >> 16) & 0xFF, (acc >> 8) & 0xFF, acc & 0xFF)

    def _apply(self, v: int) -> int:
        if v <= 0:
            return 0
        if v >= 255:
            v = 255
        return self._bg[v]

    def show(self, grb_list):
        if len(grb_list) != self.led_count:
            raise ValueError("LED count mismatch")

        buf = bytearray()
        buf.extend(b"\x00" * self.reset_bytes)

        for g, r, b in grb_list:
            g = self._apply(int(g))
            r = self._apply(int(r))
            b = self._apply(int(b))
            buf.extend(self._lut4[g])
            buf.extend(self._lut4[r])
            buf.extend(self._lut4[b])

        buf.extend(b"\x00" * self.reset_bytes)
        self.spi.xfer2(list(buf))

    def clear(self):
        self.show([(0, 0, 0)] * self.led_count)

    def close(self):
        try:
            self.clear()
        finally:
            self.spi.close()


# ==========================
# Matrix mapping
# ==========================
class Matrix:
    def __init__(self, w=16, h=16,
                 origin='bl',
                 row_order='bottom_to_top',
                 col_order='left_to_right',
                 serpentine=False):
        self.w, self.h = int(w), int(h)
        self.origin = origin
        self.row_order = row_order
        self.col_order = col_order
        self.serpentine = bool(serpentine)

        self._map = [[self.xy_to_index(x, y) for y in range(self.h)] for x in range(self.w)]

    def xy_to_index(self, x, y):
        row = (self.h - 1 - y) if self.row_order == 'bottom_to_top' else y
        col = (self.w - 1 - x) if self.col_order == 'right_to_left' else x

        if self.serpentine and (row % 2 == 1):
            col = (self.w - 1) - col

        if self.origin == 'bl':
            pass
        elif self.origin == 'br':
            col = (self.w - 1) - col
        elif self.origin == 'tl':
            row = (self.h - 1) - row
        elif self.origin == 'tr':
            row = (self.h - 1) - row
            col = (self.w - 1) - col

        return row * self.w + col

    def idx(self, x, y):
        if x < 0:
            x = 0
        elif x >= self.w:
            x = self.w - 1
        if y < 0:
            y = 0
        elif y >= self.h:
            y = self.h - 1
        return self._map[x][y]


# ==========================
# Audio state
# ==========================
@dataclass
class AudioState:
    level: float = 0.0
    level_slow: float = 0.0
    track_name: str = ""
    bands: tuple = None


def _kill_process_group(p: subprocess.Popen):
    if not p:
        return
    try:
        pgid = os.getpgid(p.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            p.terminate()
    except Exception:
        pass

    try:
        p.wait(timeout=0.6)
        return
    except Exception:
        pass

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            p.kill()
    except Exception:
        pass


class MusicPlayer(threading.Thread):
    def __init__(self, music_dir, audio_state: AudioState, lock: threading.Lock, stop_event: threading.Event,
                 sample_rate=44100, channels=2, shuffle_once=True, band_count=16):
        super().__init__(daemon=True)
        self.music_dir = music_dir
        self.audio_state = audio_state
        self.lock = lock
        self.stop_event = stop_event
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.band_count = int(band_count)

        self.shuffle_once = bool(shuffle_once)
        self.playlist = []
        self.idx = 0

        self._skip_event = threading.Event()
        self._skip_dir = 0
        self._skip_lock = threading.Lock()
        self._pl_lock = threading.Lock()

        self.fft_n = 2048
        self._hann = None
        if _HAS_NUMPY:
            self._hann = np.hanning(self.fft_n).astype(np.float32)

        self._band_floor = [1e-6] * self.band_count
        self._band_peak = [1e-3] * self.band_count

    def _peek_target_name(self, direction: int) -> str:
        with self._pl_lock:
            if not self.playlist:
                return ""
            n = len(self.playlist)
            target = (self.idx + direction) % n
            return os.path.basename(self.playlist[target])

    def request_next(self):
        name = self._peek_target_name(+1)
        if name:
            print(f"[INFO] Next: {name}")
        with self._skip_lock:
            self._skip_dir = +1
            self._skip_event.set()

    def request_prev(self):
        name = self._peek_target_name(-1)
        if name:
            print(f"[INFO] Prev: {name}")
        with self._skip_lock:
            self._skip_dir = -1
            self._skip_event.set()

    def _consume_skip(self) -> int:
        with self._skip_lock:
            if not self._skip_event.is_set():
                return 0
            d = self._skip_dir
            self._skip_dir = 0
            self._skip_event.clear()
            return d

    def _list_music(self):
        exts = (".mp3", ".wav")
        files = []
        try:
            for name in os.listdir(self.music_dir):
                p = os.path.join(self.music_dir, name)
                if os.path.isfile(p) and name.lower().endswith(exts):
                    files.append(p)
        except FileNotFoundError:
            return []
        files.sort()
        return files

    def _set_track(self, path):
        base = os.path.basename(path)
        with self.lock:
            self.audio_state.track_name = base
        print(f"[INFO] Now playing: {base}")

    def _update_rms(self, rms_norm):
        with self.lock:
            a = 0.18
            b = 0.03
            self.audio_state.level = (1 - a) * self.audio_state.level + a * rms_norm
            self.audio_state.level_slow = (1 - b) * self.audio_state.level_slow + b * rms_norm

    def _update_bands(self, mono_f32):
        if not _HAS_NUMPY:
            return
        if len(mono_f32) < self.fft_n:
            return

        x = mono_f32[-self.fft_n:]
        x = x - np.mean(x)
        x = x * self._hann

        spec = np.fft.rfft(x)
        mag = np.abs(spec).astype(np.float32)

        freqs = np.fft.rfftfreq(self.fft_n, 1.0 / self.sample_rate)

        f_min, f_max = 60.0, 10000.0
        edges = np.logspace(math.log10(f_min), math.log10(f_max), self.band_count + 1)

        bands = [0.0] * self.band_count
        for i in range(self.band_count):
            f0, f1 = edges[i], edges[i + 1]
            idx0 = int(np.searchsorted(freqs, f0))
            idx1 = int(np.searchsorted(freqs, f1))
            idx0 = max(1, idx0)
            idx1 = max(idx0 + 1, idx1)
            seg = mag[idx0:idx1]
            v = float(np.mean(seg)) if seg.size else 0.0
            bands[i] = v

        out = []
        for i, v in enumerate(bands):
            self._band_floor[i] = 0.995 * self._band_floor[i] + 0.005 * v
            self._band_peak[i] = max(self._band_peak[i] * 0.998, v)
            den = max(1e-6, self._band_peak[i] - self._band_floor[i])
            n = (v - self._band_floor[i]) / den
            n = max(0.0, min(1.0, n))
            n = math.sqrt(n)
            out.append(n)

        with self.lock:
            self.audio_state.bands = tuple(out)

    def _play_one(self, path) -> int:
        self._set_track(path)

        ffmpeg_cmd = [
            "ffmpeg", "-v", "error",
            "-i", path,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", str(self.channels),
            "-ar", str(self.sample_rate),
            "pipe:1"
        ]
        pacat_cmd = [
            "pacat", "--raw",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
            "--format=s16le"
        ]

        ff = None
        pa = None
        skip_dir = 0

        if _HAS_NUMPY:
            mono_buf = np.zeros(self.fft_n * 2, dtype=np.float32)
            mono_ptr = 0

        try:
            ff = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
            pa = subprocess.Popen(pacat_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0, start_new_session=True)

            bytes_per_frame = 2 * self.channels
            chunk_frames = 1024
            chunk_bytes = chunk_frames * bytes_per_frame

            while not self.stop_event.is_set():
                sd = self._consume_skip()
                if sd != 0:
                    skip_dir = sd
                    break

                data = ff.stdout.read(chunk_bytes)
                if not data:
                    break

                sd = self._consume_skip()
                if sd != 0:
                    skip_dir = sd
                    break

                try:
                    pa.stdin.write(data)
                except Exception:
                    break

                samples = array('h')
                samples.frombytes(data)
                if not samples:
                    continue

                ssum = 0.0
                for v in samples:
                    ssum += float(v) * float(v)
                rms = math.sqrt(ssum / len(samples)) / 32768.0
                rms = min(1.0, rms * 3.0)
                rms = math.sqrt(rms)
                self._update_rms(rms)

                if _HAS_NUMPY:
                    if self.channels == 2:
                        a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        a = a.reshape(-1, 2)
                        mono = (a[:, 0] + a[:, 1]) * (0.5 / 32768.0)
                    else:
                        mono = np.frombuffer(data, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)

                    n = mono.shape[0]
                    end = mono_ptr + n
                    if end <= mono_buf.shape[0]:
                        mono_buf[mono_ptr:end] = mono
                    else:
                        k = mono_buf.shape[0] - mono_ptr
                        mono_buf[mono_ptr:] = mono[:k]
                        mono_buf[:(n - k)] = mono[k:]
                    mono_ptr = (mono_ptr + n) % mono_buf.shape[0]

                    if mono_ptr >= self.fft_n:
                        seg = mono_buf[mono_ptr - self.fft_n:mono_ptr]
                    else:
                        seg = np.concatenate([mono_buf[-(self.fft_n - mono_ptr):], mono_buf[:mono_ptr]])
                    self._update_bands(seg)

        finally:
            try:
                if pa and pa.stdin:
                    pa.stdin.close()
            except Exception:
                pass
            try:
                if ff and ff.stdout:
                    ff.stdout.close()
            except Exception:
                pass

            _kill_process_group(ff)
            _kill_process_group(pa)

        return skip_dir

    def run(self):
        pl = self._list_music()
        if not pl:
            with self.lock:
                self.audio_state.track_name = "(no mp3/wav found in music_dir)"
                self.audio_state.bands = tuple([0.0] * self.band_count)
            print("[WARN] No mp3/wav found in music_dir")
            while not self.stop_event.is_set():
                time.sleep(0.5)
            return

        if self.shuffle_once:
            random.shuffle(pl)

        with self._pl_lock:
            self.playlist = pl
            self.idx = 0

        with self.lock:
            self.audio_state.bands = tuple([0.0] * self.band_count)

        while not self.stop_event.is_set():
            with self._pl_lock:
                path = self.playlist[self.idx]

            sd = self._play_one(path)
            if self.stop_event.is_set():
                break

            with self._pl_lock:
                if sd == 0 or sd == +1:
                    self.idx = (self.idx + 1) % len(self.playlist)
                elif sd == -1:
                    self.idx = (self.idx - 1) % len(self.playlist)


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


class EffectBase:
    name = "base"
    def reset(self): pass
    def render_rgb(self, m: Matrix, t: float, level: float, level_slow: float, bands):
        return [(0, 0, 0)] * (m.w * m.h)


class EffectSpectrumPush(EffectBase):
    name = "Mode1: SpectrumPush"

    def __init__(self, m: Matrix):
        self.m = m
        self.cols = m.w
        self.heights = [0.0] * self.cols
        self.peaks = [0.0] * self.cols

        self.attack = 0.55
        self.release = 0.18

        self.push = 0.0
        self.push_attack = 0.35
        self.push_release = 0.08
        self.prev_energy = 0.0

        self.h_bottom = 0.62
        self.h_top = 0.02
        self.v_min = 0.18
        self.v_max = 0.95

    def reset(self):
        self.heights = [0.0] * self.cols
        self.peaks = [0.0] * self.cols
        self.push = 0.0
        self.prev_energy = 0.0

    def _row_hue(self, y_from_bottom, t):
        frac = y_from_bottom / max(1, (self.m.h - 1))
        hue = (self.h_bottom * (1 - frac) + self.h_top * frac)
        hue = (hue + 0.01 * math.sin(t * 0.25)) % 1.0
        return hue

    def render_rgb(self, m: Matrix, t: float, level: float, level_slow: float, bands):
        if not bands or len(bands) != m.w:
            bands = [max(0.0, min(1.0, level * 1.2))] * m.w

        energy = 0.65 * level + 0.35 * (sum(bands) / len(bands))
        dE = max(0.0, energy - self.prev_energy)
        self.prev_energy = 0.92 * self.prev_energy + 0.08 * energy

        target_push = min(1.0, dE * 2.2 + energy * 0.20)
        if target_push > self.push:
            self.push = (1 - self.push_attack) * self.push + self.push_attack * target_push
        else:
            self.push = (1 - self.push_release) * self.push + self.push_release * target_push

        for x in range(m.w):
            b = float(bands[x])
            shape = 0.88 + 0.18 * math.sin((x / max(1, m.w - 1)) * math.pi)
            tgt = (b * 0.92 + self.push * 0.55) * shape
            tgt = min(1.0, tgt * 0.95)

            if tgt > self.heights[x]:
                self.heights[x] = (1 - self.attack) * self.heights[x] + self.attack * tgt
            else:
                self.heights[x] = (1 - self.release) * self.heights[x] + self.release * tgt

            if self.heights[x] > self.peaks[x]:
                self.peaks[x] = self.heights[x]
            else:
                self.peaks[x] *= 0.985

        leds = [(0, 0, 0)] * (m.w * m.h)
        v_base = self.v_min + (self.v_max - self.v_min) * min(1.0, level_slow * 1.15)

        for x in range(m.w):
            height = int(self.heights[x] * m.h + 1e-6)

            for y in range(m.h):
                y_from_bottom = (m.h - 1) - y
                if y_from_bottom < height:
                    hue = self._row_hue(y_from_bottom, t)
                    frac = (y_from_bottom + 1) / m.h
                    v = min(1.0, v_base * (0.78 + 0.45 * frac))
                    r, g, b = hsv_to_rgb(hue, 1.0, v)
                    leds[m.idx(x, y)] = (r, g, b)

            if height > 0:
                cap_y_from_bottom = min(m.h - 1, height - 1)
                cap_y = (m.h - 1) - cap_y_from_bottom
                hue = self._row_hue(cap_y_from_bottom, t)
                r, g, b = hsv_to_rgb(hue, 1.0, min(1.0, v_base * 1.15))
                leds[m.idx(x, cap_y)] = (r, g, b)

        return leds


class EffectRainRow2Start(EffectBase):
    name = "Mode2: Rain (Row2Start + TopRow)"

    def __init__(self, m: Matrix, fps=60,
                 background=(0, 0, 0),
                 row2_hsv=(0.55, 0.40, 0.30),
                 top_row=0,
                 top_hsv=(0.55, 0.40, 0.30),
                 audio_v_gain=0.70,
                 audio_s_gain=0.20,
                 audio_cadence_min=0.65,
                 audio_cadence_max=1.30):
        self.m = m
        self.dt = 1.0 / max(1, int(fps))
        self._rnd = random.Random()

        self.g = 0.85
        self.vmax = 0.95

        self.tail = 2
        self.tail_fade = 0.60
        self.hue = 0.58
        self.sat = 0.88
        self.lane_hue_jitter = 0.015
        self.lane_hue = [ (self.hue + (self._rnd.random() - 0.5) * 2.0 * self.lane_hue_jitter) % 1.0
                         for _ in range(self.m.w) ]

        self.row2_hsv = row2_hsv
        self.top_row = top_row
        self.top_hsv = top_hsv

        self.audio_v_gain = float(audio_v_gain)
        self.audio_s_gain = float(audio_s_gain)
        self.audio_cadence_min = float(audio_cadence_min)
        self.audio_cadence_max = float(audio_cadence_max)

        self.background = (int(background[0]), int(background[1]), int(background[2]))

        cadence_base = 0.9
        cadence_jitter = 0.15
        self.next_t = [0.0] * self.m.w
        self.cadence_nominal = []
        for _ in range(self.m.w):
            j = 1.0 + (self._rnd.random() * 2.0 - 1.0) * cadence_jitter
            self.cadence_nominal.append(max(0.15, cadence_base * j))

        self.reset()

    def reset(self):
        self.drops = []
        self.t0 = time.time()

    def _maybe_spawn(self, now, level_slow):
        k = (self.audio_cadence_max - self.audio_cadence_min)
        cad_scale = self.audio_cadence_max - k * max(0.0, min(1.0, level_slow))
        for lane in range(self.m.w):
            if now >= self.next_t[lane]:
                self.drops.append({"lane": lane, "y": 1.0 - 1e-3, "vy": 0.0})
                self.next_t[lane] = now + self.cadence_nominal[lane] * cad_scale

    def _apply_background_to_off_pixels(self, leds_rgb):
        r0, g0, b0 = self.background
        out = list(leds_rgb)
        for i, (r, g, b) in enumerate(out):
            if r == 0 and g == 0 and b == 0:
                out[i] = (r0, g0, b0)
        return out

    def render_rgb(self, m: Matrix, t: float, level: float, level_slow: float, bands):
        now = time.time() - self.t0
        self._maybe_spawn(now, level_slow)

        for d in self.drops:
            d["vy"] = min(self.vmax, d["vy"] + self.g * self.dt)
            d["y"] = d["y"] + d["vy"]

        self.drops = [d for d in self.drops if d["y"] < m.h + 0.5]

        leds = [(0, 0, 0)] * (m.w * m.h)

        if self.top_row is not None:
            r, g, b = hsv_to_rgb(*self.top_hsv)
            y = int(self.top_row)
            if y < 0:
                y = 0
            if y >= m.h:
                y = m.h - 1
            for x in range(m.w):
                leds[m.idx(x, y)] = (r, g, b)

        lane_active = [False] * m.w
        for d in self.drops:
            lane_active[d["lane"]] = True

        for d in self.drops:
            lane = d["lane"]
            base_h = self.lane_hue[lane]
            y = d["y"]
            yi = int(math.floor(y))
            frac = y - yi

            depth_shift = 0.08 * min(1.0, max(0.0, y / max(1.0, m.h)))
            h = (base_h + depth_shift) % 1.0

            for k in range(self.tail + 1):
                yy = yi - k
                if yy < 1:
                    break

                base = 1.0 if k == 0 else (self.tail_fade ** k)
                head_mix = (0.45 + 0.55 * (1.0 - frac)) if k == 0 else 1.0
                v0 = base * (0.58 + 0.42 * head_mix)

                v = v0 * (1.0 - self.audio_v_gain + self.audio_v_gain * level)
                sat0 = self.sat * (1.05 if k == 0 else 0.95)
                sat = min(1.0, sat0 * (1.0 - self.audio_s_gain + self.audio_s_gain * level_slow))

                r, g, b = hsv_to_rgb(h, sat, min(1.0, v))
                idx = m.idx(lane, yy)
                ra, ga, ba = leds[idx]
                leds[idx] = (min(255, ra + r), min(255, ga + g), min(255, ba + b))

        r2, g2, b2 = hsv_to_rgb(*self.row2_hsv)
        for lane in range(m.w):
            if not lane_active[lane]:
                leds[m.idx(lane, 1)] = (r2, g2, b2)

        return self._apply_background_to_off_pixels(leds)


class DisplayLoop(threading.Thread):
    def __init__(self, strip: WS2812_SPI, m: Matrix,
                 audio_state: AudioState, lock: threading.Lock, stop_event: threading.Event,
                 fps=60, start_effect=0, bg_color=(0, 0, 0)):
        super().__init__(daemon=True)
        self.strip = strip
        self.m = m
        self.audio_state = audio_state
        self.lock = lock
        self.stop_event = stop_event

        self.fps = int(fps)
        self.dt = 1.0 / max(1, self.fps)

        self.effects = [
            EffectSpectrumPush(m),  # 模式1
            EffectRainRow2Start(     # 模式2
                m, fps=self.fps,
                background=bg_color,
                row2_hsv=ROW2_HSV,
                top_row=TOP_ROW,
                top_hsv=TOP_HSV,
                audio_v_gain=AUDIO_V_GAIN,
                audio_cadence_min=AUDIO_CADENCE_MIN,
                audio_cadence_max=AUDIO_CADENCE_MAX,
            ),
        ]
        self.effect_idx = int(start_effect) % len(self.effects)

        if not _HAS_NUMPY:
            print("[WARN] numpy not found -> FFT spectrum disabled (Mode1 will degrade). Install: sudo apt-get install -y python3-numpy")
        print(f"[INFO] Start mode: {self.effects[self.effect_idx].name}")
        print("[INFO] Keys: n->NextSong  u->PrevSong  q->quit  Ctrl+C->quit")

    def run(self):
        t0 = time.time()
        while not self.stop_event.is_set():
            t = time.time() - t0

            with self.lock:
                level = float(self.audio_state.level)
                level_slow = float(self.audio_state.level_slow)
                bands = self.audio_state.bands

            eff = self.effects[self.effect_idx]

            leds_rgb = eff.render_rgb(self.m, t, level, level_slow, bands)
            grb = [(g, r, b) for (r, g, b) in leds_rgb]

            try:
                self.strip.show(grb)
            except Exception:
                time.sleep(0.2)

            time.sleep(self.dt)


class TTYKeyReader(threading.Thread):
    """
    只保留 n/u/q 控制（不再支持 1/2 切模式）
    """
    def __init__(self, player: MusicPlayer, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.player = player
        self.stop_event = stop_event
        self.tty_f = None
        self.old_term = None

    def _setup_tty(self):
        import termios
        import tty
        self.tty_f = open("/dev/tty", "rb", buffering=0)
        fd = self.tty_f.fileno()
        self.old_term = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def _restore_tty(self):
        if not self.tty_f:
            return
        try:
            import termios
            termios.tcsetattr(self.tty_f.fileno(), termios.TCSADRAIN, self.old_term)
        except Exception:
            pass
        try:
            self.tty_f.close()
        except Exception:
            pass
        self.tty_f = None

    def run(self):
        try:
            self._setup_tty()
        except Exception:
            print("[WARN] Cannot open /dev/tty. Keyboard control disabled.")
            return

        try:
            while not self.stop_event.is_set():
                r, _, _ = select.select([self.tty_f], [], [], 0.1)
                if not r:
                    continue
                ch = self.tty_f.read(1)
                if not ch:
                    continue
                c = ch.decode("utf-8", errors="ignore").lower()

                if c == "n":
                    self.player.request_next()
                elif c == "u":
                    self.player.request_prev()
                elif c == "q":
                    self.stop_event.set()
        finally:
            self._restore_tty()


def main():
    stop_event = threading.Event()
    lock = threading.Lock()
    audio_state = AudioState()

    # 通过启动参数选择模式
    start_effect = _parse_start_effect_from_argv()

    def handle_sig(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    m = Matrix(
        MATRIX_W, MATRIX_H,
        origin=ORIGIN,
        row_order=ROW_ORDER,
        col_order=COL_ORDER,
        serpentine=SERPENTINE
    )

    strip = WS2812_SPI(
        led_count=m.w * m.h,
        bus=SPI_BUS, device=SPI_DEV,
        spi_speed=SPI_SPEED_HZ,
        brightness=BRIGHTNESS,
        reset_us=RESET_US
    )

    player = MusicPlayer(
        MUSIC_DIR, audio_state, lock, stop_event,
        sample_rate=SAMPLE_RATE, channels=CHANNELS,
        shuffle_once=SHUFFLE_PLAYLIST,
        band_count=m.w
    )
    player.start()

    display = DisplayLoop(
        strip, m, audio_state, lock, stop_event,
        fps=FPS,
        start_effect=start_effect,
        bg_color=BG_COLOR,
    )
    display.start()

    # 只保留 n/u/q
    key_reader = TTYKeyReader(player, stop_event)
    key_reader.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        stop_event.set()
        time.sleep(0.2)
        try:
            strip.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
