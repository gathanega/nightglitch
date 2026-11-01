"""
NightGlitch - real-time GPU "glitch poster" effect for your webcam.

A person silhouette (MediaPipe selfie segmentation) is rendered almost pure white
with clustered glitch blocks over a blue low-poly background. Hand gestures
(open hand -> fist) trigger a pixel-mosaic transition between camera and effect.

Install:
    pip install -r requirements.txt

Run:
    python nightglitch.py
    python nightglitch.py --width 960 --mp-skip 3
    python nightglitch.py --test      # no MediaPipe, uses an animated dummy mask
    python nightglitch.py --auto      # cycle through the looks automatically

Keys:
    q / ESC   quit              SPACE  trigger mosaic transition
    m         switch mask       l      switch look (POSTER/CITY/DOTS/GLITCH)
    [ ]       silhouette light/dark threshold
    , .       fewer / more glitch blocks
    r         start/stop mp4 recording      f  fullscreen

Gesture:
    OPEN hand, then FIST  ->  mosaic transition (camera <-> effect).

GLITCH look (all on the GPU):
    1. Person     : near-white silhouette, thin navy contour, soft white glow.
    2. Glitch     : small streak blocks (blue / magenta / gray-blue / sky) that
                    change ~8 fps, clustered in noise patches.
    3. Background : blue low-poly facets, two layers of shards, stars, glowing
                    isometric cubes, wireframe net, confetti, navy vignette.
    4. Transition : pixel mosaic around the wave front; every block picks
                    camera / effect / pastel block.
"""
from __future__ import annotations

import argparse
import math
import threading
import time
from datetime import datetime

import cv2
import glfw
import moderngl
import numpy as np

try:
    import mediapipe as mp
    HAVE_MP = True
except Exception:  # MediaPipe missing or incompatible -> fall back to test mask
    HAVE_MP = False

APP_NAME = "NightGlitch"

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CW, CH = 640, 360            # camera capture size
MW, MH = 384, 216            # segmentation input / mask size
LOOK_SECONDS = 14.0          # time per look when --auto is on
BLEND_SECONDS = 2.0          # cross-fade between looks
CARD_SECONDS = 3.5           # time per text card (old looks)
WAVE_SPEED = 750.0           # transition wave speed (px/s at 720p)
WAVE_BAND = 260.0            # mosaic band width (px at 720p)

LOOK_NAMES = ["POSTER", "CITY", "DOTS", "GLITCH"]
NLOOKS = len(LOOK_NAMES)
LOOK_GLITCH = 3

WORDS = [
    ("HELLO", "wave your hand", "01", "STEP"),
    ("PINK", "paper and ink / two colours", "02", "INK"),
    ("MOVE", "make a fist / clear the page", "03", "GRIP"),
    ("POSTER", "everything is a layer", "04", "LOOK"),
    ("DOTS", "halftone / grain / noise", "05", "TONE"),
]


# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip) landmark ids


def classify_hand(lm: np.ndarray) -> str:
    """Return OPEN / FIST / POINT / OTHER from 21 hand landmarks."""
    wrist = lm[0]
    extended = [
        np.linalg.norm(lm[tip] - wrist) > np.linalg.norm(lm[pip] - wrist) * 1.15
        for tip, pip in FINGERS
    ]
    n = sum(extended)
    if n == 4:
        return "OPEN"
    if n == 0:
        return "FIST"
    if n == 1 and extended[0]:
        return "POINT"
    return "OTHER"


class Grip:
    """State machine: UNARMED -> ARMED (open hand) -> HELD (fist, fires once)."""

    def __init__(self) -> None:
        self.state = 0

    def update(self, gesture: str | None) -> bool:
        fired = False
        if self.state == 0 and gesture == "OPEN":
            self.state = 1
        elif self.state == 1 and gesture == "FIST":
            self.state, fired = 2, True
        elif self.state == 2 and gesture in ("OPEN", None):
            self.state = 0
        return fired


# --------------------------------------------------------------------------
# Vision (segmentation + hands) and camera worker thread
# --------------------------------------------------------------------------
class Vision:
    def __init__(self, use_test: bool = False) -> None:
        self.use_test = use_test or not HAVE_MP
        self.hands = None
        self.seg = None
        self.t0 = time.time()
        if not self.use_test:
            self._init_mp()

    def _init_mp(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def toggle(self) -> None:
        """Switch between MediaPipe and the dummy test mask."""
        if HAVE_MP:
            self.use_test = not self.use_test
            if not self.use_test and self.hands is None:
                self._init_mp()

    def test_mask(self) -> np.ndarray:
        """Animated head + body ellipses used when MediaPipe is unavailable."""
        t = time.time() - self.t0
        m = np.zeros((MH, MW), np.uint8)
        cx = int(MW / 2 + math.sin(t * 0.7) * MW * 0.18)
        cv2.ellipse(m, (cx, int(MH * 0.78)), (int(MW * 0.17), int(MH * 0.38)), 0, 0, 360, 255, -1)
        cv2.circle(m, (cx, int(MH * 0.30)), int(MH * 0.13), 255, -1)
        return m.astype(np.float32) / 255.0

    def process(self, small_bgr: np.ndarray):
        """Return (mask float32 0..1, [(gesture, (palm_x, palm_y)), ...])."""
        if self.use_test or self.seg is None:
            return self.test_mask(), []

        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mask = np.clip(self.seg.process(rgb).segmentation_mask, 0, 1).astype(np.float32)

        hands_out = []
        result = self.hands.process(rgb)
        if result.multi_hand_landmarks:
            for i, hl in enumerate(result.multi_hand_landmarks):
                src = (result.multi_hand_world_landmarks[i].landmark
                       if result.multi_hand_world_landmarks else hl.landmark)
                pts = np.array([[p.x, p.y, p.z] for p in src], np.float32)
                palm = hl.landmark[9]
                hands_out.append((classify_hand(pts), (palm.x, palm.y)))
        return mask, hands_out


class Worker(threading.Thread):
    """Captures frames and runs vision off the render thread."""

    def __init__(self, cam: int, vision: Vision, skip: int) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CW)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CH)
        self.vision = vision
        self.skip = max(1, skip)
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.alive = True

    def run(self) -> None:
        n, smooth, hands = 0, None, []
        while self.alive:
            ok, frame = self.cap.read()
            if not ok:  # camera hiccup: show a dark frame instead of crashing
                frame = np.full((CH, CW, 3), 40, np.uint8)
                time.sleep(0.02)
            frame = cv2.flip(cv2.resize(frame, (CW, CH)), 1)

            if n % self.skip == 0:
                small = cv2.resize(frame, (MW, MH), interpolation=cv2.INTER_AREA)
                mask, hands = self.vision.process(small)
                smooth = mask if smooth is None else smooth * 0.5 + mask * 0.5
            n += 1

            with self.lock:
                self.latest = (frame, (smooth * 255).astype(np.uint8), hands)
                self.seq += 1

    def get(self):
        with self.lock:
            return self.seq, self.latest

    def stop(self) -> None:
        self.alive = False
        time.sleep(0.05)
        self.cap.release()


# --------------------------------------------------------------------------
# Text card (used by the POSTER / CITY / DOTS looks)
# --------------------------------------------------------------------------
def build_text(card: tuple[str, str, str, str]) -> bytes:
    """Two-channel text image: R = ink, G = knock-out gap around the letters."""
    big, note, num, tag = card
    out = np.zeros((720, 1280), np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 7.0 if len(big) <= 5 else 5.0

    (_, th), _ = cv2.getTextSize(big, font, scale, 14)
    cv2.putText(out, big, (60, 150 + th // 2), font, scale, 255, 14, cv2.LINE_AA)
    for i, line in enumerate(note.split(" / ")):
        cv2.putText(out, line, (70, 720 - 90 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    cv2.putText(out, num, (1280 - 220, 720 - 60), font, 3.0, 255, 6, cv2.LINE_AA)
    cv2.putText(out, " ".join(tag), (1280 - 360, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)

    gap = cv2.dilate(out, np.ones((9, 9), np.uint8))
    return np.dstack([out, gap]).tobytes()


# --------------------------------------------------------------------------
# GLSL shaders
# --------------------------------------------------------------------------
VERT = """
#version 330
in vec2 in_pos;
out vec2 vUV;
void main(){
    vUV = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uCam, uMask, uText;
uniform vec2  uRes;
uniform float uTime, uBlend, uScale, uOffset, uWaveR, uThr, uGlitch, uBand;
uniform int   uLookA, uLookB, uWaveActive, uWaveOld, uWaveTarget, uCleared;
uniform vec2  uWaveC;

// ---- palettes -------------------------------------------------------------
const vec3 PAPER = vec3(255.0, 240.0, 245.0) / 255.0;
const vec3 INK   = vec3(220.0,  20.0, 120.0) / 255.0;

const vec3 NAVY  = vec3(0.05, 0.07, 0.40);
const vec3 BLUD  = vec3(0.04, 0.16, 0.85);
const vec3 BLUM  = vec3(0.20, 0.45, 1.00);
const vec3 SKY   = vec3(0.55, 0.78, 1.00);
const vec3 PALE  = vec3(0.86, 0.93, 1.00);
const vec3 MAG   = vec3(1.00, 0.25, 0.75);
const vec3 GRAYB = vec3(0.55, 0.60, 0.75);

// ---- helpers --------------------------------------------------------------
float h11(float n){ return fract(sin(n * 127.1) * 43758.5453); }
float maskAt(vec2 uv){ return smoothstep(0.35, 0.65, texture(uMask, uv).r); }
float lumAt(vec2 u){ return dot(texture(uCam, u).bgr, vec3(0.299, 0.587, 0.114)); }

float vnoise(vec2 x){
    vec2 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float a = h11(dot(i,                vec2(12.9, 78.2)));
    float b = h11(dot(i + vec2(1.0, 0.0), vec2(12.9, 78.2)));
    float c = h11(dot(i + vec2(0.0, 1.0), vec2(12.9, 78.2)));
    float d = h11(dot(i + vec2(1.0, 1.0), vec2(12.9, 78.2)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// ---- classic looks (POSTER / CITY / DOTS) ---------------------------------
float halftone(float dark, float cell, float ang, vec2 p){
    float c = cos(ang), s = sin(ang);
    float cs = cell * uScale;
    vec2 q = vec2(p.x * c + p.y * s, -p.x * s + p.y * c) / cs;
    vec2 f = fract(q) - 0.5;
    float r = sqrt(clamp(dark, 0.0, 1.0)) * 0.72;
    return clamp((r - length(f)) * cs, 0.0, 1.0);
}

float city(vec2 p){
    float cw = 46.0 * uScale;
    float id = floor(p.x / cw);
    float h  = (0.15 + 0.55 * h11(id + 1.0)) * uRes.y;
    float sp = 8.0 + h11(id + 9.0) * 30.0;
    float top = uRes.y - h + sin(uTime * 0.4 + id) * 20.0 * uScale
              + mod(uTime * sp, 40.0) * uScale;
    float lx = mod(p.x, cw);
    float body = step(top, p.y) * step(4.0 * uScale, lx);
    if (h11(id + 5.0) > 0.5) {
        float t = 3.0 * uScale;
        float edge = max(1.0 - step(top + t, p.y),
                         max(1.0 - step(t + 4.0 * uScale, lx), step(cw - t, lx)));
        body *= clamp(edge, 0.0, 1.0);
    }
    return body;
}

float dots(vec2 p){
    float cell = 30.0 * uScale;
    vec2 g = floor(p / cell);
    float rnd = h11(g.x * 13.0 + g.y * 71.0);
    vec2 c = (g + 0.5) * cell
           + vec2(cos(uTime * 0.5 + rnd * 6.28), sin(uTime * 0.4 + rnd * 6.28)) * cell * 0.15;
    float r = (0.08 + 0.2 * rnd) * cell;
    return clamp(r - distance(p, c), 0.0, 1.0);
}

float look(int id, vec2 p, float dark, float m, float edge){
    if (id == 0) {
        float sil = m * halftone(dark * 1.2, 9.0, 0.5, p);
        float bg  = halftone(0.12, 22.0, 0.0, p) * (1.0 - m) * 0.6;
        return clamp(sil + edge + bg, 0.0, 1.0);
    } else if (id == 1) {
        float sil = halftone(dark * 1.1, 7.0, 0.0, p) * m;
        return clamp(city(p) * (1.0 - m) + sil, 0.0, 1.0);
    }
    float sil = halftone(0.8, 12.0, 0.3, p) * m;
    return clamp(dots(p) * (1.0 - m) + sil, 0.0, 1.0);
}

// ---- GLITCH look ----------------------------------------------------------
vec3 shpal(float x){
    int i = int(mod(floor(x), 9.0));
    if (i == 0) return BLUD;
    if (i == 1) return BLUM;
    if (i == 2) return SKY;
    if (i == 3) return PALE;
    if (i == 4) return BLUD;
    if (i == 5) return MAG;
    if (i == 6) return NAVY;
    if (i == 7) return SKY;
    return vec3(1.0);
}

// Blue low-poly facets (voronoi, flat shading per cell).
vec3 bgBase(vec2 p){
    vec2 q = p / (300.0 * uScale);
    vec2 ip = floor(q), fp = fract(q);
    float md = 8.0;
    vec2 best = ip;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 cid = ip + g;
            vec2 o = vec2(h11(dot(cid, vec2(12.9, 78.2))), h11(dot(cid, vec2(39.3, 11.1))));
            o = 0.5 + 0.4 * sin(uTime * 0.25 + 6.2831853 * o);
            vec2 r = g + o - fp;
            float d = dot(r, r);
            if (d < md) { md = d; best = cid; }
        }
    }
    float hh = h11(dot(best, vec2(7.7, 13.3)));
    vec3 col = mix(BLUD, SKY, hh);
    col = mix(col, PALE, 0.55 * smoothstep(0.65, 1.0, h11(hh * 91.0 + 2.0)));
    col = mix(col, PALE, 0.25 * (1.0 - p.x / uRes.x));   // left brighter, right deeper
    return col;
}

// Floating shard, three variants per cell:
//   glowing isometric cube | small spiky star | two-tone polygon
vec4 shard(vec2 p, float cs, float seed){
    vec2 g = floor(p / cs);
    float h1 = h11(g.x * 17.0 + g.y * 91.0 + seed);
    if (h1 < 0.5) return vec4(0.0);
    float h2 = h11(h1 * 311.0 + 1.7), h3 = h11(h1 * 57.0 + 9.1);
    vec2 c = (g + 0.5) * cs + (vec2(h2, h3) - 0.5) * cs * 0.15
           + vec2(sin(uTime * 0.6 + h2 * 6.28), cos(uTime * 0.5 + h3 * 6.28)) * cs * 0.05;
    float rad = (0.12 + 0.22 * h3) * cs;
    float ang = uTime * (h2 - 0.5) * 0.8 + h1 * 6.28;
    float kind = h11(h1 * 23.7 + seed * 3.1);
    vec2 q = p - c;

    // -- glowing isometric cube (does not rotate, keeps the isometric angle)
    if (kind < 0.12) {
        vec2 u = vec2(q.x, -q.y);
        float R = rad * 1.1;
        float hexd = max(abs(u.x), max(abs(dot(u, vec2(0.5, 0.8660254))),
                                       abs(dot(u, vec2(-0.5, 0.8660254))))) - 0.8660254 * R;
        if (hexd > 0.6 * R) return vec4(0.0);
        if (hexd > 0.0) {                                   // halo outside the cube
            float gl = exp(-hexd / (0.22 * R)) * 0.55 * (1.0 - smoothstep(0.35 * R, 0.6 * R, hexd));
            return vec4(mix(SKY, vec3(1.0), 0.35), gl);
        }
        float t = mod(atan(u.y, u.x) - 0.5235988 + 6.2831853, 6.2831853);
        int face = int(floor(t / 2.0943951));               // 0 top, 1 left, 2 right
        vec3 col = (face == 0) ? vec3(0.93, 0.97, 1.0) : (face == 1) ? BLUM : BLUD;
        col = mix(col, vec3(1.0), 0.25 * (1.0 - length(u) / R));
        float e = 9.0;                                      // distance to the 3 inner edges
        for (int k = 0; k < 3; k++) {
            float th = 0.5235988 + float(k) * 2.0943951;
            vec2 dr = vec2(cos(th), sin(th));
            if (dot(u, dr) > 0.0) e = min(e, abs(u.x * dr.y - u.y * dr.x));
        }
        float rim = min(e, -hexd);
        float ln = 1.0 - smoothstep(0.0, 1.6 * uScale, rim);
        col = mix(col, vec3(1.0), ln * 0.9);
        return vec4(col, clamp(-hexd / (1.5 * uScale), 0.0, 1.0));
    }

    float cc = cos(ang), ss = sin(ang);
    q = vec2(q.x * cc + q.y * ss, -q.x * ss + q.y * cc);

    // -- small spiky star (8 spikes)
    if (kind < 0.28) {
        float a = atan(q.x, q.y) + 3.14159265;
        float sp = abs(cos(4.0 * a));
        float rs = rad * 0.6;                               // star size
        float r = rs * (0.16 + 0.84 * pow(sp, 4.0));
        float d = (length(q) - r) * 0.6;
        if (d > 0.0) return vec4(0.0);
        vec3 col = mix(shpal(h1 * 9.0 + seed), vec3(1.0), 0.55);
        col = (q.x + q.y * 0.4 > 0.0) ? col : col * 0.85;
        return vec4(col, clamp(-d / (1.2 * uScale), 0.0, 1.0));
    }

    // -- two-tone triangle / square
    float n = (h3 > 0.5) ? 3.0 : 4.0;
    float r = 6.2831853 / n;
    float a = atan(q.x, q.y) + 3.14159265;
    float d = cos(floor(0.5 + a / r) * r - a) * length(q) - rad;
    if (d > 0.0) return vec4(0.0);
    vec3 c1 = shpal(h1 * 9.0 + seed), c2 = shpal(h2 * 9.0 + 4.0) * 0.8;
    vec3 col = (q.x + q.y * 0.4 > 0.0) ? c1 : c2;
    return vec4(col, clamp(-d / (1.5 * uScale), 0.0, 1.0));
}

float seg(vec2 p, vec2 a, vec2 b, float cs){
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    float d = length(pa - ba * h);
    float fade = 1.0 - smoothstep(0.8 * cs, 1.6 * cs, length(ba));
    return (1.0 - smoothstep(0.0, 1.4 * uScale, d)) * fade;
}

// Wireframe net of dots + lines (3x3 point neighbourhood).
float net(vec2 p){
    float cs = 150.0 * uScale;
    vec2 id = floor(p / cs);
    vec2 pts[9];
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = id + vec2(float(i), float(j));
            float a = h11(dot(g, vec2(12.9, 78.2)));
            float b = h11(dot(g, vec2(39.3, 11.1)));
            pts[(j + 1) * 3 + (i + 1)] =
                (g + 0.5 + 0.35 * vec2(sin(uTime * 0.5 + a * 6.28), cos(uTime * 0.4 + b * 6.28))) * cs;
        }
    }
    float t = 0.0;
    for (int i = 0; i < 9; i++) {
        if (i == 4) continue;
        t += seg(p, pts[4], pts[i], cs);
    }
    t += seg(p, pts[1], pts[3], cs) + seg(p, pts[1], pts[5], cs)
       + seg(p, pts[7], pts[3], cs) + seg(p, pts[7], pts[5], cs);
    for (int i = 0; i < 9; i++)
        t += 1.0 - smoothstep(2.0 * uScale, 4.5 * uScale, distance(p, pts[i]));
    return clamp(t, 0.0, 1.0);
}

vec4 confetti(vec2 p){
    float cell = 22.0 * uScale;
    vec2 g = floor(p / cell);
    float h = h11(dot(g, vec2(12.9, 78.2)) + 7.0);
    if (h > 0.10) return vec4(0.0);
    vec2 c = (g + 0.5) * cell + (vec2(h11(h * 91.0), h11(h * 53.0)) - 0.5) * cell * 0.5
           + vec2(sin(uTime + h * 40.0), cos(uTime * 0.8 + h * 30.0)) * cell * 0.15;
    float s = (1.5 + 3.0 * h11(h * 17.0)) * uScale;
    vec2 d = abs(p - c);
    if (max(d.x, d.y) > s) return vec4(0.0);
    float k = h11(h * 131.0);
    vec3 col = (k < 0.40) ? vec3(1.0) : (k < 0.65) ? MAG : (k < 0.85) ? SKY : BLUM;
    return vec4(col, 1.0);
}

// Glitch blocks: 3 sizes, horizontal streaks, re-rolled ~8 times per second.
vec4 glitch(vec2 p, float dens){
    float tq = mod(floor(uTime * 8.0), 97.0);
    vec4 res = vec4(0.0);
    for (int l = 0; l < 3; l++) {
        float fl = float(l);
        float bh = 5.0 * exp2(fl) * uScale;
        float row = floor(p.y / bh);
        float hr = h11(row * 7.13 + fl * 31.7 + tq * 0.37);
        float stretch = 1.0 + floor(hr * 4.0) * 1.5;
        float bw = bh * stretch;
        float colId = floor(p.x / bw + hr * 10.0);
        float hc = h11(colId * 13.37 + row * 91.7 + fl * 5.3 + tq * 1.7);
        if (hc < dens / (1.0 + fl)) {
            float hk = h11(hc * 977.0 + 3.1);
            vec3 c = (hk < 0.30) ? BLUM : (hk < 0.55) ? MAG : (hk < 0.80) ? GRAYB
                   : (hk < 0.92) ? SKY : BLUD;
            res = vec4(c, 1.0);
        }
    }
    return res;
}

vec3 glitchLook(vec2 uv, vec2 p, float m){
    // Thin navy contour at mask = 0.5, constant width (~2px).
    float mr = texture(uMask, uv).r;
    float fw = max(fwidth(mr), 0.0001) * 0.9;
    float line = 1.0 - smoothstep(0.0, fw, abs(mr - 0.5));
    float lum = lumAt(uv);

    vec3 col = vec3(0.0);
    if (m < 0.995) {                                   // background only outside the silhouette
        col = bgBase(p);
        vec2 drift = vec2(uTime * 5.0, uTime * 2.0) * uScale;
        vec4 s = shard(p + drift * 0.5, 400.0 * uScale, 1.0);
        col = mix(col, s.rgb, s.a);
        float nm = smoothstep(0.50, 0.72, vnoise(p / (520.0 * uScale) + vec2(uTime * 0.02, 3.0)));
        col = mix(col, vec3(0.80, 0.94, 1.0), net(p) * nm * 0.9);
        s = shard(p + drift, 170.0 * uScale, 7.0);
        col = mix(col, s.rgb, s.a);
        s = confetti(p);
        col = mix(col, s.rgb, s.a);

        // Dark navy vignette (top-left corner).
        float vg = smoothstep(0.65, 0.0, length(uv * vec2(1.0, 1.6)));
        col = mix(col, NAVY * 0.45, vg * 0.75);

        // Soft white glow around the silhouette.
        float halo = 0.0;
        for (int k = 0; k < 8; k++) {
            float a = float(k) * 0.7853982;
            vec2 d = vec2(cos(a), sin(a));
            halo += textureLod(uMask, uv + d * 12.0 * uScale / uRes, 0.0).r;
            halo += textureLod(uMask, uv + d * 26.0 * uScale / uRes, 0.0).r;
        }
        halo /= 16.0;
        col = mix(col, vec3(0.88, 0.95, 1.0), halo * 0.55 * (1.0 - m));
    }

    // Person: almost pure white, faint cool tint in the shadows.
    float l = clamp((lum - uThr) * 2.0 + 0.5, 0.0, 1.0);
    vec3 pc = mix(vec3(0.90, 0.93, 1.0), vec3(1.0), 0.65 + 0.35 * l);

    // Glitch only in clustered noise patches, not everywhere.
    float patch = smoothstep(0.52, 0.78,
                  vnoise(p / (170.0 * uScale) + vec2(0.0, floor(uTime * 8.0) * 0.9)));
    float dens = uGlitch * 3.2 * patch + 0.30 * float(uWaveActive);
    vec4 g = glitch(p, dens);
    pc = mix(pc, g.rgb, g.a);
    col = mix(col, pc, m);

    col = mix(col, NAVY, line * 0.9);
    return clamp(col, 0.0, 1.0);
}

vec3 lookColor(int id, vec2 uv, vec2 p, float dark, float m, float edge){
    if (id == 3) return glitchLook(uv, p, m);

    float ink = look(id, p, dark, m, edge);
    vec2 t = texture(uText, uv - vec2(uOffset, 0.0)).rg;
    ink = clamp(ink * (1.0 - t.g) + t.r, 0.0, 1.0);

    float sy = mod(uTime * 90.0 * uScale, uRes.y);       // scan line
    if (p.y > sy && p.y < sy + 3.0 * uScale) ink = clamp(ink + 0.5, 0.0, 1.0);
    return mix(PAPER, INK, ink);
}

void main(){
    vec2 uv = vUV;
    vec2 p  = uv * uRes;
    vec3 cam = texture(uCam, uv).bgr;
    float dark = 1.0 - dot(cam, vec3(0.299, 0.587, 0.114));
    float m = maskAt(uv);

    vec2 o = vec2(2.0) / uRes;
    float edge = clamp((abs(maskAt(uv + vec2(o.x, 0.0)) - maskAt(uv - vec2(o.x, 0.0)))
                      + abs(maskAt(uv + vec2(0.0, o.y)) - maskAt(uv - vec2(0.0, o.y)))) * 2.0,
                       0.0, 1.0);

    vec3 poster = lookColor(uLookB, uv, p, dark, m, edge);
    if (uBlend < 1.0)
        poster = mix(lookColor(uLookA, uv, p, dark, m, edge), poster, uBlend);

    vec3 outc = (uCleared == 1) ? cam : poster;

    if (uWaveActive == 1) {
        // Mosaic transition: each block picks camera / effect / pastel block.
        float bs = 28.0 * uScale;
        vec2 cell = floor(p / bs);
        vec2 cc = (cell + 0.5) * bs;
        float dc = distance(cc, uWaveC) - uWaveR;         // > 0: wave has not reached it yet
        float W = uBand * uScale;
        float t = clamp(0.5 - 0.5 * dc / W, 0.0, 1.0);    // 0 = old state, 1 = new state
        float h  = h11(dot(cell, vec2(12.9, 78.2)));
        float h2 = h11(h * 91.7 + 3.3);
        int s = (h < t) ? uWaveTarget : uWaveOld;
        outc = (s == 1) ? cam : poster;

        float pm = 0.40 * (1.0 - abs(2.0 * t - 1.0));     // most pastel blocks mid-band
        if (h2 < pm) {
            vec3 sc = texture(uCam, cc / uRes).bgr;
            vec3 mc = mix(sc, vec3(0.80, 0.86, 1.0), 0.45);
            float hk = h11(h * 57.3 + 1.1);
            if (hk < 0.18)      mc = vec3(0.95, 0.35, 0.70);
            else if (hk < 0.30) mc = vec3(0.95, 0.85, 0.55);
            else if (hk < 0.45) mc = vec3(0.30, 0.45, 0.95);
            outc = mc;
        }
    }
    fragColor = vec4(outc, 1.0);
}
"""


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{APP_NAME} - real-time GPU glitch poster")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=1280, help="window width (16:9)")
    ap.add_argument("--mp-skip", type=int, default=2, help="run MediaPipe every N frames")
    ap.add_argument("--test", action="store_true", help="use dummy mask, no MediaPipe")
    ap.add_argument("--auto", action="store_true", help="switch looks automatically")
    return ap.parse_args()


def init_window(width: int):
    if not glfw.init():
        raise SystemExit("glfw failed to init")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(width, width * 9 // 16, APP_NAME, None, None)
    if not win:
        glfw.terminate()
        raise SystemExit("window could not be created (OpenGL 3.3 required)")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    return win


def main() -> None:
    args = parse_args()
    win = init_window(args.width)

    ctx = moderngl.create_context()
    print("GPU:", ctx.info.get("GL_RENDERER"))

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])

    def set_uniform(name: str, value) -> None:
        # Unused uniforms are optimised away by the GLSL compiler -> skip silently.
        if name in prog:
            prog[name].value = value

    tex_cam = ctx.texture((CW, CH), 3, dtype="f1")
    tex_mask = ctx.texture((MW, MH), 1, dtype="f1")
    tex_text = ctx.texture((1280, 720), 2, dtype="f1")
    for tex in (tex_cam, tex_mask, tex_text):
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = False
    tex_mask.write(np.zeros((MH, MW), np.uint8).tobytes())
    set_uniform("uCam", 0)
    set_uniform("uMask", 1)
    set_uniform("uText", 2)

    vision = Vision(args.test)
    worker = Worker(args.cam, vision, args.mp_skip)
    worker.start()

    state = dict(
        look=LOOK_GLITCH, look_t=time.time(),
        card=0, card_t=time.time(),
        cleared=False, wave=None, fire_center=None,
        writer=None, rec_size=None, full=False,
        thr=0.40,   # silhouette light/dark threshold
        gl=0.08,    # glitch density
    )
    grips = [Grip(), Grip()]
    tex_text.write(build_text(WORDS[0]))

    def on_key(w, key, scancode, action, mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        # Keys that auto-repeat when held.
        if key == glfw.KEY_LEFT_BRACKET:
            state["thr"] = max(0.1, state["thr"] - 0.02)
        elif key == glfw.KEY_RIGHT_BRACKET:
            state["thr"] = min(0.9, state["thr"] + 0.02)
        elif key == glfw.KEY_COMMA:
            state["gl"] = max(0.0, state["gl"] - 0.01)
        elif key == glfw.KEY_PERIOD:
            state["gl"] = min(0.5, state["gl"] + 0.01)

        # One-shot keys.
        elif action != glfw.PRESS:
            return
        elif key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(w, True)
        elif key == glfw.KEY_SPACE:
            state["fire_center"] = (0.5, 0.5)
        elif key == glfw.KEY_M:
            vision.toggle()
        elif key == glfw.KEY_L:
            state["look"] = (state["look"] + 1) % NLOOKS
            state["look_t"] = time.time()
        elif key == glfw.KEY_F:
            toggle_fullscreen(w)
        elif key == glfw.KEY_R:
            toggle_recording(w)

    def toggle_fullscreen(w) -> None:
        state["full"] = not state["full"]
        if state["full"]:
            mon = glfw.get_primary_monitor()
            vm = glfw.get_video_mode(mon)
            glfw.set_window_monitor(w, mon, 0, 0, vm.size.width, vm.size.height, vm.refresh_rate)
        else:
            glfw.set_window_monitor(w, None, 100, 100, args.width, args.width * 9 // 16, 0)

    def toggle_recording(w) -> None:
        if state["writer"] is None:
            fbw, fbh = glfw.get_framebuffer_size(w)
            name = datetime.now().strftime("nightglitch_%Y%m%d_%H%M%S.mp4")
            state["writer"] = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (fbw, fbh))
            state["rec_size"] = (fbw, fbh)
            print("recording:", name)
        else:
            state["writer"].release()
            state["writer"] = None
            print("recording stopped")

    glfw.set_key_callback(win, on_key)

    last_seq, last_t = -1, time.time()
    frames, fps_t = 0, time.time()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = time.time()
        dt, last_t = now - last_t, now
        fbw, fbh = glfw.get_framebuffer_size(win)
        if fbw == 0 or fbh == 0:  # minimised
            continue
        scale = fbh / 720.0

        # ---- new camera frame / mask / gestures -------------------------
        seq, data = worker.get()
        if data is not None and seq != last_seq:
            last_seq = seq
            frame, mask8, hands = data
            tex_cam.write(frame.tobytes())
            tex_mask.write(mask8.tobytes())
            if hands:
                for i, (gesture, (px, py)) in enumerate(hands[:2]):
                    if grips[i].update(gesture):
                        state["fire_center"] = (px, py)
            else:
                for grip in grips:
                    grip.update(None)

        # ---- start a transition wave ------------------------------------
        if state["fire_center"] is not None:
            cx, cy = state["fire_center"]
            state["fire_center"] = None
            # r starts at -band so no block has changed yet.
            state["wave"] = dict(c=(cx, cy), r=-WAVE_BAND * scale,
                                 old=state["cleared"], target=not state["cleared"])

        # ---- look / card timers -----------------------------------------
        if args.auto and now - state["look_t"] > LOOK_SECONDS:
            state["look_t"] = now
            state["look"] = (state["look"] + 1) % NLOOKS
        if now - state["card_t"] > CARD_SECONDS:
            state["card_t"] = now
            state["card"] = (state["card"] + 1) % len(WORDS)
            tex_text.write(build_text(WORDS[state["card"]]))
        blend = min(1.0, (now - state["look_t"]) / BLEND_SECONDS)

        # ---- advance the wave -------------------------------------------
        wave = state["wave"]
        if wave is not None:
            wave["r"] += WAVE_SPEED * scale * dt
            cxp, cyp = wave["c"][0] * fbw, wave["c"][1] * fbh
            farthest = max(math.hypot(cxp - x, cyp - y) for x in (0, fbw) for y in (0, fbh))
            if wave["r"] > farthest + WAVE_BAND * scale:  # band passed all corners
                state["cleared"] = wave["target"]
                state["wave"] = wave = None

        # ---- uniforms + draw --------------------------------------------
        ctx.viewport = (0, 0, fbw, fbh)
        set_uniform("uRes", (float(fbw), float(fbh)))
        set_uniform("uTime", now % 10000.0)
        set_uniform("uScale", scale)
        set_uniform("uBlend", blend)
        set_uniform("uThr", state["thr"])
        set_uniform("uGlitch", state["gl"])
        set_uniform("uBand", WAVE_BAND)
        set_uniform("uLookA", (state["look"] - 1) % NLOOKS)
        set_uniform("uLookB", state["look"])
        set_uniform("uOffset", 0.02 * math.sin((now - state["card_t"]) * 0.5))
        set_uniform("uCleared", int(state["cleared"]))
        set_uniform("uWaveActive", 1 if wave else 0)
        if wave:
            set_uniform("uWaveC", (wave["c"][0] * fbw, wave["c"][1] * fbh))
            set_uniform("uWaveR", float(wave["r"]))
            set_uniform("uWaveOld", int(wave["old"]))
            set_uniform("uWaveTarget", int(wave["target"]))

        tex_cam.use(0)
        tex_mask.use(1)
        tex_text.use(2)
        ctx.clear(0, 0, 0)
        vao.render(moderngl.TRIANGLE_STRIP)

        # ---- recording ---------------------------------------------------
        if state["writer"] is not None:
            raw = ctx.screen.read(components=3)
            img = np.frombuffer(raw, np.uint8).reshape(fbh, fbw, 3)
            img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)
            if (fbw, fbh) == state["rec_size"]:
                state["writer"].write(img)

        glfw.swap_buffers(win)

        # ---- title bar stats ---------------------------------------------
        frames += 1
        if now - fps_t >= 1.0:
            glfw.set_window_title(
                win,
                f"{APP_NAME} | {frames} fps | {LOOK_NAMES[state['look']]} | "
                f"thr {state['thr']:.2f} | glitch {state['gl']:.2f} | "
                f"mask: {'test' if vision.use_test else 'mediapipe'}",
            )
            frames, fps_t = 0, now

    worker.stop()
    if state["writer"] is not None:
        state["writer"].release()
    glfw.terminate()


if __name__ == "__main__":
    main()
