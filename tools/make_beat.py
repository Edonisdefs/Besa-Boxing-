import numpy as np, wave, sys

SR = 44100
BPM = 140
BEAT = 60 / BPM
STEP = BEAT / 4  # 16th
BAR = BEAT * 4
rng = np.random.default_rng(7)

N_BARS = 32
total = int((N_BARS * BAR + 3) * SR)
L = np.zeros(total)
R = np.zeros(total)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, t, pan=0.0, gain=1.0):
    i = int(t * SR)
    if i >= total:
        return
    n = min(len(sig), total - i)
    l = gain * np.cos((pan + 1) * np.pi / 4)
    r = gain * np.sin((pan + 1) * np.pi / 4)
    L[i:i + n] += sig[:n] * l
    R[i:i + n] += sig[:n] * r


def fft_filter(x, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if lo:
        m *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** 4)
    if hi:
        m *= 1 / (1 + (f / hi) ** 4)
    return np.fft.irfft(X * m, len(x))


def tt(d):
    return np.arange(int(d * SR)) / SR


# ---------- instruments ----------
def kick():
    t = tt(0.45)
    f = 45 + 110 * np.exp(-t * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7)
    click = fft_filter(rng.standard_normal(len(t)), lo=2000) * np.exp(-t * 300) * 0.25
    return np.tanh((s + click) * 1.6) * 0.95


def snare():
    t = tt(0.35)
    n = fft_filter(rng.standard_normal(len(t)), lo=1200, hi=9000) * np.exp(-t * 16)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 28) * 0.5
    return np.tanh((n * 1.2 + tone) * 1.5) * 0.8


def clap():
    t = tt(0.4)
    env = np.zeros_like(t)
    for off in (0, 0.011, 0.023):
        env += np.where(t >= off, np.exp(-(t - off) * 90), 0) * 0.6
    env += np.exp(-t * 14) * 0.5
    n = fft_filter(rng.standard_normal(len(t)), lo=900, hi=7000)
    return n * env * 0.8


def hat(open_=False, vel=1.0):
    d = 0.28 if open_ else 0.06
    t = tt(d)
    n = fft_filter(rng.standard_normal(len(t)), lo=7000)
    return n * np.exp(-t * (14 if open_ else 80)) * 0.35 * vel


def eight08(note, dur, glide_from=None):
    t = tt(dur)
    f0 = hz(note)
    f = np.full_like(t, f0)
    if glide_from:
        f = f0 + (hz(glide_from) - f0) * np.exp(-t * 25)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.25 * np.sin(2 * ph)
    env = np.minimum(t / 0.004, 1) * np.exp(-t * 2.2)
    rel = np.minimum((dur - t) / 0.05, 1)
    return np.tanh(s * env * rel * 2.2) * 0.9


def bell(note, dur=1.2, vel=1.0):
    t = tt(dur)
    f = hz(note)
    mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t * 6)
    s = np.sin(2 * np.pi * f * t + mod) + 0.35 * np.sin(2 * np.pi * f * 2 * t)
    return s * np.exp(-t * 3.2) * 0.35 * vel


def pad(notes, dur):
    t = tt(dur)
    out = np.zeros_like(t)
    for m in notes:
        for det in (-0.12, 0.0, 0.12):
            f = hz(m + det)
            out += 2 * ((f * t) % 1) - 1  # saw
    out = fft_filter(out, hi=1100)
    env = np.minimum(t / 0.8, 1) * np.minimum((dur - t) / 0.6, 1)
    return out * env * 0.05


def gong():
    t = tt(4.0)
    out = np.zeros_like(t)
    for f, a in [(180, 1), (270, .8), (415, .6), (587, .5), (913, .4), (1283, .25)]:
        out += a * np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * 3 * t) * .5) * np.exp(-t * (1.0 + f / 900))
    return out * 0.18


def ring_bell():
    # klassische Boxring-Glocke: schnelle Dreier-Schlaege
    t = tt(1.4)
    out = np.zeros_like(t)
    for hit in (0, 0.16, 0.32):
        tl = t - hit
        m = tl >= 0
        for f, a in [(1180, 1), (1770, .6), (2650, .4), (3320, .25)]:
            out[m] += a * np.sin(2 * np.pi * f * tl[m]) * np.exp(-tl[m] * 6)
    return out * 0.22


def riser(dur):
    t = tt(dur)
    n = rng.standard_normal(len(t))
    sw = np.zeros_like(n)
    # grobe Sweep-Filterung in Bloecken
    blk = 2048
    for i in range(0, len(n) - blk, blk):
        c = 300 + (i / len(n)) ** 2 * 9000
        sw[i:i + blk] = fft_filter(n[i:i + blk], lo=c * 0.5, hi=c * 1.5)
    return sw * (t / dur) ** 2 * 0.5


KICK, SNARE, CLAP = kick(), snare(), clap()

# ---------- Komposition (f-Moll) ----------
# Progression: Fm | Db | Ab | Eb   (je ein Takt)
ROOT808 = [29, 37, 32, 39]          # F1.. (midi 29=F1, 37=Db2, 32=Ab1, 39=Eb2)
CHORDS = [[53, 56, 60], [49, 53, 56], [56, 60, 63], [51, 55, 58]]
# Motiv (Step-Position im 4-Takt-Block -> Note), dunkle, hymnische Glocken-Linie
MOTIF = {
    0: (77, 3), 3: (80, 1), 4: (84, 3), 8: (82, 2), 10: (80, 2), 12: (77, 4),
    16: (80, 3), 19: (77, 1), 20: (73, 3), 24: (75, 2), 26: (77, 2), 28: (72, 4),
    32: (77, 3), 35: (80, 1), 36: (84, 3), 40: (87, 2), 42: (84, 2), 44: (82, 4),
    48: (80, 2), 50: (82, 2), 52: (79, 3), 56: (75, 2), 58: (77, 2), 60: (80, 4),
}

# Sektionen (in Takten): 0-3 Intro, 4-11 Hook A, 12-15 Break, 16-27 Drop, 28-31 Outro
def section(bar):
    if bar < 4: return "intro"
    if bar < 12: return "A"
    if bar < 16: return "break"
    if bar < 28: return "drop"
    return "outro"

add(ring_bell(), 0.0, gain=1.0)
add(gong(), 0.3, gain=0.8)

for bar in range(N_BARS):
    sec = section(bar)
    t0 = bar * BAR
    ci = bar % 4
    # Pad
    if sec != "outro" or bar < 30:
        add(pad(CHORDS[ci], BAR + 0.3), t0, pan=0.0, gain=1.0 if sec in ("intro", "break") else 0.7)

    # Riser vor Drop und Hook
    if bar in (3, 15):
        add(riser(BAR), t0, gain=0.8)

    for s in range(16):
        t = t0 + s * STEP
        sw = 0.0 if s % 2 == 0 else STEP * 0.12  # leichter Swing
        t += sw

        if sec in ("A", "drop"):
            # Kicks: Boom-Bap/Trap-Muster mit Variation
            kpat = [0, 7, 10] if bar % 2 == 0 else [0, 6, 10, 14]
            if sec == "drop" and bar % 4 == 3:
                kpat = [0, 3, 6, 10, 12, 14]
            if s in kpat:
                add(KICK, t, gain=0.95)
            # Snare/Clap auf 3 (Halftime)
            if s == 8:
                add(SNARE, t, gain=0.85)
                add(CLAP, t, gain=0.7)
            if sec == "drop" and bar % 4 == 3 and s in (13, 14, 15):
                add(SNARE, t, gain=0.5 + 0.1 * (s - 12))
            # Hi-Hats mit Rolls
            if s % 2 == 0:
                add(hat(vel=1.0 if s % 4 == 0 else 0.6), t, pan=0.25)
            else:
                add(hat(vel=0.35), t, pan=-0.25)
            if sec == "drop" and s in (6, 7, 14, 15):
                add(hat(vel=0.55), t + STEP / 2, pan=0.3)
            if s == 15 and bar % 2 == 1:
                add(hat(open_=True), t + STEP * 0.5, pan=0.3)
        elif sec == "break":
            if s == 0 and bar == 12:
                add(KICK, t, gain=0.9)
            if s % 4 == 0 and bar >= 14:
                add(KICK, t, gain=0.5 + 0.1 * (bar - 14))
        elif sec == "outro":
            if bar < 30 and s in (0, 10):
                add(KICK, t, gain=0.8)
            if bar < 30 and s == 8:
                add(SNARE, t, gain=0.6)

    # 808-Bass
    if sec in ("A", "drop"):
        rn = ROOT808[ci]
        add(eight08(rn, BAR * 0.45), t0, gain=0.95)
        add(eight08(rn, BAR * 0.2, glide_from=rn + 12), t0 + 10 * STEP + 0.02, gain=0.8)
        if bar % 4 == 3:
            add(eight08(rn + 7, STEP * 3), t0 + 13 * STEP, gain=0.7)
    elif sec == "outro" and bar < 30:
        add(eight08(ROOT808[ci], BAR * 0.5), t0, gain=0.85)

# Melodie (Glocken) – pro 4-Takt-Block
for blk in range(N_BARS // 4):
    b0 = blk * 4
    sec = section(b0)
    if sec in ("A", "drop") or (sec == "break" and blk == 3):
        for st, (note, dur16) in MOTIF.items():
            t = b0 * BAR + st * STEP
            add(bell(note, dur=1.4, vel=1.0 if sec == "drop" else 0.75), t, pan=-0.2)
            if sec == "drop":
                add(bell(note - 12, dur=1.4, vel=0.5), t + 0.01, pan=0.2)

# Letzter Schlag + Gong
end_t = N_BARS * BAR - BAR * 0.5
add(KICK, end_t, gain=1.0)
add(gong(), end_t, gain=1.0)
add(ring_bell(), end_t + 0.1, gain=0.9)

# ---------- Mixdown ----------
# Hall (einfach, Mehrfach-Delay)
def reverb(x, amt=0.18):
    out = x.copy()
    for d, g in [(0.029, .5), (0.043, .4), (0.067, .3), (0.097, .22), (0.151, .15)]:
        k = int(d * SR)
        out[k:] += x[:-k] * g * amt * 3
    return out

end = int((N_BARS * BAR + 3.0) * SR)
L = reverb(L)[:end]
R = reverb(np.roll(R, 37))[:end]

# Sidechain-artiges leichtes Ducking ueberspringen; weicher Limiter
peak = max(np.abs(L).max(), np.abs(R).max())
L, R = L / peak * 1.4, R / peak * 1.4
L, R = np.tanh(L), np.tanh(R)
fade = int(2.5 * SR)
L[-fade:] *= np.linspace(1, 0, fade)
R[-fade:] *= np.linspace(1, 0, fade)
stereo = np.stack([L, R], axis=1)
stereo = stereo / np.abs(stereo).max() * 0.92
pcm = (stereo * 32767).astype(np.int16)

out = sys.argv[1]
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("seconds:", len(pcm) / SR, "peak:", np.abs(stereo).max())
