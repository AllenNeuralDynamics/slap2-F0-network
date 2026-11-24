"""Architecture figure of the F0 network (the deep architecture, S4's deep arm): inputs,
branches, fusion, outputs, what the output at (t, r, c) sees, and how it is trained.

usage: python architecture_figure.py [OUT.svg]   (default docs/figures/network_architecture.svg)

The numbers are written in by hand from code/slap2_f0/model.py and the deep arm's
training config (S4_deep2: --fuse_blocks 4 --branch_mlp 1 --pointwise_init zero_out);
update them here when the architecture changes. Box lines starting with "+ " (the
pointwise blocks new in the deep architecture) are drawn in bold, without the "+ ". A PNG for viewing:
    chrome --headless=new --hide-scrollbars --window-size=1600,1040 --screenshot=OUT.png file:///.../OUT.svg
"""
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "docs" / "figures" / "network_architecture.svg")
W, H = 1600, 1040
FONT = "Helvetica, Arial, sans-serif"

# palette: one hue per kind of context (fills light, strokes dark)
C = {
    "blue": ("#DCE8F7", "#2F5F9E"), "green": ("#DDF1E4", "#2C7A4B"), "orange": ("#FBE6CF", "#B5651D"),
    "purple": ("#ECE4F6", "#6B44A0"), "gray": ("#F2F3F5", "#7A808A"), "slate": ("#E8ECF1", "#34495E"),
    "red": ("#F9D9D5", "#B03A2E"), "ink": "#1F2933", "muted": "#52606D",
}
out = []


def add(s):
    out.append(s)


def text(x, y, s, size=12, weight="normal", fill=None, anchor="start", italic=False, rotate=None):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill or C["ink"]}" '
        f'text-anchor="{anchor}"{st}{tr}>{escape(s)}</text>')


def rect(x, y, w, h, fill, stroke, sw=1.2, rx=6, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{o} '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=None, sw=1.4, arrow=False, dash=None):
    m = ' marker-end="url(#arr)"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke or C["muted"]}" '
        f'stroke-width="{sw}"{m}{d}/>')


def box(x, y, w, h, color, title, lines, right=None, dash=None, size=11.2, step=15.2):
    f, s = C[color]
    rect(x, y, w, h, f, s, 1.4, 7, dash)
    text(x + 12, y + 20, title, 13, "bold", s)
    if right:
        text(x + w - 12, y + 20, right, 12, "bold", s, "end")
    for i, ln in enumerate(lines):
        new = ln.startswith("+ ")  # new in the deep architecture: bold, in the box's color
        text(x + 12, y + 38 + i * step, ln[2:] if new else ln, size, "bold" if new else "normal", s if new else C["ink"])


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{C["muted"]}"/></marker>'
    '<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    f'<rect width="5" height="5" fill="{C["red"][0]}"/><line x1="0" y1="0" x2="0" y2="5" stroke="{C["red"][1]}" '
    'stroke-width="1.6"/></pattern></defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ---------------------------------------------------------------- title
text(20, 34, "SLAP2 F0 network: the deep architecture (S4 deep arm, the default from now on)", 21, "bold")
text(20, 56, "One forward pass predicts the baseline F0 and the noise variance σ² at every pixel and frame. "
     "The output at (t, r, c) never sees the movie within ±8 frames and ±4 px of itself, so events there cannot leak into it.",
     12.5, fill=C["muted"])
text(20, 74, "Deep: pointwise (1×1×1) residual blocks after every dilated block and after the fusion (bold, new) add nonlinear "
     "processing of the same window; they see one voxel's features, so the reach and the blind spot are unchanged.",
     12.5, fill=C["muted"])
for x, s in ((20, "1  Inputs"), (350, "2  Branches: each sees a different part of the window"),
             (815, "3  Fusion and outputs"), (1175, "4  What the output at (t, r, c) sees")):
    text(x, 100, s, 15, "bold")

# ---------------------------------------------------------------- 1 inputs
fx, fy, s, dx, dy = 40, 162, 118, 112, -46  # front face, side, depth offset (time)
fill, stroke = "#F4F6F8", "#5D6B7A"
add(f'<path d="M{fx},{fy} L{fx + dx},{fy + dy} L{fx + s + dx},{fy + dy} L{fx + s},{fy} z" fill="#E9EDF1" stroke="{stroke}"/>')
add(f'<path d="M{fx + s},{fy} L{fx + s + dx},{fy + dy} L{fx + s + dx},{fy + s + dy} L{fx + s},{fy + s} z" fill="#DDE3E9" stroke="{stroke}"/>')
rect(fx, fy, s, s, fill, stroke, 1.2, 0)
# loss region: central 192 of 718 frames, central 48 of 94 px
a0, a1, q = 263 / 718, 455 / 718, s * 48 / 94
ix, iy = fx + (s - q) / 2, fy + (s - q) / 2
for a in (a0, a1):
    rect(ix + a * dx, iy + a * dy, q, q, "none", C["blue"][1], 1.3, 0, "4 3")
for cx, cy in ((ix, iy), (ix + q, iy), (ix, iy + q), (ix + q, iy + q)):
    line(cx + a0 * dx, cy + a0 * dy, cx + a1 * dx, cy + a1 * dy, C["blue"][1], 1.1, dash="4 3")
text(fx + s / 2, fy + s + 18, "94 px", 11, anchor="middle", fill=C["muted"])
text(fx - 8, fy + s / 2, "94 px", 11, anchor="middle", fill=C["muted"], rotate=-90)
text(fx + s + dx / 2 + 10, fy + s + dy / 2 + 14, "time: 718 frames", 11, fill=C["muted"], rotate=-22)
text(20, 322, "Training window: 718 frames × 94 × 94 px", 12.2, "bold")
text(20, 339, "loss on the central 192 frames × 48 × 48 px (dashed)", 11.2, fill=C["muted"])
text(20, 354, "prediction tiles whole trials the same way", 11.2, fill=C["muted"])

text(20, 388, "5 input channels at every voxel", 12.5, "bold")
chans = ["movie ÷ trial scale (missing = 0)", "validity flag (imaged pixel, kept frame)",
         "mean image M ÷ scale (whole trial)", "log variance factor (SiLo's, ÷ trial median)",
         "log (variance map ÷ g · M): events add variance"]
for i, c in enumerate(chans):
    y = 412 + i * 23
    add(f'<circle cx="30" cy="{y - 4}" r="8" fill="{C["slate"][0]}" stroke="{C["slate"][1]}"/>')
    text(30, y, str(i + 1), 10.5, "bold", C["slate"][1], "middle")
    text(46, y, c, 11.3)
for i, ln in enumerate(["Channels 3–5 are whole-trial maps, constant in", "time: each frame enters with weight 1/T,",
                        "a negligible leak through the blind spot.", "",
                        "Per trial: scale (median brightness) and", "photon gain g (variance per count)."]):
    text(20, 542 + i * 15.5, ln, 11, fill=C["muted"])

# input bus
bus_in = 330
line(300, 270, bus_in, 270, sw=1.6)
line(bus_in, 160, bus_in, 735, sw=1.6)

# ---------------------------------------------------------------- 2 branches
bx, bw = 350, 425
boxes = [
    (112, 119, "blue", "Past branch", "32 ch",
     ["1×1×1 conv 5 → 32, then 7 residual (ReZero) blocks, each:",
      "per-frame 3×3 conv, dilation 1, 1, 1, 1, 2, 2, 4 (blocks 3–7 at ½ res)",
      "and a causal temporal conv, kernel 3, dilation 1, 2, 4, …, 64",
      "+ then a pointwise block: x + α · W₂ GELU(W₁ x), 32 → 64 → 32",
      "then shifted by 9 frames: sees frames t−263 … t−9, ±23 px"], None),
    (241, 44, "blue", "Future branch: the same on reversed time (t+9 … t+263)", "32 ch", [], None),
    (295, 134, "green", "Half-plane branches ×4 (one trunk per direction)", "4 × 24 ch",
     ["1×1×1 conv 5 → 24, then 5 blocks: a 2×3 conv that sees rows r−d … r",
      "only (dilation 1, 1, 1, 2, 4; blocks 3–5 at ½ res) and a symmetric",
      "temporal conv, dilation 1, 2, 4, 8, 16; then shifted down 5 rows.",
      "+ After each block a pointwise block, 24 → 48 → 24",
      "Rotated 0, 90, 180, 270°, the four see every pixel outside the box,",
      "up to 22 px away, in frames t−31 … t+31: the output frame too"], None),
    (443, 89, "orange", "Pooled same-frame features", "8 ch",
     ["fractional deviation Σ(movie − M) / Σ M over squares of half-width",
      "8, 16, 32 px and the whole window, each minus the box, in frame t",
      "and over frames t−3 … t+4: shared brightness changes, even fast ones"], None),
    (546, 104, "purple", "The pixel's own long window, and its whole-trial maps", "6 ch",
     ["mean W̄ and variance of the movie at (r, c) over 9 ≤ |dt| ≤ 262:",
      "log (W̄ / M), log (variance / g W̄), share of the window with data;",
      "and log M, log variance ratio, log variance factor at (r, c).",
      "Enter at the output only: inside a branch they would reach hidden frames"], None),
    (664, 74, "gray", "Motion features (v10b, optional)", "12 ch",
     ["per pool square, the surround fitted as a M − d · ∇M + b · ∇²M;",
      "the change that d and b make at the pixel, and its own ∇M / M, ∇²M / M"], "5 4"),
]
bus_out = 795
for y, h, col, title, right, lines, dash in boxes:
    box(bx, y, bw, h, col, title, lines, right, dash)
    line(bus_in, y + h / 2, bx - 2, y + h / 2, arrow=True)
    line(bx + bw, y + h / 2, bus_out, y + h / 2, sw=1.6, dash="5 4" if dash else None)
line(bus_out, 140, bus_out, 701, sw=1.6)

# ---------------------------------------------------------------- 3 fusion and outputs
fx0, fw = 815, 330
line(bus_out, 150, fx0 - 2, 150, arrow=True)
box(fx0, 120, fw, 60, "slate", "Concatenate at every voxel", ["174 channels (186 with motion features)"])
line(fx0 + fw / 2, 180, fx0 + fw / 2, 200, arrow=True)
box(fx0, 202, fw, 76, "slate", "Pointwise fusion",
    ["1×1×1 conv 174 → 64, GELU", "1×1×1 conv 64 → 32, GELU"])
line(fx0 + fw / 2, 278, fx0 + fw / 2, 298, arrow=True)
box(fx0, 300, fw, 100, "slate", "4 pointwise residual blocks",
    ["+ each x + α · W₂ GELU(W₁ x), 1×1×1, 32 → 64 → 32",
     "combine the branches, window statistics and",
     "same-frame context nonlinearly: 10 pointwise",
     "layers after the concatenation instead of 2"], "new")
hw, gap = 98, 18
hx = [fx0 + i * (hw + gap) for i in range(3)]
for x, (nm, sub) in zip(hx, (("mix m", "→ base"), ("correction c", "→ F0"), ("noise v", "→ σ²"))):
    line(fx0 + fw / 2, 400, x + hw / 2, 426, arrow=True)
    rect(x, 428, hw, 46, C["slate"][0], C["slate"][1], 1.3, 6)
    text(x + hw / 2, 446, nm, 11.8, "bold", C["slate"][1], "middle")
    text(x + hw / 2, 463, "1×1×1, 32 → 1", 10.2, fill=C["muted"], anchor="middle")
    line(x + hw / 2, 474, x + hw / 2, 496, arrow=True)
rect(fx0, 498, fw, 118, "#FFFFFF", C["slate"][1], 1.3, 7)
eq = [("base = M + tanh(m) · (W̄ − M)", "between the mean image and the window mean"),
      ("F0 = base · exp(0.7 · tanh(c / 0.7))", "a correction of ×0.5 … ×2 at most"),
      ("σ² = (g · F0 + floor) · exp(3 · tanh(v / 3))", "photon noise, corrected")]
for i, (e, note) in enumerate(eq):
    text(fx0 + 12, 520 + i * 34, e, 12.6, "bold")
    text(fx0 + 12, 536 + i * 34, note, 10.6, fill=C["muted"])
line(fx0 + fw / 2, 616, fx0 + fw / 2, 636, arrow=True)
rect(fx0, 638, fw, 80, C["slate"][1], C["slate"][1], 1.3, 7)
text(fx0 + fw / 2, 662, "Outputs at every pixel and frame", 13.5, "bold", "#FFFFFF", "middle")
text(fx0 + fw / 2, 683, "F0 (t, r, c): baseline, counts", 12.4, fill="#FFFFFF", anchor="middle")
text(fx0 + fw / 2, 702, "σ² (t, r, c): noise variance, counts²", 12.4, fill="#FFFFFF", anchor="middle")
text(fx0, 740, "Heads start at 0: the untrained network predicts F0 = M.", 11, fill=C["muted"])
text(fx0, 756, "Pointwise blocks start as the identity (α = 1, W₂ = 0; α is learned).", 11, fill=C["muted"])
text(fx0, 772, "432,524 parameters (309,510 without the pointwise blocks).", 11, fill=C["muted"])

# ---------------------------------------------------------------- 4 what the output sees
px0, pw = 1215, 355  # plot x range


def tr_panel(y0, h, t_rng, r_rng, title, ticks_t, ticks_r):
    """Time (x) by row (y) slice through the output's column; t_rng, r_rng half-ranges."""
    cx, cy = px0 + pw / 2, y0 + h / 2
    sx, sy = pw / (2 * t_rng), h / (2 * r_rng)
    X = lambda t: cx + t * sx  # noqa: E731
    Y = lambda r: cy - r * sy  # noqa: E731
    add(f'<clipPath id="clip{int(y0)}"><rect x="{px0}" y="{y0}" width="{pw}" height="{h}"/></clipPath>')
    rect(px0, y0, pw, h, "#F7F8FA", "#B8C0CA", 1, 0)
    add(f'<g clip-path="url(#clip{int(y0)})">')

    def reg(t0, t1, r0, r1, col, op=1.0):
        rect(X(t0), Y(r1), X(t1) - X(t0), Y(r0) - Y(r1), C[col][0], C[col][1], 0.9, 0, opacity=op)
    reg(-263, -9, -23, 23, "blue")
    reg(9, 263, -23, 23, "blue")
    for r0, r1 in ((5, 22), (-22, -5)):
        reg(-31, 31, r0, r1, "green", 0.92)
    for r0, r1 in ((5, 47), (-47, -5)):
        reg(-3, 4, r0, r1, "orange")
    for t0, t1 in ((-262, -9), (9, 262)):
        add(f'<rect x="{X(t0):.1f}" y="{Y(0.5):.1f}" width="{X(t1) - X(t0):.1f}" height="{max(Y(-0.5) - Y(0.5), 2.2):.1f}" '
            f'fill="{C["purple"][1]}"/>')
    add('</g>')
    rect(X(-8), Y(4), X(8) - X(-8), Y(-4) - Y(4), "url(#hatch)", C["red"][1], 1.4, 0)
    add(f'<circle cx="{X(0):.1f}" cy="{Y(0):.1f}" r="2.6" fill="{C["ink"]}"/>')
    for t, lab in ticks_t:
        line(X(t), y0 + h, X(t), y0 + h + 4, "#8A94A0", 1)
        text(X(t), y0 + h + 16, lab, 10.2, fill=C["muted"], anchor="middle")
    for r, lab in ticks_r:
        line(px0 - 4, Y(r), px0, Y(r), "#8A94A0", 1)
        text(px0 - 7, Y(r) + 3.5, lab, 10.2, fill=C["muted"], anchor="end")
    text(px0, y0 - 8, title, 12, "bold")
    text(px0 - 36, cy, "row offset (px)", 10.2, fill=C["muted"], anchor="middle", rotate=-90)
    return X, Y


tr_panel(132, 132, 263, 47, "a  Time × row through the output's column: the whole window",
         [(-263, "t−263"), (-128, "−128"), (0, "t"), (128, "+128"), (263, "t+263")],
         [(47, "47"), (23, "23"), (0, "0"), (-23, "−23"), (-47, "−47")])
text(px0 + pw / 2, 298, "frames", 10.2, fill=C["muted"], anchor="middle")
X, Y = tr_panel(334, 178, 40, 26, "b  Zoom on the output: frames t−40 … t+40, rows ±26",
                [(-40, "−40"), (-31, "−31"), (-9, "−9"), (0, "t"), (9, "+9"), (31, "+31"), (40, "+40")],
                [(22, "22"), (5, "5"), (0, "0"), (-5, "−5"), (-22, "−22")])
text(px0 + pw / 2, 546, "frames", 10.2, fill=C["muted"], anchor="middle")
# annotate the zoom
rect(X(-8) - 122, Y(4) - 47, 120, 32, "#FFFFFF", "#FFFFFF", 0, 3, opacity=0.9)
text(X(-8) - 4, Y(4) - 34, "hidden box", 10.4, "bold", C["red"][1], "end")
text(X(-8) - 4, Y(4) - 21, "|dt| ≤ 8, |dr|, |dc| ≤ 4", 10.2, fill=C["red"][1], anchor="end")
line(X(-8) - 2, Y(4) - 18, X(-5), Y(4) - 1, C["red"][1], 1)

# c: rows x columns at frame t
cs, cyc = 176, 592  # square side, top
ccx = px0 + cs / 2
sxy = cs / 94
P = lambda r, c: (ccx + c * sxy, cyc + cs / 2 - r * sxy)  # noqa: E731
text(px0, cyc - 8, "c  Rows × columns in frame t (no past or future branch here)", 12, "bold")
x0, y0 = P(47, -47)
rect(x0, y0, cs, cs, "#F7F8FA", "#B8C0CA", 1, 0)
x1, y1 = P(22, -22)
rect(x1, y1, 44 * sxy, 44 * sxy, C["green"][0], C["green"][1], 1, 0)
for hwid in (8, 16, 32):
    xa, ya = P(hwid, -hwid)
    rect(xa, ya, 2 * hwid * sxy, 2 * hwid * sxy, "none", C["orange"][1], 1.3, 0, "4 3")
xb, yb = P(4, -4)
rect(xb, yb, 8 * sxy, 8 * sxy, "url(#hatch)", C["red"][1], 1.4, 0)
cx_, cy_ = P(0, 0)
add(f'<circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="2.4" fill="{C["ink"]}"/>')
for (dxa, dya), lab in (((0, -1), "↑"), ((0, 1), "↓"), ((-1, 0), "←"), ((1, 0), "→")):
    text(cx_ + dxa * 30, cy_ + dya * 30 + 4, lab, 13, "bold", C["green"][1], "middle")
xr = x0 + cs + 14
for i, ln in enumerate([("green", "four half-plane views ↑ ↓ ← →"), ("green", "together: every pixel outside"),
                        ("green", "the box, up to 22 px away"), ("orange", "dashed: pool squares of half-"),
                        ("orange", "width 8, 16, 32 px; also the"), ("orange", "whole window, minus the box")]):
    text(xr, cyc + 30 + i * 16 + (8 if i > 2 else 0), ln[1], 10.4, fill=C[ln[0]][1])

# legend
ly = 792
items = [("blue", "past and future branches"), ("green", "half-plane branches"), ("orange", "pooled features"),
         ("purple", "the pixel's own long window"), ("red", "hidden: never seen")]
for i, (col, lab) in enumerate(items):
    xx, yy = px0 + (i % 2) * 190, ly + (i // 2) * 19
    if col == "red":
        rect(xx, yy - 10, 14, 11, "url(#hatch)", C["red"][1], 1.1, 0)
    else:
        rect(xx, yy - 10, 14, 11, C[col][0], C[col][1], 1.1, 0)
    text(xx + 20, yy, lab, 10.8)

# ---------------------------------------------------------------- training strip
rect(20, 866, W - 40, 156, "#FAFBFC", "#B8C0CA", 1.1, 8)
text(36, 890, "Training", 14, "bold")
tl = [
    "Mean loss: Σ w (y − F0)² / photon variance at M. It is minimized by the conditional mean, so the baseline is unbiased at any brightness "
    "(a robust likelihood on skewed low counts is not).",
    "Noise loss: 0.1 × Gaussian negative log-likelihood of movie − activity around F0 (held fixed), so σ² learns the actual spread around the baseline.",
    "Targets: simulated windows (real trials as templates, with fast brightness changes and realistic low counts) → the true baseline; "
    "real windows → movie − GIAnT's activity, no loss within 2 s of a trial's ends (to be re-exported: GIAnT under-fits activity, GIAnT-MATLAB#56).",
    "Deep arm (S4): from v6r, pretrained 3000 steps on simulations, then fine-tuned 1700 steps on half real, half simulated windows (event mask 0.5); "
    "up/down flips only; half of the simulated windows end or start at a trial edge.",
    "Shipped weights: WiSE-FT 0.5, the mean of the pretrained weights (step 0) and the average of fine-tuning steps 800–1700. "
    "On held-out mice it beats the control on every simulated kind; real data are re-tested after the re-export.",
]
for i, ln in enumerate(tl):
    text(36, 912 + i * 21, ln, 11.6)
add("</svg>")
open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("wrote", OUT)
