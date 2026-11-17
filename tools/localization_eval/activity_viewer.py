"""A self-contained HTML viewer comparing SILo runs' activity images and sources.

usage: python activity_viewer.py OUT.html NAME=RUN_DIR [NAME=RUN_DIR ...] [--paths Path1,Path2]
           [--flash SOURCE_CHECK_DIR] [--no_trials] [--title TEXT]

RUN_DIR is a SILo output folder (``source_extraction/experiment_summary.h5``
and, unless --no_trials, ``per_trial_summary.h5``). The page shows two panels
side by side, each any run's activity image (all trials or one trial) or the
mean image, with synchronized zoom and pan, contrast controls, a flicker
button, and the sources of both panels' runs marked by whether the other run
has them (one-to-one within 1.5 px). Clicking a source shows its SNR in each
run, its distance to the ROI edge and, with --flash (source_flash_check.py's
output for that path), its event-triggered flash image and kinetics.
Images are embedded as 8-bit PNGs (each scaled between its 0.5th and 99.8th
percentiles; NaN transparent), so the file opens without a server.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def png(img):
    """8-bit gray + alpha PNG data URI, scaled between robust percentiles; returns (uri, lo, hi)."""
    ok = np.isfinite(img)
    if not ok.any():
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (float(v) for v in np.percentile(img[ok], [0.5, 99.8]))
        hi = hi if hi > lo else lo + 1
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    a = (ok * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.stack([g, a], -1), "LA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), lo, hi


def rowcol(a):
    """A MATLAB-written image (stored col, row) as (row, col)."""
    return np.asarray(a, np.float64).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--paths", default="Path1,Path2")
    ap.add_argument("--flash", default="")
    ap.add_argument("--no_trials", action="store_true")
    ap.add_argument("--title", default="SILo activity images")
    args = ap.parse_args()
    runs = [r.split("=", 1) for r in args.runs]
    data = {"title": args.title, "runs": [n for n, _ in runs], "paths": {}}
    for path in args.paths.split(","):
        P = {"images": {}, "sources": {}, "trials": 0}
        valid = None
        for ri, (name, d) in enumerate(runs):
            d = Path(d) / "source_extraction"
            with h5py.File(d / "experiment_summary.h5", "r") as f:
                g = f[path]
                act = rowcol(g["visualizations/act_im"][:, :, 0])
                if ri == 0:
                    mean = np.asarray(g["visualizations/mean_im"][()])
                    for ch in range(mean.shape[-1]):
                        P["images"][f"mean image, channel {ch + 1}"] = png(np.sqrt(np.clip(rowcol(mean[:, :, 0, ch]), 0, None)))
                    m0 = rowcol(mean[:, :, 0, 0])
                    valid = np.isfinite(m0) & (m0 != 0)
                    P["shape"] = list(act.shape)
                xy = np.asarray(g["sources/spatial/coords"][()]).T[:, 1:3]
                snr = np.asarray(g["sources/temporal/SNR"][()]).ravel()
            P["images"][f"{name}: activity, all trials"] = png(act)
            edge = distance_transform_edt(valid)
            rc = np.clip(np.round(xy).astype(int), 0, np.array(act.shape) - 1)
            P["sources"][name] = [[round(float(r), 2), round(float(c), 2), round(float(s), 2), round(float(e), 1)]
                                  for (r, c), s, e in zip(xy, snr, edge[rc[:, 0], rc[:, 1]])]
            if not args.no_trials and (d / "per_trial_summary.h5").exists():
                with h5py.File(d / "per_trial_summary.h5", "r") as f:
                    pt = f[f"{path}/visualizations/per_trial_act_im"]
                    P["trials"] = pt.shape[-1]
                    for t in range(pt.shape[-1]):
                        P["images"][f"{name}: activity, trial {t + 1}"] = png(rowcol(pt[:, :, 0, t]))
            print(f"{path} {name}: {len(xy)} sources", flush=True)
        if args.flash and (Path(args.flash) / f"{path}_flash.npz").exists():
            S = np.load(Path(args.flash) / f"{path}_sources.npz")
            F = np.load(Path(args.flash) / f"{path}_flash.npz", allow_pickle=True)
            cols = [str(c) for c in F["cols"]]
            P["flash"] = {"lags": F["lags"].tolist(), "cols": cols + ["sync"], "items": [
                {"xy": [round(float(v), 2) for v in S["xy"][k]], "cat": int(S["cat"][k]),
                 "img": np.round(np.nan_to_num(F["flash"][k], nan=-9), 3).tolist(),
                 "valid": S["valid"][k].astype(int).tolist(),
                 "trace": np.round(np.nan_to_num(F["trace"][k]), 3).tolist(),
                 "meas": [None if not np.isfinite(v) else round(float(v), 3) for v in list(F["meas"][k]) + [F["sync"][k]]]}
                for k in range(len(S["cat"]))]}
        data["paths"][path] = P
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", args.title)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f6f4;--panel:#ffffff;--ink:#1d1d1b;--muted:#6b6b66;--line:#d9d9d4;--accent:#2563eb;
--both:#16a34a;--onlyL:#ea580c;--onlyR:#7c3aed;--imgbg:#e8e8e4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151514;--panel:#1f1f1d;--ink:#ecece8;--muted:#a3a39c;
--line:#3a3a36;--accent:#60a5fa;--both:#4ade80;--onlyL:#fb923c;--onlyR:#c4b5fd;--imgbg:#2a2a27}}
:root[data-theme="dark"]{--bg:#151514;--panel:#1f1f1d;--ink:#ecece8;--muted:#a3a39c;--line:#3a3a36;--accent:#60a5fa;
--both:#4ade80;--onlyL:#fb923c;--onlyR:#c4b5fd;--imgbg:#2a2a27}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:12px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0;font-weight:600}
label{color:var(--muted);font-size:12px;display:flex;gap:6px;align-items:center}
select,button,input[type=range]{font:inherit;color:var(--ink)}input[type=range]{width:120px;max-width:40vw}select,button{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
main{display:grid;grid-template-columns:1fr 1fr 320px;gap:12px;padding:12px 16px}
@media (max-width:1100px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px;min-width:0}
.panel .bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px}
.panel select{max-width:100%}
canvas.view{width:100%;display:block;background:var(--imgbg);border-radius:4px;cursor:grab;touch-action:none}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:6px}
.legend span{display:flex;gap:5px;align-items:center}
#detail h2{font-size:14px;margin:0 0 6px}#detail table{border-collapse:collapse;font-size:12px;width:100%}
#detail td{padding:2px 4px;border-bottom:1px solid var(--line)}#detail td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:var(--muted);font-size:12px}
#tablewrap{padding:0 16px 24px}#srctable{border-collapse:collapse;width:100%;font-size:12px;background:var(--panel)}
#srctable th,#srctable td{padding:3px 6px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
#srctable th{cursor:pointer;color:var(--muted);font-weight:500;position:sticky;top:0;background:var(--panel)}
#srctable td:first-child,#srctable th:first-child{text-align:left}#srctable tr:hover td{background:var(--imgbg);cursor:pointer}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;display:none;z-index:5}
</style></head><body>
<header><h1 id="title"></h1>
<label>Path <select id="path"></select></label>
<label>Trial <input id="trial" type="range" min="1" max="1" value="1"><span id="trialv">1</span></label>
<label>Black <input id="lo" type="range" min="0" max="254" value="0"></label>
<label>White <input id="hi" type="range" min="1" max="255" value="255"></label>
<label>Gamma <input id="gamma" type="range" min="0.3" max="2.5" step="0.05" value="1"></label>
<label><input id="marks" type="checkbox" checked> sources</label>
<button id="flicker" title="Show the right panel's image in the left panel while held">Flicker (hold)</button>
<button id="reset">Reset view</button>
<button id="theme" title="Light / dark">◐</button>
</header>
<main>
<div class="panel"><div class="bar"><select id="selL"></select></div><canvas class="view" id="cL"></canvas></div>
<div class="panel"><div class="bar"><select id="selR"></select></div><canvas class="view" id="cR"></canvas></div>
<div class="panel" id="detail"><h2>Source</h2><p class="muted">Click a marker or a table row. Wheel to zoom, drag to pan.</p></div>
</main>
<div id="tablewrap"><div class="legend" id="legend"></div><table id="srctable"></table></div>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
$("title").textContent = D.title;
const cache = {};  // image key -> {w,h,gray,alpha,canvas}
let view = {s: 1, x: 0, y: 0}, sel = null, flick = false, P = null;
function loadImage(key, uri) {
  return new Promise(res => {
    if (cache[key]) return res(cache[key]);
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas"); c.width = im.width; c.height = im.height;
      const g = c.getContext("2d"); g.drawImage(im, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data, n = c.width * c.height;
      const gray = new Uint8Array(n), alpha = new Uint8Array(n);
      for (let i = 0; i < n; i++) { gray[i] = d[4 * i]; alpha[i] = d[4 * i + 3]; }
      cache[key] = {w: c.width, h: c.height, gray, alpha, canvas: c, stamp: ""};
      res(cache[key]);
    };
    im.src = uri;
  });
}
function lutStamp() { return [$("lo").value, $("hi").value, $("gamma").value].join(","); }
function applyLut(e) {
  const st = lutStamp(); if (e.stamp === st) return;
  const lo = +$("lo").value, hi = Math.max(+$("hi").value, lo + 1), gm = +$("gamma").value, lut = new Uint8Array(256);
  for (let v = 0; v < 256; v++) lut[v] = Math.round(255 * Math.pow(Math.min(Math.max((v - lo) / (hi - lo), 0), 1), gm));
  const g = e.canvas.getContext("2d"), id = g.createImageData(e.w, e.h), d = id.data;
  for (let i = 0; i < e.gray.length; i++) { const v = lut[e.gray[i]]; d[4*i] = d[4*i+1] = d[4*i+2] = v; d[4*i+3] = e.alpha[i]; }
  g.putImageData(id, 0, 0); e.stamp = st;
}
function imgKey(which) {
  const v = $(which).value, t = $("trial").value;
  return v.includes("{trial}") ? v.replace("{trial}", t) : v;
}
function runOf(which) { const v = $(which).value; const m = D.runs.find(r => v.startsWith(r + ":")); return m || null; }
function match(a, b) {  // one-to-one within 1.5 px, closest first -> Map index a -> index b
  const pairs = [];
  a.forEach((p, i) => b.forEach((q, j) => { const d = Math.hypot(p[0] - q[0], p[1] - q[1]); if (d <= 1.5) pairs.push([d, i, j]); }));
  pairs.sort((x, y) => x[0] - y[0]);
  const ua = new Set(), ub = new Set(), m = new Map();
  for (const [, i, j] of pairs) if (!ua.has(i) && !ub.has(j)) { ua.add(i); ub.add(j); m.set(i, j); }
  return m;
}
function marks() {  // markers for both panels' runs
  const L = runOf("selL"), R = runOf("selR"), out = [];
  if (!L && !R) return out;
  const a = L ? P.sources[L] : [], b = R ? P.sources[R] : [];
  if (L && R && L !== R) {
    const m = match(a, b), mb = new Set(m.values());
    a.forEach((p, i) => out.push({p, kind: m.has(i) ? "both" : "onlyL", run: L, i, other: m.has(i) ? b[m.get(i)] : null}));
    b.forEach((p, j) => { if (!mb.has(j)) out.push({p, kind: "onlyR", run: R, i: j, other: null}); });
  } else {
    const r = L || R; P.sources[r].forEach((p, i) => out.push({p, kind: "both", run: r, i, other: null}));
  }
  for (const m of out) m.pass = passes(flashFor(m.p));
  return out;
}
let M = [];
function draw(cid, which) {
  const c = $(cid), e = cache[imgKey(which)];
  const W = c.clientWidth, H = Math.round(W * P.shape[0] / P.shape[1]);
  if (c.width !== W * devicePixelRatio) { c.width = W * devicePixelRatio; c.height = H * devicePixelRatio; c.style.height = H + "px"; }
  const g = c.getContext("2d"); g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height);
  if (!e) return;
  applyLut(e);
  const k = c.width / P.shape[1] * view.s;
  g.imageSmoothingEnabled = false;
  g.setTransform(k, 0, 0, k, -view.x * k, -view.y * k);
  g.drawImage(e.canvas, 0, 0);
  if (!$("marks").checked) return;
  g.setTransform(1, 0, 0, 1, 0, 0);
  const col = {both: css("--both"), onlyL: css("--onlyL"), onlyR: css("--onlyR")}, r = Math.max(4 * devicePixelRatio, 1.6 * k);
  for (const m of M) {
    const X = (m.p[1] - view.x) * k, Y = (m.p[0] - view.y) * k;
    if (X < -10 || Y < -10 || X > c.width + 10 || Y > c.height + 10) continue;
    const isSel = sel && sel.run === m.run && sel.i === m.i;
    g.strokeStyle = col[m.kind]; g.lineWidth = (isSel ? 3 : 1.5) * devicePixelRatio;
    g.setLineDash(m.pass === false ? [2.5 * devicePixelRatio, 2 * devicePixelRatio] : []);
    g.beginPath();
    if (m.kind === "both") g.arc(X, Y, r, 0, 2 * Math.PI);
    else if (m.kind === "onlyL") { g.moveTo(X, Y - r); g.lineTo(X + r, Y); g.lineTo(X, Y + r); g.lineTo(X - r, Y); g.closePath(); }
    else g.rect(X - r * 0.8, Y - r * 0.8, r * 1.6, r * 1.6);
    g.stroke();
  }
}
async function render() {
  const keys = [imgKey("selL"), imgKey("selR")];
  for (const k of keys) if (P.images[k]) await loadImage(P.path + "|" + k, P.images[k][0]).then(e => cache[k] = e);
  M = marks();
  if (flick) { const t = $("selL").value; $("selL").value = $("selR").value; draw("cL", "selL"); $("selL").value = t; }
  else draw("cL", "selL");
  draw("cR", "selR");
}
function options() {
  const opts = [];
  for (const k of Object.keys(P.images)) {
    if (/: activity, trial \d+$/.test(k)) { const b = k.replace(/trial \d+$/, "trial {trial}"); if (!opts.includes(b)) opts.push(b); }
    else opts.push(k);
  }
  return opts;
}
function setPath(p) {
  P = D.paths[p]; P.path = p; $("path").value = p;
  for (const k in cache) delete cache[k];
  const opts = options(), runsAll = opts.filter(o => o.endsWith("all trials"));
  for (const [id, def] of [["selL", runsAll[0]], ["selR", runsAll[1] || runsAll[0]]]) {
    const s = $(id), cur = s.value; s.innerHTML = "";
    for (const o of opts) { const e = document.createElement("option"); e.value = o; e.textContent = o.replace("{trial}", "selected"); s.appendChild(e); }
    s.value = opts.includes(cur) ? cur : def;
  }
  $("trial").max = Math.max(P.trials, 1); $("trial").disabled = !P.trials;
  view = {s: 1, x: 0, y: 0}; sel = null; render(); table(); legend();
}
function legend() {
  const L = runOf("selL"), R = runOf("selR");
  $("legend").innerHTML = L && R && L !== R ?
    `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="none" stroke="${css("--both")}" stroke-width="1.5"/></svg>both runs</span>` +
    `<span><svg width="12" height="12"><path d="M6 1 L11 6 L6 11 L1 6Z" fill="none" stroke="${css("--onlyL")}" stroke-width="1.5"/></svg>only ${L} (left)</span>` +
    `<span><svg width="12" height="12"><rect x="2" y="2" width="8" height="8" fill="none" stroke="${css("--onlyR")}" stroke-width="1.5"/></svg>only ${R} (right)</span>` +
    (P.flash ? `<span><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="none" stroke="${css("--muted")}" stroke-width="1.5" stroke-dasharray="2.5 2"/></svg>dashed: fails the flash test</span>` : "")
    : `<span>Choose two runs' activity images to compare their sources.</span>`;
}
function flashFor(p) {
  if (!P.flash) return null;
  let best = null, bd = 1.01;
  for (const it of P.flash.items) { const d = Math.hypot(it.xy[0] - p[0], it.xy[1] - p[1]); if (d < bd) { bd = d; best = it; } }
  return best;
}
function fm(f, name) {  // a flash measure by name, or null
  if (!f) return null; const i = P.flash.cols.indexOf(name); return i < 0 ? null : f.meas[i];
}
function passes(f) {  // the selection-free flash test, or null without data
  const r = fm(f, "ring_z"), t = fm(f, "tail_z");
  return r === null || t === null ? null : r > 3 && t > 3;
}
function heat(v) {  // magma-like ramp for the flash image
  const s = [[0,0,4],[59,15,112],[140,41,129],[222,73,104],[254,159,109],[252,253,191]];
  const x = Math.min(Math.max(v, 0), 1) * (s.length - 1), i = Math.min(Math.floor(x), s.length - 2), f = x - i;
  return s[i].map((a, k) => Math.round(a + f * (s[i + 1][k] - a)));
}
function detail(m) {
  const d = $("detail"), f = flashFor(m.p), L = runOf("selL"), R = runOf("selR");
  const kind = {both: "found by both runs", onlyL: `only in ${L}`, onlyR: `only in ${R}`}[m.kind];
  let h = `<h2>${m.run} source ${m.i + 1}</h2><p class="muted">${kind}; row ${m.p[0].toFixed(1)}, col ${m.p[1].toFixed(1)}</p><table>`;
  h += `<tr><td>SNR (${m.run})</td><td>${m.p[2].toFixed(2)}</td></tr>`;
  if (m.other) h += `<tr><td>SNR (other run)</td><td>${m.other[2].toFixed(2)}</td></tr>`;
  h += `<tr><td>distance to ROI edge</td><td>${m.p[3].toFixed(1)} px</td></tr>`;
  if (f) {
    const lab = {sigma_major: "flash sigma, major (px)", sigma_minor: "flash sigma, minor (px)", orientation: "orientation (deg)",
      centre_offset: "flash centre offset (px)", fwhm_frames: "kinetics FWHM (frames)", decay_frames: "decay to half (frames)",
      zeros: "frames exactly 0", motion_pct: "motion at events (pct)", event_size: "event size (counts)", sync: "events shared with far sources",
      ring_z: "ring 3–4 px at events (z)", tail_z: "decay tail t+3..8 (z)"};
    P.flash.cols.forEach((c, i) => { const v = f.meas[i]; h += `<tr><td>${lab[c] || c}</td><td>${v === null ? "–" : (+v).toFixed(2)}</td></tr>`; });
  }
  h += `</table>`;
  const ps = passes(f);
  if (ps !== null) h += `<p style="margin:8px 0 0"><b>${ps ? "Passes" : "Fails"}</b> <span class="muted">the flash test: a synapse-sized
    disk (ring z &gt; 3) with sensor decay (tail z &gt; 3), both measured outside the window used to pick the events.</span></p>`;
  if (f) h += `<p class="muted" style="margin:10px 0 4px">Event-triggered flash (30 largest events; ROI edge in cyan) and centre kinetics</p>
    <div style="display:flex;gap:8px;align-items:flex-start"><canvas id="fl" width="17" height="17" style="width:136px;height:136px;image-rendering:pixelated;border-radius:4px"></canvas>
    <svg id="kin" width="160" height="136" viewBox="0 0 160 136"></svg></div>`;
  else h += `<p class="muted">No flash data for this source (flash data covers ${P.flash ? "the first run's sources, the second run's unmatched ones and control pixels" : "nothing on this path"}).</p>`;
  d.innerHTML = h;
  if (f) {
    const c = $("fl"), g = c.getContext("2d"), id = g.createImageData(17, 17);
    f.img.forEach((row, r) => row.forEach((v, cc) => { const k = 4 * (r * 17 + cc);
      if (v <= -8) { id.data[k + 3] = 0; return; }
      const [R_, G_, B_] = heat((v + 0.2) / 1.2); id.data.set([R_, G_, B_, 255], k);
      if (!f.valid[r][cc]) { id.data[k] = id.data[k] * 0.5; id.data[k + 1] = id.data[k + 1] * 0.5 + 90; id.data[k + 2] = id.data[k + 2] * 0.5 + 90; } }));
    g.putImageData(id, 0, 0);
    const lags = P.flash.lags, x = l => 8 + (l - lags[0]) / (lags[lags.length - 1] - lags[0]) * 146, y = v => 110 - v * 90;
    const pts = f.trace.map((v, i) => `${x(lags[i]).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    $("kin").innerHTML = `<line x1="8" x2="154" y1="${y(0)}" y2="${y(0)}" stroke="${css("--line")}"/>
      <line x1="${x(0)}" x2="${x(0)}" y1="10" y2="120" stroke="${css("--line")}" stroke-dasharray="2 2"/>
      <polyline points="${pts}" fill="none" stroke="${css("--accent")}" stroke-width="1.5"/>
      <text x="8" y="132" font-size="10" fill="${css("--muted")}">${lags[0]}</text><text x="${x(0) - 3}" y="132" font-size="10" fill="${css("--muted")}">0</text>
      <text x="138" y="132" font-size="10" fill="${css("--muted")}">${lags[lags.length - 1]}</text>`;
  }
}
let sortKey = "kind", sortDir = 1;
function table() {
  const rows = marks().map(m => {
    const f = flashFor(m.p);
    const ps = passes(f);
    return {m, kind: m.kind, run: m.run, id: m.i + 1, snr: m.p[2], other: m.other ? m.other[2] : null, edge: m.p[3],
      sigma: fm(f, "sigma_major"), fwhm: fm(f, "fwhm_frames"), zeros: fm(f, "zeros"), ring: fm(f, "ring_z"), tail: fm(f, "tail_z"),
      pass: ps === null ? null : ps ? "pass" : "fail"};
  });
  const order = {both: 0, onlyL: 1, onlyR: 2};
  rows.sort((a, b) => { const va = sortKey === "kind" ? order[a.kind] : a[sortKey], vb = sortKey === "kind" ? order[b.kind] : b[sortKey];
    return ((va ?? -1e9) > (vb ?? -1e9) ? 1 : (va ?? -1e9) < (vb ?? -1e9) ? -1 : 0) * sortDir; });
  const L = runOf("selL"), R = runOf("selR"), kn = {both: "both", onlyL: `only ${L}`, onlyR: `only ${R}`};
  const cols = [["kind", "category"], ["run", "run"], ["id", "#"], ["snr", "SNR"], ["other", "SNR other run"], ["edge", "edge dist (px)"],
    ["sigma", "flash sigma (px)"], ["fwhm", "FWHM (frames)"], ["zeros", "zero frames"], ["ring", "ring z"], ["tail", "tail z"],
    ["pass", "flash test"]];
  const fmt = v => v === null || v === undefined ? "–" : typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(2)) : v;
  $("srctable").innerHTML = "<thead><tr>" + cols.map(([k, t]) => `<th data-k="${k}">${t}${sortKey === k ? (sortDir > 0 ? " ▲" : " ▼") : ""}</th>`).join("") +
    "</tr></thead><tbody>" + rows.map((r, i) => `<tr data-i="${i}">` + cols.map(([k]) => `<td>${k === "kind" ? kn[r.kind] : fmt(r[k])}</td>`).join("") + "</tr>").join("") + "</tbody>";
  $("srctable").querySelectorAll("th").forEach(th => th.onclick = () => { const k = th.dataset.k; sortDir = sortKey === k ? -sortDir : 1; sortKey = k; table(); });
  $("srctable").querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => {
    const m = rows[+tr.dataset.i].m; sel = M.find(q => q.run === m.run && q.i === m.i) || m;
    view.s = Math.max(view.s, 4); view.x = m.p[1] - P.shape[1] / view.s / 2; view.y = m.p[0] - P.shape[0] / view.s / 2;
    detail(sel); render(); window.scrollTo({top: 0, behavior: "smooth"});
  });
}
function toImage(c, ev) {
  const b = c.getBoundingClientRect(), k = b.width / P.shape[1] * view.s;
  return [(ev.clientY - b.top) / k + view.y, (ev.clientX - b.left) / k + view.x, k];
}
for (const cid of ["cL", "cR"]) {
  const c = $(cid); let drag = null;
  c.addEventListener("wheel", ev => { ev.preventDefault(); const [r, cc] = toImage(c, ev), f = ev.deltaY < 0 ? 1.25 : 0.8;
    view.s = Math.min(Math.max(view.s * f, 1), 40); const [r2, c2] = toImage(c, ev); view.x += cc - c2; view.y += r - r2; render(); }, {passive: false});
  c.addEventListener("pointerdown", ev => { drag = {x: ev.clientX, y: ev.clientY, vx: view.x, vy: view.y, moved: false}; c.setPointerCapture(ev.pointerId); });
  c.addEventListener("pointermove", ev => {
    if (drag) { const b = c.getBoundingClientRect(), k = b.width / P.shape[1] * view.s;
      if (Math.abs(ev.clientX - drag.x) + Math.abs(ev.clientY - drag.y) > 3) drag.moved = true;
      view.x = drag.vx - (ev.clientX - drag.x) / k; view.y = drag.vy - (ev.clientY - drag.y) / k; render(); return; }
    const [r, cc, k] = toImage(c, ev); let best = null, bd = 8 / k;
    for (const m of M) { const d = Math.hypot(m.p[0] - r, m.p[1] - cc); if (d < bd) { bd = d; best = m; } }
    const tip = $("tip");
    if (best) { tip.style.display = "block"; tip.style.left = ev.clientX + 12 + "px"; tip.style.top = ev.clientY + 12 + "px";
      tip.textContent = `${best.run} #${best.i + 1}: SNR ${best.p[2].toFixed(1)}, ${best.kind === "both" ? "both runs" : "one run only"}`; }
    else tip.style.display = "none";
  });
  c.addEventListener("pointerup", ev => {
    if (drag && !drag.moved) { const [r, cc, k] = toImage(c, ev); let best = null, bd = 8 / k;
      for (const m of M) { const d = Math.hypot(m.p[0] - r, m.p[1] - cc); if (d < bd) { bd = d; best = m; } }
      if (best) { sel = best; detail(best); render(); } }
    drag = null; });
  c.addEventListener("pointerleave", () => $("tip").style.display = "none");
}
for (const p of Object.keys(D.paths)) { const o = document.createElement("option"); o.value = o.textContent = p; $("path").appendChild(o); }
$("path").onchange = () => setPath($("path").value);
for (const id of ["selL", "selR"]) $(id).onchange = () => { render(); table(); legend(); };
for (const id of ["lo", "hi", "gamma", "marks"]) $(id).oninput = render;
$("trial").oninput = () => { $("trialv").textContent = $("trial").value; render(); };
$("reset").onclick = () => { view = {s: 1, x: 0, y: 0}; render(); };
$("flicker").onpointerdown = () => { flick = true; render(); };
for (const ev of ["pointerup", "pointerleave"]) $("flicker").addEventListener(ev, () => { if (flick) { flick = false; render(); } });
$("theme").onclick = () => { const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark"; render(); legend(); };
window.addEventListener("resize", render);
setPath(Object.keys(D.paths).includes("Path2") ? "Path2" : Object.keys(D.paths)[0]);
</script></body></html>
"""

if __name__ == "__main__":
    main()
