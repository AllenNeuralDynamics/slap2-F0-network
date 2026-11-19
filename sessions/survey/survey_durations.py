"""SLAP2 sessions on aind-open-data: listings, metadata for sessions not in the old list,
and recording time of every experiment .dat file (first and last line timestamps).

usage: python survey.py WORKDIR   (WORKDIR holds data_descriptions.json from the first pass)
Writes WORKDIR/listings.json, meta_new.json, durations.json.
"""
import concurrent.futures as cf
import csv
import json
import re
import struct
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

W = Path(sys.argv[1])
BUCKET = "https://aind-open-data.s3.amazonaws.com"
MAGIC = 322379495
EXCL = re.compile(r"(reference_stack|static_data|structural_data|invalid_data|refStack|/structure|ref_stack)", re.I)
EXP = re.compile(r"-(TRIAL|CYCLE)\d*", re.I)
V2 = ["firstCycleOffsetBytes", "lineHeaderSizeBytes", "laserPathIdx", "bytesPerCycle", "linesPerCycle",
      "superPixelsPerCycle", "dmdPixelsPerRow", "dmdPixelsPerColumn", "numChannels", "channelMask", "numSlices",
      "channelsInterleave", "fpgaSystemClock_Hz", "referenceTimestamp_lower", "referenceTimestamp_upper"]


def get(url, headers=None, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=90) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise


def list_objects(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(get(BUCKET + "/?" + urllib.parse.urlencode(q)))
        ns = {"s": root.tag.split("}")[0].strip("{")}
        for c in root.findall("s:Contents", ns):
            out.append((c.find("s:Key", ns).text, int(c.find("s:Size", ns).text)))
        if root.find("s:IsTruncated", ns).text == "true":
            token = root.find("s:NextContinuationToken", ns).text
        else:
            return out


def rng(key, a, b):
    return get(BUCKET + "/" + urllib.parse.quote(key), {"Range": f"bytes={a}-{b}"})


def dat_info(key, size):
    """(cycles, seconds from the first to past the last cycle, lines per cycle) of one .dat file."""
    h = rng(key, 0, 1023)
    u = struct.unpack("<%dI" % (len(h) // 4), h)
    if u[0] != MAGIC:
        return {"error": "magic"}
    if u[1] == 1:
        off, bpc, lpc, clk = u[2], u[5], u[6], 200e6
    elif u[1] == 2:
        n = u[2] // 4
        pairs = u[3:n - 1]
        f = {V2[k]: v for k, v in zip(pairs[0::2], pairs[1::2]) if k < len(V2)}
        off, bpc, lpc, clk = f["firstCycleOffsetBytes"], f["bytesPerCycle"], f["linesPerCycle"], f.get("fpgaSystemClock_Hz") or 200e6
    else:
        return {"error": f"version {u[1]}"}
    nc = (size - off) // bpc if bpc else 0
    if nc < 2:
        return {"cycles": int(nc), "seconds": 0.0, "lines_per_cycle": lpc}

    def ts(b):
        v = struct.unpack("<IIIIIhhQ", b[:32])
        if v[1] != MAGIC:
            raise ValueError("line magic")
        return v[7]
    t0 = ts(rng(key, off, off + 31))
    t1 = ts(rng(key, off + (nc - 1) * bpc, off + (nc - 1) * bpc + 31))
    dur = (t1 - t0) / clk
    return {"cycles": int(nc), "seconds": dur * nc / (nc - 1), "lines_per_cycle": lpc, "clock": clk}


def main():
    dd = json.load(open(W / "data_descriptions.json"))
    slap = sorted(p for p, d in dd.items() if any(m in ("slap2", "slap") for m in (d.get("modalities") or [])))
    errs = sorted(p for p, d in dd.items() if "error" in d)
    old = {r["prefix"] for r in csv.DictReader(open(r"C:\GitHub\slap2-F0-network\sessions\slap2_iglusnfr4_sessions.csv"))}
    todo = slap + errs
    lst_fn = W / "listings.json"
    listings = json.load(open(lst_fn)) if lst_fn.exists() else {}
    with cf.ThreadPoolExecutor(24) as ex:
        for p, l in zip([p for p in todo if p not in listings], ex.map(lambda p: list_objects(p + "/"), [p for p in todo if p not in listings])):
            listings[p] = l
    json.dump(listings, open(lst_fn, "w"))
    print("listed", len(listings), flush=True)

    # metadata for sessions the old list does not have
    meta = {}
    for p in [p for p in todo if p not in old]:
        keys = [k for k, _ in listings.get(p, [])]
        m = {"n_files": len(keys), "has_slap2_dir": any("/slap2/" in k or "/slap/" in k for k in keys)}
        for nm in ("procedures.json", "acquisition.json", "session.json"):
            k = f"{p}/{nm}"
            if k in keys:
                try:
                    m[nm] = json.loads(get(BUCKET + "/" + urllib.parse.quote(k)))
                except Exception as e:
                    m[nm] = {"error": str(e)[:80]}
        ann = [k for k in keys if k.endswith(".annotation.json")]
        m["n_annotation"] = len(ann)
        if ann:
            try:
                m["annotation"] = json.loads(get(BUCKET + "/" + urllib.parse.quote(ann[len(ann) // 2])))
                m["annotation_key"] = ann[len(ann) // 2]
            except Exception as e:
                m["annotation"] = {"error": str(e)[:80]}
        meta[p] = m
    json.dump(meta, open(W / "meta_new.json", "w"))
    print("metadata for", len(meta), "new or unreadable prefixes", flush=True)

    # recording time: every experiment .dat file
    dur_fn = W / "durations.json"
    durs = json.load(open(dur_fn)) if dur_fn.exists() else {}
    jobs = []
    for p in slap:
        for k, s in listings.get(p, []):
            if k.endswith(".dat") and EXP.search(k) and not EXCL.search(k) and k not in durs:
                jobs.append((k, s))
    print(len(jobs), "dat files to time", flush=True)

    def one(ks):
        k, s = ks
        try:
            return k, dict(dat_info(k, s), size=s)
        except Exception as e:
            return k, {"error": str(e)[:80], "size": s}
    done = 0
    with cf.ThreadPoolExecutor(48) as ex:
        for k, r in ex.map(one, jobs):
            durs[k] = r
            done += 1
            if done % 1000 == 0:
                print(done, flush=True)
                json.dump(durs, open(dur_fn, "w"))
    json.dump(durs, open(dur_fn, "w"))
    print("timed", len(durs), "files;", sum("error" in v for v in durs.values()), "errors", flush=True)


if __name__ == "__main__":
    main()
