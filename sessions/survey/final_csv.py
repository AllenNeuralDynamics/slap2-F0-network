import json, os, re, csv, sys, collections, statistics

W = sys.argv[1]
OUT = sys.argv[2]
R = json.load(open(os.path.join(W, 'rows.json')))
CO = json.load(open(os.path.join(W, 'co_lookup.json')))
L = json.load(open(os.path.join(W, 'step3_listing.json')))
M = collections.defaultdict(list)
for l in open(os.path.join(W, 'meta_all.jsonl')):
    d = json.loads(l)
    n, f = d['file'].split('__', 1)
    d['acqfile'] = f
    M[n].append(d)
FOV_PIX = 800 * 1280
IGLU = re.compile(r'iGluSnFR4', re.I)
EXCL = re.compile(r'(reference_stack|static_data|structural_data|invalid_data|refStack|/structure|ref_stack)', re.I)


def meta_kind(d):
    if d['nROIs'] == 0:
        return 'noROI'
    if d.get('isSimpleRaster'):
        return 'simple-raster'
    if d['roiPixMax'] > 0.5 * FOV_PIX:
        return 'full-field-ROI'
    if d['nRaster'] == 0:
        return 'integration-only'
    if d['nROIs'] >= 40 and (d['roiPixMedian'] or 0) < 300:
        return 'band-scan-pattern'
    return 'multiROI'


def exp_files(n):
    return {os.path.basename(k): s for k, s in L.get(n, []) if k.endswith('.dat') and not EXCL.search(k) and re.search(r'-(TRIAL|CYCLE)', k)}


rows_by = {r['prefix']: r for r in R}
out = []
for r in R:
    if not r['n_dynamic_dat_files']:
        continue
    n = r['prefix']
    notes = []
    legacy = n.startswith('SLAP2_')
    twin = n[6:] if legacy else None
    is_dup = False
    if legacy and twin in rows_by and rows_by[twin]['n_dynamic_dat_files']:
        a, b = exp_files(n), exp_files(twin)
        is_dup = (set(a) == set(b)) and all(a[k] == b[k] for k in a)
    # ---- .meta
    ms = sorted(M.get(n, []), key=lambda d: d['acqfile'])
    kinds = set(meta_kind(d) for d in ms)
    nI = sum(d['nIntegrate'] for d in ms)
    meta_desc = ', '.join('%s R%d/I%d' % (re.sub(r'\.meta$', '', d['acqfile']).split('_')[-1] if r['_n_acq'] == 1 else re.sub(r'\.meta$', '', d['acqfile'])[-14:], d['nRaster'], d['nIntegrate']) for d in ms)
    if kinds == {'multiROI'}:
        meta_mode = 'multiROI'
    elif len(kinds) == 1:
        meta_mode = kinds.pop()
    else:
        meta_mode = 'mixed(' + '/'.join(sorted(kinds)) + ')'
    # ---- recorded label
    labels = [m for m in r['_modes'] if m not in ('None', '')]
    label_src = 'annotation.json slap2_mode'
    if not labels:
        at = (r['_acq_type'] or '')
        stripped = at.replace('& full-field raster structure stack', '').replace('full-field raster structure stack', '').strip()
        if stripped and stripped.lower() != 'slap2':
            labels = [stripped]
            label_src = 'acquisition.json acquisition_type'
    label = '|'.join(sorted(set(labels)))
    if labels and all(x.startswith('multi-roi raster') for x in labels):
        label_class = 'multi-roi raster'
    elif not labels:
        label_class = 'missing'
    else:
        label_class = label
    # ---- multiroi decision
    if label_class == 'multi-roi raster' and meta_mode == 'multiROI':
        mr = 'yes'
    elif label_class == 'missing' and meta_mode == 'multiROI':
        mr = 'yes'
        notes.append('mode from .meta only (no slap2_mode/acquisition_type recorded)')
    elif label_class == 'missing':
        mr = 'unknown'
    elif label_class == 'multi-roi raster' and meta_mode != 'multiROI':
        mr = 'unknown'
        notes.append('label multi-roi raster but .meta=%s' % meta_mode)
    elif meta_mode == 'multiROI' and 'integration scan' in label_class and all(d['nIntegrate'] >= d['nRaster'] for d in ms):
        mr = 'no'
        notes.append('recorded "integration scan"; .meta consistent (Integrate ROIs >= Raster ROIs on every DMD: %s)' % meta_desc)
    elif meta_mode == 'multiROI' and 'integration scan' in label_class:
        mr = 'unknown'
        notes.append('MODE CONFLICT: recorded "integration scan" but annotation planes lists show raster dendrite planes, and .meta has Raster-only DMD(s) (%s); same ROI structure as this mouse\'s sessions labelled multi-roi raster' % meta_desc)
    elif meta_mode == 'multiROI' and 'overlaid' in label_class:
        mr = 'unknown'
        notes.append('PARTIAL: recorded "%s" (files named raster+int_*); .meta has few Raster ROIs plus many large Integrate ROIs (%s)' % (label, meta_desc))
    elif meta_mode == 'multiROI' and 'full-field raster' in label_class:
        mr = 'unknown'
        notes.append('MODE CONFLICT: recorded "%s" (annotation type=ref_stack) but .meta has hand-drawn Raster ROIs, no full-field ROI, isSimpleRaster=false (%s); experiment_notes describe ~30 min continuous recording with "Raster ROIs over dendrites and somata" -> likely mislabelled multi-ROI raster' % (label, meta_desc))
    elif meta_mode == 'multiROI' and label_class == 'band scan':
        mr = 'unknown'
        notes.append('MODE CONFLICT: recorded band scan but .meta lacks band-scan ROI pattern (%s)' % meta_desc)
    elif label_class == 'band scan' and meta_mode == 'band-scan-pattern':
        mr = 'no'
        notes.append('.meta band-scan pattern: %s ROIs/DMD, median %s px each' % ('/'.join(str(d['nROIs']) for d in ms), '/'.join(str(int(d['roiPixMedian'])) for d in ms)))
    else:
        mr = 'no'
    mode_source = '%s="%s"' % (label_src, label) if labels else 'no recorded mode'
    mode_source += ' ; .meta AcquisitionContainer.ROIs: %s -> %s' % (meta_desc, meta_mode)
    if nI and mr in ('yes', 'unknown'):
        notes.append('includes %d Integrate ROIs across DMD .meta (in addition to Raster ROIs)' % nI)
    # ---- indicator
    ind = r['indicator']
    ind_src = r['indicator_source']
    iglu_recorded = bool(IGLU.search(ind)) and not ind.startswith('secondary')
    notes_ind = r.get('_notes_ind', [])
    if ind == 'None recorded' and notes_ind:
        ind = 'None recorded in metadata; notes mention ' + '/'.join(notes_ind)
        ind_src = 'experiment_notes.txt / acquisition.json notes (indicator fields empty)'
    if ind.startswith('secondary: '):
        ind = ind.replace('secondary: ', 'procedures construct: ')
    if ind == 'None recorded' and not legacy:
        sib = sorted(set(x for p in rows_by if rows_by[p]['subject_id'] == r['subject_id'] and p != n for x in rows_by[p].get('_notes_ind', [])))
        vir = sorted(set(m[1] or m[0] for m in r.get('_mats', []) if (m[1] or m[0])))
        notes.append('indicator fields empty and notes blank; procedures virus %s; other sessions of this mouse note %s' % (','.join(vir) or 'none', '/'.join(sib) or 'nothing'))
    # notes-vs-record conflict
    g = set(x.lower() for x in re.findall(r'iGluSnFR4\w*', r['indicator']))
    gn = set(x.lower().rstrip('.') for x in notes_ind if x.lower().startswith('iglusnfr4'))
    if g and gn and not (g & gn):
        notes.append('indicator variant conflict: metadata %s vs experiment_notes %s (other sessions of this mouse: %s)' % ('/'.join(sorted(g)), '/'.join(sorted(gn)), ','.join(sorted(set(rows_by[p]['indicator'] for p in rows_by if rows_by[p]['subject_id'] == r['subject_id'] and p != n and rows_by[p]['n_dynamic_dat_files'])))))
    # ---- acquisition style
    st = r['acquisition_style']
    perd = collections.Counter(int(re.search(r'_DMD(\d)', b).group(1)) for b in exp_files(n))
    per = '/'.join(str(perd[k]) for k in sorted(perd))
    if st.startswith('triggered'):
        st = 'triggered trials (TRIAL files; DMD1/2 = %s trials)' % per
    elif 'trial ids' in st:
        st = 'continuous, cycle-chunked within %s triggered trials (TRIAL-CYCLE files; DMD1/2 = %s files)' % (re.search(r'(\d+) trial ids', st).group(1), per)
    elif 'TRIAL-CYCLE' in st:
        st = 'continuous (single TRIAL, CYCLE chunks; DMD1/2 = %s files)' % per
    elif 'CYCLE' in st:
        st = 'continuous (CYCLE chunks; DMD1/2 = %s files)' % per
    # ---- qualifies
    if is_dup:
        q = 'no'
        tq = rows_by[twin]
        notes.insert(0, 'DUPLICATE legacy upload: experiment .dat set byte-identical (names+sizes) to %s; use that prefix' % twin)
        ind = 'None recorded here (twin %s: %s)' % (twin, tq['indicator'])
    elif legacy:
        notes.insert(0, 'legacy SLAP2_-prefixed asset (outside <id>_<date>_<time> pattern); non-standard layout %s' % ','.join(r['_dirs']))
        q = 'unclear' if mr == 'yes' else 'no'
        if n == 'SLAP2_803496_2025-07-02_15-33-26':
            notes.append('indicator only from procedures.json injection material VIR32222_PHPeB = pAAV-syn-flex-iGluSnFR4f-NGR-WPRE (VIP-Cre mouse); no channel/annotation metadata; same mouse\'s bare-prefix sessions 2025-07-25..08-01 are iGluSnFR4f multi-roi raster; folder also holds processed outputs (ALIGNMENTDATA.mat, REGISTERED tifs); likely satisfies (a)+(b)')
        if n == 'SLAP2_822057_2025-10-08_15-30-52':
            notes.append('no indicator recorded anywhere (procedures injection_materials empty, no notes, no annotation); only evidence for (b) is .meta')
    else:
        if iglu_recorded and mr == 'yes':
            q = 'yes'
        elif iglu_recorded and mr == 'unknown':
            q = 'unclear'
        else:
            q = 'no'
    if q == 'no' and not is_dup:
        why = []
        if not iglu_recorded:
            why.append('indicator not iGluSnFR4')
        if mr == 'no':
            why.append('not multiROI (%s)' % label)
        if why:
            notes.insert(0, 'fails: ' + '; '.join(why))
    if r['_n_acq'] > 1:
        notes.append('%d experiment acquisitions: %s' % (r['_n_acq'], ','.join(r['_acqs'])))
    try:
        ddn = json.load(open(os.path.join(W, 'meta', n, 'data_description.json'), encoding='utf-8')).get('name')
        if ddn and ddn != n:
            notes.append('data_description.name=%s differs from S3 prefix' % ddn)
    except Exception:
        pass
    if not ms:
        notes.append('no experiment .meta found')
    if r['_ann_missing'] and not legacy:
        notes.append('annotation.json missing for %d DMD acquisition(s)' % r['_ann_missing'])
    co = CO.get(n) or {}
    coid = ';'.join(a[0] for a in co.get('exact', []))
    out.append({
        'prefix': n, 'subject_id': r['subject_id'], 'date': r['date'],
        'indicator': ind, 'indicator_source': ind_src,
        'multiroi': mr, 'mode_source': mode_source,
        'n_dmd_paths': r['n_dmd_paths'], 'n_dynamic_dat_files': r['n_dynamic_dat_files'],
        'dynamic_data_GB': r['dynamic_data_GB'], 'acquisition_style': st,
        'codeocean_asset_id': coid, 'qualifies': q, 'notes': ' | '.join(notes),
    })
cols = ['prefix', 'subject_id', 'date', 'indicator', 'indicator_source', 'multiroi', 'mode_source', 'n_dmd_paths', 'n_dynamic_dat_files', 'dynamic_data_GB', 'acquisition_style', 'codeocean_asset_id', 'qualifies', 'notes']
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for o in sorted(out, key=lambda o: (o['prefix'].startswith('SLAP2_'), o['prefix'])):
        w.writerow(o)
print(len(out), collections.Counter(o['qualifies'] for o in out), collections.Counter(o['multiroi'] for o in out))
