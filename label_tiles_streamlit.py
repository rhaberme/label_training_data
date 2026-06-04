"""
Streamlit-based tile labeling tool.
Images and masks are loaded from Google Drive; labels are persisted
to tile_labels.json in the Drive state folder.
"""

import base64
import io
import json
import threading
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as _stc
from PIL import Image
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

# ── Visual settings ────────────────────────────────────────────────────────
PANEL_SIZE = 600
MASK_ALPHA = 0.45

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (65,  182, 230),
    2: (255,  80,   0),
}
CLASS_NAMES: dict[int, str] = {
    1: "Gebäude",
    2: "Versiegelt",
}

# ── Prefetch settings ──────────────────────────────────────────────────────
PREFETCH_BATCH   = 100
PREFETCH_TRIGGER = 20

_prefetch_lock   = threading.Lock()
_prefetch_active: set[str] = set()

# ── Module-level caches (survive Streamlit reruns) ─────────────────────────
_bytes_cache: dict[str, bytes]     = {}   # file_id  → raw bytes
_composite_cache: dict[str, bytes] = {}   # tile_id  → JPEG bytes

_bytes_cache_lock     = threading.RLock()
_composite_cache_lock = threading.RLock()

_save_lock = threading.Lock()
_last_save_error: str | None = None

# ── Drive REST endpoints ───────────────────────────────────────────────────
_FILES_URL  = "https://www.googleapis.com/drive/v3/files"
_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# ── Session helpers ────────────────────────────────────────────────────────

def _make_session(info: dict) -> AuthorizedSession:
    creds = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    session = AuthorizedSession(creds)
    session.trust_env = False   # ignore http_proxy / https_proxy env-vars
    return session


@st.cache_resource
def _session() -> AuthorizedSession:
    return _make_session(dict(st.secrets["gcp_service_account"]))


def _thread_session() -> AuthorizedSession:
    return _make_session(dict(st.secrets["gcp_service_account"]))

# ── Drive API wrappers ─────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def list_folder(folder_id: str) -> dict[str, str]:
    files: dict[str, str] = {}
    params: dict[str, Any] = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "nextPageToken, files(id, name)",
        "pageSize": 1000,
    }
    while True:
        resp = _session().get(_FILES_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        for f in data.get("files", []):
            files[f["name"]] = f["id"]
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        params["pageToken"] = page_token
    return files


def _find_file(folder_id: str, name: str) -> str | None:
    resp = _session().get(_FILES_URL, params={
        "q": f"'{folder_id}' in parents and name='{name}' and trashed=false",
        "fields": "files(id)",
    })
    resp.raise_for_status()
    files = resp.json().get("files", [])
    return files[0]["id"] if files else None


def _fetch_bytes(session: AuthorizedSession, file_id: str) -> bytes:
    resp = session.get(f"{_FILES_URL}/{file_id}", params={"alt": "media"})
    resp.raise_for_status()
    return resp.content


def download_bytes(file_id: str) -> bytes:
    with _bytes_cache_lock:
        if file_id in _bytes_cache:
            return _bytes_cache[file_id]
    data = _fetch_bytes(_session(), file_id)
    with _bytes_cache_lock:
        _bytes_cache[file_id] = data
    return data

# ── Dataset configuration ──────────────────────────────────────────────────

def _get_datasets() -> dict[str, dict]:
    drive_cfg = st.secrets["drive"]
    return {k: drive_cfg[k] for k in list(drive_cfg.get("datasets", []))}

# ── Image processing ───────────────────────────────────────────────────────

def build_composite(img_bytes: bytes, mask_bytes: bytes,
                    panel_w: int, panel_h: int) -> Image.Image:
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((panel_w, panel_h), Image.LANCZOS)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((panel_w, panel_h), Image.NEAREST)
    orig     = np.array(img,  dtype=np.float32)
    mask_arr = np.array(mask)
    blended  = orig.copy()
    for cls_id, color in CLASS_COLORS.items():
        fg = mask_arr == cls_id
        if fg.any():
            blended[fg] = blended[fg] * (1 - MASK_ALPHA) + np.array(color, dtype=np.float32) * MASK_ALPHA
    mask_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.float32)
    for cls_id, color in CLASS_COLORS.items():
        fg = mask_arr == cls_id
        if fg.any():
            mask_rgb[fg] = color
    combined = np.concatenate([
        np.clip(orig,     0, 255).astype(np.uint8),
        np.clip(blended,  0, 255).astype(np.uint8),
        np.clip(mask_rgb, 0, 255).astype(np.uint8),
    ], axis=1)
    return Image.fromarray(combined)


def _composite_jpeg(tile_id: str, img_fid: str, mask_fid: str) -> bytes:
    with _composite_cache_lock:
        if tile_id in _composite_cache:
            return _composite_cache[tile_id]
    pil = build_composite(download_bytes(img_fid), download_bytes(mask_fid), PANEL_SIZE, PANEL_SIZE)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    with _composite_cache_lock:
        _composite_cache[tile_id] = data
    return data


def _img_tag(tile_id: str, img_fid: str, mask_fid: str, hidden: bool = False) -> str:
    jpeg = _composite_jpeg(tile_id, img_fid, mask_fid)
    b64  = base64.b64encode(jpeg).decode()
    style = "display:none" if hidden else "width:100%;border-radius:4px"
    return f'<img src="data:image/jpeg;base64,{b64}" style="{style}">'

# ── Prefetch ───────────────────────────────────────────────────────────────

def _prefetch_worker(
    tile_ids: list[str],
    image_ids: dict[str, str],
    mask_ids: dict[str, str],
    batch_key: str,
) -> None:
    try:
        session = _thread_session()
    except Exception:
        with _prefetch_lock:
            _prefetch_active.discard(batch_key)
        return

    for tid in tile_ids:
        img_fid  = image_ids.get(tid)
        mask_fid = mask_ids.get(tid)
        img_bytes = mask_bytes = None

        for fid, store in [(img_fid, "img"), (mask_fid, "mask")]:
            if not fid:
                continue
            with _bytes_cache_lock:
                cached = _bytes_cache.get(fid)
            if cached:
                if store == "img":
                    img_bytes = cached
                else:
                    mask_bytes = cached
                continue
            try:
                data = _fetch_bytes(session, fid)
                with _bytes_cache_lock:
                    _bytes_cache[fid] = data
                if store == "img":
                    img_bytes = data
                else:
                    mask_bytes = data
            except Exception:
                break

        if img_bytes and mask_bytes:
            with _composite_cache_lock:
                already = tid in _composite_cache
            if not already:
                try:
                    pil = build_composite(img_bytes, mask_bytes, PANEL_SIZE, PANEL_SIZE)
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG", quality=90)
                    with _composite_cache_lock:
                        _composite_cache[tid] = buf.getvalue()
                except Exception:
                    pass

    with _prefetch_lock:
        _prefetch_active.discard(batch_key)


def _maybe_prefetch(
    dataset_key: str,
    idx: int,
    all_ids: list[str],
    image_ids: dict[str, str],
    mask_ids: dict[str, str],
) -> None:
    batches = [idx // PREFETCH_BATCH]
    if PREFETCH_BATCH - (idx % PREFETCH_BATCH) <= PREFETCH_TRIGGER:
        batches.append(idx // PREFETCH_BATCH + 1)

    for batch_num in batches:
        batch_key = f"{dataset_key}_{batch_num}"
        with _prefetch_lock:
            if batch_key in _prefetch_active:
                continue
            start = batch_num * PREFETCH_BATCH
            end   = min(start + PREFETCH_BATCH, len(all_ids))
            ids   = all_ids[start:end]
            if not ids:
                continue
            _prefetch_active.add(batch_key)
        threading.Thread(
            target=_prefetch_worker,
            args=(ids, image_ids, mask_ids, batch_key),
            daemon=True,
        ).start()

# ── State helpers ──────────────────────────────────────────────────────────

def _tile_status(tile_id: str) -> str:
    if tile_id in st.session_state.training_set:
        return "training"
    if tile_id in st.session_state.unused_set:
        return "unused"
    return "open"


def _save_state() -> None:
    global _last_save_error
    ss = st.session_state
    state_copy = {
        "training": list(ss.label_state["training"]),
        "unused":   list(ss.label_state["unused"]),
    }
    file_id = ss.state_file_id
    with _save_lock:
        try:
            if not file_id:
                return
            resp = _session().patch(
                f"{_UPLOAD_URL}/{file_id}",
                params={"uploadType": "media"},
                data=json.dumps(state_copy, indent=2).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            _last_save_error = None
        except Exception as exc:
            _last_save_error = str(exc)


def _label(category: str) -> None:
    ss  = st.session_state
    idx = ss.current_index
    if idx >= len(ss.all_ids):
        return
    tile_id = ss.all_ids[idx]
    other   = "unused" if category == "training" else "training"
    if tile_id in ss.label_state[other]:
        ss.label_state[other].remove(tile_id)
        (ss.unused_set if other == "unused" else ss.training_set).discard(tile_id)
    if tile_id not in ss.label_state[category]:
        ss.label_state[category].append(tile_id)
        (ss.training_set if category == "training" else ss.unused_set).add(tile_id)
    ss.current_index = min(idx + 1, len(ss.all_ids) - 1)
    _save_state()


def _navigate(direction: int) -> None:
    ss = st.session_state
    ss.current_index = max(0, min(ss.current_index + direction, len(ss.all_ids) - 1))

# ── Initialization ─────────────────────────────────────────────────────────

def _initialize(dataset_key: str) -> None:
    ss  = st.session_state
    cfg = st.secrets["drive"][dataset_key]
    _session()

    images = list_folder(cfg["images_folder_id"])
    masks  = list_folder(cfg["masks_folder_id"])
    img_map  = {PurePosixPath(n).stem: fid for n, fid in images.items() if n.endswith(".jpg")}
    mask_map = {PurePosixPath(n).stem: fid for n, fid in masks.items()  if n.endswith(".png")}
    common   = sorted(set(img_map) & set(mask_map))

    ss.all_ids   = common
    ss.image_ids = img_map
    ss.mask_ids  = mask_map

    state_filename = f"tile_labels_{dataset_key}.json"
    file_id = _find_file(cfg["state_folder_id"], state_filename)
    if file_id:
        state = json.loads(_fetch_bytes(_session(), file_id))
    else:
        state = {"training": [], "unused": []}
        resp = _session().post(
            _UPLOAD_URL,
            params={"uploadType": "multipart"},
            files={
                "metadata": (None, json.dumps({"name": state_filename, "parents": [cfg["state_folder_id"]]}), "application/json; charset=UTF-8"),
                "media": (None, json.dumps(state, indent=2).encode(), "application/json"),
            },
        )
        resp.raise_for_status()
        file_id = resp.json()["id"]

    ss.label_state    = state
    ss.state_file_id  = file_id
    ss.state_filename = state_filename
    ss.training_set   = set(state.get("training", []))
    ss.unused_set     = set(state.get("unused",   []))

    labeled = ss.training_set | ss.unused_set
    idx = 0
    while idx < len(common) and common[idx] in labeled:
        idx += 1
    ss.current_index  = idx
    ss.active_dataset = dataset_key
    ss.initialized    = True
    _maybe_prefetch(dataset_key, idx, common, img_map, mask_map)

# ── Fragment: interactive labeling UI ─────────────────────────────────────

@st.fragment
def _labeling_fragment(dataset_key: str) -> None:
    ss      = st.session_state
    all_ids = ss.all_ids
    idx     = ss.current_index

    if not all_ids:
        st.error("Keine Tiles gefunden. Prüfe die Ordner-IDs in den Secrets.")
        return
    if idx >= len(all_ids):
        st.success("✓ Alle Tiles wurden bewertet!")
        return

    tile_id       = all_ids[idx]
    status        = _tile_status(tile_id)
    labeled_count = len(ss.training_set | ss.unused_set)
    total         = len(all_ids)
    n_train       = len(ss.training_set)
    n_unused      = len(ss.unused_set)
    n_open        = total - labeled_count
    pct           = int(labeled_count / total * 100) if total else 0

    # ── Header card ──────────────────────────────────────────────────────────
    BADGE = {
        "training": ("TRAINING",      "#a6e3a1", "#1e1e2e"),
        "unused":   ("NICHT GENUTZT", "#f38ba8", "#1e1e2e"),
        "open":     ("OFFEN",         "#f9e2af", "#1e1e2e"),
    }
    badge_text, badge_bg, badge_fg = BADGE[status]

    st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;
            background:#313244;border-radius:12px;
            padding:0.7rem 1.1rem;margin-bottom:0.65rem;
            border:1px solid #45475a;">
  <span style="color:#6c7086;font-size:0.68rem;text-transform:uppercase;
               letter-spacing:0.1em;font-weight:700;flex-shrink:0">Tile</span>
  <code style="background:#1e1e2e;color:#89b4fa;padding:0.2rem 0.55rem;
               border-radius:6px;font-size:0.88rem;font-weight:600">{tile_id}</code>
  <span style="background:{badge_bg};color:{badge_fg};padding:0.2rem 0.7rem;
               border-radius:99px;font-size:0.68rem;font-weight:800;
               text-transform:uppercase;letter-spacing:0.08em;flex-shrink:0">{badge_text}</span>
  <span style="margin-left:auto;color:#a6adc8;font-weight:700;
               font-size:0.88rem;flex-shrink:0">{labeled_count}&thinsp;/&thinsp;{total}
    <span style="color:#6c7086;font-weight:400;font-size:0.8rem">&nbsp;({pct}%)</span>
  </span>
</div>
""", unsafe_allow_html=True)
    st.progress(labeled_count / total if total else 0)

    # ── Image card ───────────────────────────────────────────────────────────
    jpeg = _composite_jpeg(tile_id, ss.image_ids[tile_id], ss.mask_ids[tile_id])
    b64  = base64.b64encode(jpeg).decode()

    preload_tag = ""
    if idx + 1 < len(all_ids):
        nxt = all_ids[idx + 1]
        with _composite_cache_lock:
            nxt_jpeg = _composite_cache.get(nxt)
        if nxt_jpeg:
            preload_tag = (
                f'<img src="data:image/jpeg;base64,{base64.b64encode(nxt_jpeg).decode()}"'
                f' style="display:none">'
            )

    st.markdown(f"""
<div style="border-radius:14px;overflow:hidden;
            box-shadow:0 8px 40px rgba(0,0,0,0.55),0 0 0 1px #45475a;
            margin-bottom:0.35rem;">
  <img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;">
  {preload_tag}
</div>
<p style="text-align:center;color:#585b70;font-size:0.68rem;
          letter-spacing:0.14em;text-transform:uppercase;margin:0 0 0.6rem;">
  Originalbild &nbsp;·&nbsp; Überlagert &nbsp;·&nbsp; Maske
</p>
""", unsafe_allow_html=True)

    # ── Label buttons ─────────────────────────────────────────────────────────
    col_nein, col_ja = st.columns(2)
    if col_nein.button("✗  Nein", width="stretch", key="btn_nein"):
        _label("unused")
        st.rerun(scope="fragment")
    if col_ja.button("✓  Ja", width="stretch", type="primary", key="btn_ja"):
        _label("training")
        st.rerun(scope="fragment")

    # ── Navigation ────────────────────────────────────────────────────────────
    col_back, col_hint, col_fwd = st.columns([1, 5, 1])
    if col_back.button("◀", width="stretch", key="btn_back"):
        _navigate(-1)
        st.rerun(scope="fragment")
    if col_fwd.button("▶", width="stretch", key="btn_fwd"):
        _navigate(1)
        st.rerun(scope="fragment")
    col_hint.markdown(
        "<div style='text-align:center;padding-top:0.6rem'>"
        "<span style='color:#45475a;font-size:0.72rem'>"
        "<kbd>Y</kbd> Ja &nbsp; <kbd>N</kbd> Nein &nbsp; <kbd>← →</kbd> Navigation"
        "</span></div>",
        unsafe_allow_html=True,
    )

    # ── Stats chips ───────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;gap:0.55rem;flex-wrap:wrap;margin:0.6rem 0 0.4rem;">
  <div style="background:#313244;border:1px solid #45475a;border-radius:8px;
              padding:0.3rem 0.8rem;font-size:0.82rem;">
    <span style="color:#a6e3a1;font-weight:700">{n_train}</span>
    <span style="color:#6c7086;margin-left:0.3rem">Training</span>
  </div>
  <div style="background:#313244;border:1px solid #45475a;border-radius:8px;
              padding:0.3rem 0.8rem;font-size:0.82rem;">
    <span style="color:#f38ba8;font-weight:700">{n_unused}</span>
    <span style="color:#6c7086;margin-left:0.3rem">Nicht genutzt</span>
  </div>
  <div style="background:#313244;border:1px solid #45475a;border-radius:8px;
              padding:0.3rem 0.8rem;font-size:0.82rem;">
    <span style="color:#f9e2af;font-weight:700">{n_open}</span>
    <span style="color:#6c7086;margin-left:0.3rem">Offen</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Class legend ──────────────────────────────────────────────────────────
    pills = ""
    for cls_id, name in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[cls_id]
        fg = "#1e1e2e" if (r * 0.299 + g * 0.587 + b * 0.114) > 140 else "#ffffff"
        pills += (f'<span style="background:rgb({r},{g},{b});color:{fg};'
                  f'padding:0.25rem 0.8rem;border-radius:6px;font-size:0.75rem;'
                  f'font-weight:700;letter-spacing:0.04em">{name}</span>')
    st.markdown(f'<div style="display:flex;gap:0.45rem;margin-bottom:0.5rem">{pills}</div>',
                unsafe_allow_html=True)

    # ── Recent activity ───────────────────────────────────────────────────────
    state = ss.label_state
    recent = (
        [(tid, "training") for tid in reversed(state["training"][-10:])] +
        [(tid, "unused")   for tid in reversed(state["unused"][-10:])]
    )
    recent.sort(key=lambda x: x[0], reverse=True)
    recent = recent[:10]

    if recent:
        st.markdown(
            "<p style='color:#6c7086;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.1em;font-weight:700;margin:0.75rem 0 0.4rem'>Zuletzt bewertet</p>",
            unsafe_allow_html=True,
        )
        rows = ""
        for tid, cat in recent:
            color     = "#a6e3a1" if cat == "training" else "#f38ba8"
            label_txt = "Training" if cat == "training" else "Nicht genutzt"
            rows += (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:0.35rem 0.75rem;border-radius:8px;background:#313244;'
                f'margin-bottom:0.22rem;border-left:3px solid {color};">'
                f'<code style="background:transparent;color:#89b4fa;font-size:0.8rem;padding:0">{tid}</code>'
                f'<span style="color:{color};font-size:0.75rem;font-weight:600">{label_txt}</span>'
                f'</div>'
            )
        st.markdown(rows, unsafe_allow_html=True)

    _maybe_prefetch(dataset_key, idx, all_ids, ss.image_ids, ss.mask_ids)

    # st.components.v1.html renders in a real iframe → window.parent is the
    # Streamlit app. Keyboard shortcuts and Nein-button styling both work
    # reliably this way. The guard on window.parent._lr prevents duplicate
    # listeners across fragment reruns.
    _stc.html("""
<script>
(function () {
    var p = window.parent;
    if (p._lr) return;
    p._lr = 1;
    var d = p.document;
    function styleNein() {
        d.querySelectorAll('button').forEach(function (b) {
            if ((b.textContent || '').indexOf('Nein') !== -1)
                b.classList.add('btn-nein');
        });
    }
    d.addEventListener('keydown', function (e) {
        var t = e.target.tagName;
        if (t==='INPUT'||t==='TEXTAREA'||t==='SELECT') return;
        if (e.metaKey||e.ctrlKey||e.altKey) return;
        function c(x) {
            var done = false;
            d.querySelectorAll('button').forEach(function (b) {
                if (!done && (b.textContent||'').indexOf(x) !== -1) {
                    b.dispatchEvent(new MouseEvent('click',{bubbles:true,cancelable:true}));
                    done = true;
                }
            });
        }
        if      (e.key==='y'||e.key==='Y') c('Ja');
        else if (e.key==='n'||e.key==='N') c('Nein');
        else if (e.key==='ArrowLeft')  { e.preventDefault(); c('◀'); }
        else if (e.key==='ArrowRight') { e.preventDefault(); c('▶'); }
    });
    new p.MutationObserver(styleNein)
        .observe(d.body, {childList:true, subtree:true});
    styleNein();
})();
</script>
""", height=0)

# ── Main ───────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Base ── */
.stApp { background: #1e1e2e !important; color: #cdd6f4; }
.block-container { padding: 0.75rem 2rem 2rem !important; max-width: 100% !important; }
.appview-container > section.main { padding-top: 0 !important; }
section[data-testid="stSidebar"] {
    background: #181825 !important;
    border-right: 1px solid #313244 !important;
    transform: none !important;
    min-width: 240px !important;
    max-width: 240px !important; }
section[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] {
    height: 0 !important; min-height: 0 !important;
    padding: 0 !important; overflow: visible !important;
    background: transparent !important; border: none !important; }
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
#MainMenu, footer { display: none !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: #313244 !important; border-radius: 99px !important; height: 7px !important; }
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #a6e3a1, #94e2d5) !important;
    border-radius: 99px !important; }

/* ── Ja button (primary) ── */
/* Use kind= attribute which is stable across Streamlit versions;
   data-testid changed from baseButton-* to stBaseButton-* in 1.36. */
button[kind="primary"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #a6e3a1 0%, #94e2d5 100%) !important;
    color: #1e1e2e !important; border: none !important; border-radius: 12px !important;
    font-size: 1.1rem !important; font-weight: 800 !important;
    letter-spacing: 0.02em !important; min-height: 3.4rem !important;
    box-shadow: 0 4px 24px rgba(166,227,161,0.28) !important;
    transition: transform 0.1s, box-shadow 0.1s !important; }
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(166,227,161,0.48) !important; }
button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active,
button[data-testid="stBaseButton-primary"]:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(166,227,161,0.2) !important; }

/* ── Secondary buttons ── */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"] {
    background: #313244 !important; color: #cdd6f4 !important;
    border: 1.5px solid #45475a !important; border-radius: 12px !important;
    font-size: 1.0rem !important; font-weight: 600 !important;
    min-height: 3.4rem !important;
    transition: all 0.1s !important; }
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #cba6f7 !important; color: #cba6f7 !important;
    background: rgba(203,166,247,0.08) !important; }
button[kind="secondary"]:active,
button[data-testid="baseButton-secondary"]:active,
button[data-testid="stBaseButton-secondary"]:active { transform: scale(0.97) !important; }

/* ── Nein button (class added via JS) ── */
.btn-nein,
.btn-nein:hover,
.btn-nein:focus {
    background: rgba(243,139,168,0.08) !important;
    border-color: #f38ba8 !important;
    color: #f38ba8 !important; }

/* ── kbd ── */
kbd {
    background: #313244; border: 1px solid #45475a; border-radius: 4px;
    padding: 0.05rem 0.35rem; font-size: 0.72rem; font-family: monospace; color: #a6adc8; }

/* ── Code ── */
code { background: #313244 !important; color: #89b4fa !important;
    border-radius: 5px !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #313244 !important; border-color: #45475a !important;
    border-radius: 8px !important; color: #cdd6f4 !important; }

/* ── Divider / hr ── */
hr { border-color: #313244 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] > div { border-top-color: #cba6f7 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] p { color: #bac2de; }
</style>
"""


def main() -> None:
    st.set_page_config(layout="wide", page_title="Tile Labeling Tool", page_icon="🗺️",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)

    datasets = _get_datasets()
    if not datasets:
        st.error("Keine Datensätze konfiguriert. Bitte `[drive.datasets]` in den Secrets prüfen.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='color:#cba6f7;font-size:0.7rem;text-transform:uppercase;"
            "letter-spacing:0.12em;font-weight:700;margin-bottom:0.5rem'>Datensatz</p>",
            unsafe_allow_html=True,
        )
        dataset_key = st.selectbox(
            "Datensatz",
            options=list(datasets.keys()),
            format_func=lambda k: datasets[k].get("label", k),
            label_visibility="collapsed",
        )

    if (
        "initialized" not in st.session_state
        or st.session_state.get("active_dataset") != dataset_key
    ):
        with st.spinner(f"Verbinde mit Google Drive …"):
            _initialize(dataset_key)

    # ── Sidebar stats ─────────────────────────────────────────────────────────
    ss = st.session_state
    if "initialized" in ss:
        total         = len(ss.all_ids)
        labeled_count = len(ss.training_set | ss.unused_set)
        n_train       = len(ss.training_set)
        n_unused      = len(ss.unused_set)
        n_open        = total - labeled_count
        pct           = int(labeled_count / total * 100) if total else 0
        with st.sidebar:
            st.markdown(
                f"""
<div style="margin-top:1.25rem">
  <p style="color:#6c7086;font-size:0.68rem;text-transform:uppercase;
            letter-spacing:0.1em;font-weight:700;margin-bottom:0.75rem">Fortschritt</p>
  <div style="display:grid;gap:0.5rem;margin-bottom:0.75rem">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span style="color:#a6adc8;font-size:0.85rem">Training</span>
      <span style="color:#a6e3a1;font-weight:700;font-size:0.95rem">{n_train}</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span style="color:#a6adc8;font-size:0.85rem">Nicht genutzt</span>
      <span style="color:#f38ba8;font-weight:700;font-size:0.95rem">{n_unused}</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span style="color:#a6adc8;font-size:0.85rem">Offen</span>
      <span style="color:#f9e2af;font-weight:700;font-size:0.95rem">{n_open}</span>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding-top:0.4rem;border-top:1px solid #313244">
      <span style="color:#6c7086;font-size:0.82rem">Gesamt</span>
      <span style="color:#585b70;font-size:0.82rem">{total}</span>
    </div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
            st.progress(labeled_count / total if total else 0)
            if _prefetch_active:
                st.markdown(
                    "<p style='color:#585b70;font-size:0.72rem;margin-top:0.5rem'>"
                    "⏳ Bilder werden vorgeladen …</p>",
                    unsafe_allow_html=True,
                )

    if _last_save_error:
        st.error(f"Speicherfehler: {_last_save_error}")

    _labeling_fragment(dataset_key)


if __name__ == "__main__":
    main()
