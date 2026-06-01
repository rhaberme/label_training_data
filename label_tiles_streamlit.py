"""
Streamlit-based tile labeling tool.
Images and masks are loaded from Google Drive; labels are persisted
to tile_labels.json in the Drive state folder.
"""

import io
import json
import threading
from pathlib import PurePosixPath
from typing import Any

import requests as _requests
import numpy as np
import streamlit as st
from PIL import Image
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

# ── Visual settings ────────────────────────────────────────────────────────
PANEL_SIZE = 400
MASK_ALPHA = 0.45

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (65,  182, 230),   # Gebäude    – cyan-blue
    2: (255,  80,   0),   # Versiegelt – orange-red
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

# ── Caches ─────────────────────────────────────────────────────────────────
# All module-level so they survive Streamlit reruns.

_bytes_cache: dict[str, bytes]     = {}   # file_id → raw bytes
_composite_cache: dict[str, bytes] = {}   # tile_id → JPEG bytes of composite

_bytes_cache_lock     = threading.RLock()
_composite_cache_lock = threading.RLock()

# Populated once in _session() so background threads can build their own client.
_svc_account_info: dict = {}

# Serialises background Drive uploads so fast clicks never corrupt state.
_save_lock = threading.Lock()

# ── Drive REST endpoints ───────────────────────────────────────────────────
_FILES_URL  = "https://www.googleapis.com/drive/v3/files"
_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# ── Google Drive session helpers ───────────────────────────────────────────

def _make_session(info: dict) -> AuthorizedSession:
    creds = Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    session = AuthorizedSession(creds)
    # Ignore http_proxy / https_proxy env-vars (e.g. set by PyCharm),
    # which cause ssl.SSLError: WRONG_VERSION_NUMBER.
    session.trust_env = False
    return session


@st.cache_resource
def _session() -> AuthorizedSession:
    global _svc_account_info
    _svc_account_info = dict(st.secrets["gcp_service_account"])
    return _make_session(_svc_account_info)


def _thread_session() -> AuthorizedSession:
    """Fresh, independent session for background threads."""
    return _make_session(_svc_account_info)

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
    dataset_keys: list[str] = list(drive_cfg.get("datasets", []))
    return {k: drive_cfg[k] for k in dataset_keys}

# ── Image processing ───────────────────────────────────────────────────────

def build_composite(img_bytes: bytes, mask_bytes: bytes,
                    panel_w: int, panel_h: int) -> Image.Image:
    img  = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((panel_w, panel_h), Image.LANCZOS)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L").resize((panel_w, panel_h), Image.NEAREST)

    orig     = np.array(img,  dtype=np.float32)
    mask_arr = np.array(mask)

    blended = orig.copy()
    for cls_id, color in CLASS_COLORS.items():
        fg = mask_arr == cls_id
        if fg.any():
            c = np.array(color, dtype=np.float32)
            blended[fg] = blended[fg] * (1 - MASK_ALPHA) + c * MASK_ALPHA

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
    """Return cached JPEG bytes of the composite, building it if necessary."""
    with _composite_cache_lock:
        if tile_id in _composite_cache:
            return _composite_cache[tile_id]
    img_bytes  = download_bytes(img_fid)
    mask_bytes = download_bytes(mask_fid)
    pil = build_composite(img_bytes, mask_bytes, PANEL_SIZE, PANEL_SIZE)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    with _composite_cache_lock:
        _composite_cache[tile_id] = data
    return data

# ── Prefetch ───────────────────────────────────────────────────────────────

def _prefetch_worker(
    tile_ids: list[str],
    image_ids: dict[str, str],
    mask_ids: dict[str, str],
    batch_key: str,
) -> None:
    """Download bytes and pre-build composites for a batch in the background."""
    try:
        session = _thread_session()
    except Exception:
        with _prefetch_lock:
            _prefetch_active.discard(batch_key)
        return

    for tid in tile_ids:
        img_fid  = image_ids.get(tid)
        mask_fid = mask_ids.get(tid)

        # Download image
        img_bytes = None
        if img_fid:
            with _bytes_cache_lock:
                img_bytes = _bytes_cache.get(img_fid)
            if img_bytes is None:
                try:
                    img_bytes = _fetch_bytes(session, img_fid)
                    with _bytes_cache_lock:
                        _bytes_cache[img_fid] = img_bytes
                except Exception:
                    continue

        # Download mask
        mask_bytes = None
        if mask_fid:
            with _bytes_cache_lock:
                mask_bytes = _bytes_cache.get(mask_fid)
            if mask_bytes is None:
                try:
                    mask_bytes = _fetch_bytes(session, mask_fid)
                    with _bytes_cache_lock:
                        _bytes_cache[mask_fid] = mask_bytes
                except Exception:
                    continue

        # Pre-build composite so the main thread only needs to show it
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
    batches_to_check = [idx // PREFETCH_BATCH]
    if PREFETCH_BATCH - (idx % PREFETCH_BATCH) <= PREFETCH_TRIGGER:
        batches_to_check.append(idx // PREFETCH_BATCH + 1)

    for batch_num in batches_to_check:
        batch_key = f"{dataset_key}_{batch_num}"
        with _prefetch_lock:
            if batch_key in _prefetch_active:
                continue
            start = batch_num * PREFETCH_BATCH
            end   = min(start + PREFETCH_BATCH, len(all_ids))
            batch_tile_ids = all_ids[start:end]
            if not batch_tile_ids:
                continue
            _prefetch_active.add(batch_key)

        t = threading.Thread(
            target=_prefetch_worker,
            args=(batch_tile_ids, image_ids, mask_ids, batch_key),
            daemon=True,
        )
        t.start()

# ── State helpers ──────────────────────────────────────────────────────────

def _tile_status(tile_id: str) -> str:
    if tile_id in st.session_state.training_set:
        return "training"
    if tile_id in st.session_state.unused_set:
        return "unused"
    return "open"


def _save_state_async() -> None:
    """Upload label state to Drive in a background thread (non-blocking)."""
    ss = st.session_state
    cfg = st.secrets["drive"][ss.active_dataset]
    state_copy = {
        "training": list(ss.label_state["training"]),
        "unused":   list(ss.label_state["unused"]),
    }
    file_id   = ss.state_file_id
    folder_id = cfg["state_folder_id"]
    filename  = ss.state_filename

    def _do() -> None:
        with _save_lock:   # serialise so rapid clicks never interleave uploads
            try:
                if not file_id:
                    return
                content = json.dumps(state_copy, indent=2).encode()
                session = _thread_session()
                resp = session.patch(
                    f"{_UPLOAD_URL}/{file_id}",
                    params={"uploadType": "media"},
                    data=content,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
            except Exception:
                pass   # silent – next successful save will catch up

    threading.Thread(target=_do, daemon=True).start()


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
    _save_state_async()   # non-blocking: Drive upload runs in background


def _navigate(direction: int) -> None:
    ss = st.session_state
    ss.current_index = max(0, min(ss.current_index + direction, len(ss.all_ids) - 1))

# ── Initialization ─────────────────────────────────────────────────────────

def _initialize(dataset_key: str) -> None:
    ss  = st.session_state
    cfg = st.secrets["drive"][dataset_key]

    _session()   # ensure _svc_account_info is set before threads start

    images = list_folder(cfg["images_folder_id"])
    masks  = list_folder(cfg["masks_folder_id"])

    img_map  = {PurePosixPath(n).stem: fid for n, fid in images.items() if n.endswith(".jpg")}
    mask_map = {PurePosixPath(n).stem: fid for n, fid in masks.items()  if n.endswith(".png")}
    common   = sorted(set(img_map) & set(mask_map))

    ss.all_ids   = common
    ss.image_ids = img_map
    ss.mask_ids  = mask_map

    state_filename = f"tile_labels_{dataset_key}.json"
    state_folder   = cfg["state_folder_id"]
    file_id = _find_file(state_folder, state_filename)
    if file_id:
        state = json.loads(download_bytes(file_id))
    else:
        state = {"training": [], "unused": []}
        file_id = None

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

# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(layout="wide", page_title="Tile Labeling Tool", page_icon="🗺️")

    st.markdown("""
    <style>
    .stApp { background-color: #1e1e2e; color: #cdd6f4; }
    .block-container { padding-top: 1rem; max-width: 100%; }
    [data-testid="stImage"] img { border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Dataset selector ─────────────────────────────────────────────────────
    datasets = _get_datasets()
    if not datasets:
        st.error("Keine Datensätze in den Secrets konfiguriert. Bitte `[drive.datasets]` prüfen.")
        return

    with st.sidebar:
        st.markdown("### Datensatz")
        dataset_key = st.selectbox(
            "Datensatz auswählen",
            options=list(datasets.keys()),
            format_func=lambda k: datasets[k].get("label", k),
            label_visibility="collapsed",
        )

    if (
        "initialized" not in st.session_state
        or st.session_state.get("active_dataset") != dataset_key
    ):
        label = datasets[dataset_key].get("label", dataset_key)
        with st.spinner(f"Verbinde mit Google Drive – {label} …"):
            _initialize(dataset_key)

    ss      = st.session_state
    all_ids = ss.all_ids
    idx     = ss.current_index

    if not all_ids:
        st.error("Keine Tiles gefunden. Prüfe die Ordner-IDs in den Secrets.")
        return

    if idx >= len(all_ids):
        st.success("Alle Tiles wurden bewertet!")
        return

    tile_id = all_ids[idx]
    status  = _tile_status(tile_id)

    labeled_count = len(ss.training_set | ss.unused_set)
    total         = len(all_ids)

    # ── Header ──────────────────────────────────────────────────────────────
    BADGE = {
        "training": ("TRAINING",      "#a6e3a1"),
        "unused":   ("NICHT GENUTZT", "#f38ba8"),
        "open":     ("OFFEN",         "#f9e2af"),
    }
    badge_text, badge_color = BADGE[status]

    col_id, col_badge, col_prog = st.columns([3, 2, 1])
    col_id.markdown(f"**ID:** `{tile_id}`")
    col_badge.markdown(
        f"<span style='color:{badge_color};font-weight:bold'>{badge_text}</span>",
        unsafe_allow_html=True,
    )
    col_prog.markdown(f"**{labeled_count} / {total}**")

    st.progress(labeled_count / total if total else 0)

    # ── Composite image (from cache if prefetch already built it) ────────────
    jpeg = _composite_jpeg(tile_id, ss.image_ids[tile_id], ss.mask_ids[tile_id])
    st.image(jpeg, width="stretch", caption="Originalbild  |  Überlagert  |  Maske")

    # ── Label buttons ────────────────────────────────────────────────────────
    col_nein, col_ja = st.columns(2)
    if col_nein.button("✗  Nein", width="stretch", key="btn_nein"):
        _label("unused")
        st.rerun()
    if col_ja.button("✓  Ja", width="stretch", type="primary", key="btn_ja"):
        _label("training")
        st.rerun()

    # ── Navigation ───────────────────────────────────────────────────────────
    col_back, _, col_fwd = st.columns([1, 5, 1])
    if col_back.button("◀  Zurück", width="stretch", key="btn_back"):
        _navigate(-1)
        st.rerun()
    if col_fwd.button("Weiter  ▶", width="stretch", key="btn_fwd"):
        _navigate(1)
        st.rerun()

    # ── Stats ────────────────────────────────────────────────────────────────
    n_train  = len(ss.training_set)
    n_unused = len(ss.unused_set)
    n_open   = total - labeled_count
    st.markdown(
        f"Training: **{n_train}**  |  Nicht genutzt: **{n_unused}**  |  Offen: **{n_open}**"
    )

    # ── Class legend ─────────────────────────────────────────────────────────
    legend_parts = []
    for cls_id, name in CLASS_NAMES.items():
        r, g, b = CLASS_COLORS[cls_id]
        legend_parts.append(
            f"<span style='background:rgb({r},{g},{b});padding:2px 8px;"
            f"border-radius:3px;color:#11111b;font-weight:bold'>{name}</span>"
        )
    st.markdown("&nbsp;&nbsp;".join(legend_parts), unsafe_allow_html=True)

    # ── Recent activity ───────────────────────────────────────────────────────
    state = ss.label_state
    recent = (
        [(tid, "training") for tid in reversed(state["training"][-10:])] +
        [(tid, "unused")   for tid in reversed(state["unused"][-10:])]
    )
    recent.sort(key=lambda x: x[0], reverse=True)
    recent = recent[:10]

    if recent:
        st.divider()
        st.caption("Zuletzt bewertet")
        for tid, cat in recent:
            color = "#a6e3a1" if cat == "training" else "#f38ba8"
            label = "Training" if cat == "training" else "Nicht genutzt"
            st.markdown(
                f"<span style='color:{color}'>`{tid}`  →  {label}</span>",
                unsafe_allow_html=True,
            )

    # ── Sidebar stats + prefetch indicator ───────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("### Fortschritt")
        st.markdown(
            f"Training: **{n_train}**  \n"
            f"Nicht genutzt: **{n_unused}**  \n"
            f"Offen: **{n_open}**  \n"
            f"Gesamt: **{total}**"
        )
        st.progress(labeled_count / total if total else 0)

        if _prefetch_active:
            st.caption("⏳ Bilder werden vorgeladen …")

    _maybe_prefetch(dataset_key, idx, all_ids, ss.image_ids, ss.mask_ids)


if __name__ == "__main__":
    main()
