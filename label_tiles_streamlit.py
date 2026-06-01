"""
Streamlit-based tile labeling tool.
Images and masks are loaded from Google Drive; labels are persisted
to tile_labels.json in the Drive state folder.
"""

import io
import json
from pathlib import PurePosixPath

import numpy as np
import streamlit as st
from PIL import Image
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ── Visual settings ────────────────────────────────────────────────────────
PANEL_SIZE = 400
MASK_ALPHA = 0.45
STATE_FILE = "tile_labels.json"

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (65,  182, 230),   # Gebäude    – cyan-blue
    2: (255,  80,   0),   # Versiegelt – orange-red
}
CLASS_NAMES: dict[int, str] = {
    1: "Gebäude",
    2: "Versiegelt",
}

# ── Google Drive helpers ───────────────────────────────────────────────────

@st.cache_resource
def _drive_service():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


@st.cache_data(ttl=600)
def list_folder(folder_id: str) -> dict[str, str]:
    """Return {filename: file_id} for all non-trashed files in folder."""
    svc = _drive_service()
    files: dict[str, str] = {}
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            files[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


@st.cache_data
def download_bytes(file_id: str) -> bytes:
    """Download a Drive file by ID and return raw bytes (cached per file_id)."""
    svc = _drive_service()
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def _find_file(folder_id: str, name: str) -> str | None:
    svc = _drive_service()
    q = f"'{folder_id}' in parents and name='{name}' and trashed=false"
    resp = svc.files().list(q=q, fields="files(id)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def _upload_json(data: dict, folder_id: str, file_id: str | None) -> str:
    svc = _drive_service()
    content = json.dumps(data, indent=2).encode()
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json")
    if file_id:
        svc.files().update(fileId=file_id, media_body=media).execute()
        return file_id
    # Service Accounts haben keine eigene Storage-Quota und können keine neuen
    # Dateien erstellen. tile_labels.json muss einmalig manuell in den Drive-Ordner
    # hochgeladen und mit dem Service Account geteilt werden.
    st.error(
        f"**tile_labels.json nicht gefunden.**\n\n"
        f"Bitte eine leere Datei mit dem Inhalt `{{\"training\": [], \"unused\": []}}` "
        f"erstellen und in den Drive-Ordner mit der ID `{folder_id}` hochladen. "
        f"Danach die Seite neu laden."
    )
    st.stop()

# ── Dataset configuration ──────────────────────────────────────────────────

def _get_datasets() -> dict[str, dict]:
    """Return ordered dict of {key: dataset_config} from secrets."""
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

# ── State helpers ──────────────────────────────────────────────────────────

def _tile_status(tile_id: str) -> str:
    if tile_id in st.session_state.training_set:
        return "training"
    if tile_id in st.session_state.unused_set:
        return "unused"
    return "open"


def _save_state() -> None:
    ss = st.session_state
    cfg = st.secrets["drive"][ss.active_dataset]
    file_id = _upload_json(
        ss.label_state,
        cfg["state_folder_id"],
        ss.state_file_id,
    )
    ss.state_file_id = file_id


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

    _save_state()
    ss.current_index = min(idx + 1, len(ss.all_ids) - 1)


def _navigate(direction: int) -> None:
    ss = st.session_state
    ss.current_index = max(0, min(ss.current_index + direction, len(ss.all_ids) - 1))

# ── Initialization ─────────────────────────────────────────────────────────

def _initialize(dataset_key: str) -> None:
    ss  = st.session_state
    cfg = st.secrets["drive"][dataset_key]

    images = list_folder(cfg["images_folder_id"])
    masks  = list_folder(cfg["masks_folder_id"])

    img_map  = {PurePosixPath(n).stem: fid for n, fid in images.items() if n.endswith(".jpg")}
    mask_map = {PurePosixPath(n).stem: fid for n, fid in masks.items()  if n.endswith(".png")}
    common   = sorted(set(img_map) & set(mask_map))

    ss.all_ids   = common
    ss.image_ids = img_map
    ss.mask_ids  = mask_map

    state_folder = cfg["state_folder_id"]
    file_id = _find_file(state_folder, STATE_FILE)
    if file_id:
        state = json.loads(download_bytes(file_id))
    else:
        state = {"training": [], "unused": []}
        file_id = None

    ss.label_state   = state
    ss.state_file_id = file_id
    ss.training_set  = set(state.get("training", []))
    ss.unused_set    = set(state.get("unused",   []))

    labeled = ss.training_set | ss.unused_set
    idx = 0
    while idx < len(common) and common[idx] in labeled:
        idx += 1
    ss.current_index  = idx
    ss.active_dataset = dataset_key
    ss.initialized    = True

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

    # Re-initialize when the selected dataset changes
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

    # ── Composite image ──────────────────────────────────────────────────────
    img_bytes  = download_bytes(ss.image_ids[tile_id])
    mask_bytes = download_bytes(ss.mask_ids[tile_id])
    composite  = build_composite(img_bytes, mask_bytes, PANEL_SIZE, PANEL_SIZE)
    st.image(composite, width="stretch",
             caption="Originalbild  |  Überlagert  |  Maske")

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

    # ── Sidebar stats ─────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    main()
