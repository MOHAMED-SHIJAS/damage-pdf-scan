"""
Multimodal Document Rendering, Native PDF Stream & OCR Helper
=============================================================
File: agent/ocr_engine.py
Pure Python 3 standard library (+ Pillow / NumPy). Zero heavy agent frameworks.

Provides:
1. Deep Pure-Python PDF Binary & Stream Parser:
   - Balanced-parenthesis metadata extraction (Title, Author, Creator, Producer, Keywords, Dates).
   - Searchable text extraction (FlateDecode, BT/ET, Tj/TJ, ToUnicode CMaps, /URI links).
   - Embedded raster image reconstruction (DCTDecode JPEG + FlateDecode RGB/Gray + SMask RGBA).
2. High-Resolution Page Rendering & Local Offline OCR (Windows WinRT PDF + OCR APIs via PowerShell,
   gracefully skipped on non-Windows hosts).
3. Optional Live Multimodal Vision API Client (Gemini / OpenAI via native urllib.request).
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# 1. Pure-Python Deep PDF Metadata, Text Stream & Embedded Image Extractor
# ============================================================================

def _extract_balanced_pdf_str(raw: bytes, key: bytes) -> str:
    """Extracts a PDF Info dictionary literal string supporting nested parentheses."""
    pattern = key + rb'\s*\('
    m = re.search(pattern, raw)
    if not m:
        return ""
    idx = m.end()
    depth = 1
    out = bytearray()
    n = len(raw)
    while idx < n and depth > 0:
        b = raw[idx]
        if b == 0x5C and idx + 1 < n:  # backslash escape
            nxt = raw[idx + 1]
            if nxt in (0x28, 0x29, 0x5C):
                out.append(nxt)
                idx += 2
                continue
            elif nxt == 0x6E:  # \n
                out.append(0x0A)
                idx += 2
                continue
        if b == 0x28:  # '('
            depth += 1
            out.append(b)
        elif b == 0x29:  # ')'
            depth -= 1
            if depth == 0:
                break
            out.append(b)
        else:
            out.append(b)
        idx += 1

    # Decode UTF-16BE BOM if present, else latin1/utf-8
    raw_bytes = bytes(out)
    if raw_bytes.startswith(b"\xfe\xff"):
        return raw_bytes[2:].decode("utf-16-be", errors="ignore").strip()
    try:
        return raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw_bytes.decode("latin1", errors="ignore").strip()


def normalize_pdf_date(raw_date: str) -> Optional[str]:
    """Converts raw PDF date like 'D:20250813083139+00'00'' into ISO format."""
    if not raw_date:
        return None
    cleaned = raw_date.strip()
    if cleaned.startswith("D:"):
        cleaned = cleaned[2:]
    digits_match = re.match(r"^(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(.*)", cleaned)
    if not digits_match:
        return raw_date
    yr = digits_match.group(1)
    mo = digits_match.group(2) or "01"
    dy = digits_match.group(3) or "01"
    hr = digits_match.group(4) or "00"
    mn = digits_match.group(5) or "00"
    sc = digits_match.group(6) or "00"
    tz_raw = (digits_match.group(7) or "").replace("'", "").strip()
    tz_str = " UTC" if tz_raw in ("Z", "+0000", "-0000") else (f" {tz_raw}" if tz_raw else "")
    try:
        dt = datetime(int(yr), int(mo), int(dy), int(hr), int(mn), int(sc))
        return dt.strftime("%Y-%m-%d %H:%M:%S") + tz_str
    except ValueError:
        return f"{yr}-{mo}-{dy}"


def _parse_tounicode_cmaps(raw: bytes) -> Dict[str, str]:
    """Extracts basic hex->unicode mappings from ToUnicode CMap streams in a PDF."""
    char_map: Dict[str, str] = {}
    streams = re.findall(rb'stream[\r\n]+(.*?)endstream', raw, re.DOTALL)
    for st in streams:
        st_clean = st.rstrip(b'\r\n')
        try:
            dec = zlib.decompress(st_clean)
        except Exception:
            dec = st_clean
        if b'beginbfchar' in dec or b'beginbfrange' in dec:
            # Parse bfchar entries: <0003> <0020>
            for src_h, dst_h in re.findall(rb'<([0-9a-fA-F]+)>\s+<([0-9a-fA-F]+)>', dec):
                try:
                    u_bytes = bytes.fromhex(dst_h.decode("ascii"))
                    text_ch = u_bytes.decode("utf-16-be", errors="ignore")
                    if text_ch:
                        char_map[src_h.decode("ascii").upper()] = text_ch
                except Exception:
                    pass
    return char_map


def extract_pdf_native_details(pdf_path: str, assets_dir: str) -> Dict[str, Any]:
    """
    Parses a PDF file cleanly in Pure Python:
    - Info Metadata dictionary
    - Page count & MediaBox physical dimensions
    - Embedded hyperlinks (/URI)
    - Plain-text streams (BT..ET / Tj / TJ)
    - Embedded raster images (DCTDecode JPEG + FlateDecode RGB/Gray + SMask RGBA)
    """
    with open(pdf_path, "rb") as f:
        raw = f.read()

    # 1. PDF Version & Page Count
    ver_match = re.match(rb'%PDF-(\d+\.\d+)', raw[:32])
    pdf_version = ver_match.group(1).decode("ascii") if ver_match else "Unknown"

    pages = re.findall(rb'/Type\s*/Page\b', raw)
    page_count = max(len(pages), 1)

    # MediaBox dimensions
    mediaboxes = []
    for mb in re.findall(rb'/MediaBox\s*\[\s*([^\]]+)\]', raw):
        parts = mb.decode("latin1", errors="ignore").split()
        if len(parts) == 4:
            try:
                x0, y0, x1, y1 = [float(p) for p in parts]
                w_pt = round(abs(x1 - x0), 1)
                h_pt = round(abs(y1 - y0), 1)
                w_in = round(w_pt / 72.0, 2)
                h_in = round(h_pt / 72.0, 2)
                orient = "Square" if abs(w_pt - h_pt) < 5 else ("Landscape" if w_pt > h_pt else "Portrait")
                mediaboxes.append(f"{w_pt} x {h_pt} pt ({w_in} x {h_in} in, {orient})")
            except ValueError:
                pass

    # 2. Metadata Dictionary (balanced parens)
    metadata: Dict[str, str] = {}
    for key_label, pdf_key in [
        ("Title", b"/Title"),
        ("Author", b"/Author"),
        ("Subject", b"/Subject"),
        ("Keywords", b"/Keywords"),
        ("Creator", b"/Creator"),
        ("Producer", b"/Producer"),
        ("CreationDate", b"/CreationDate"),
        ("ModDate", b"/ModDate"),
        ("Language", b"/Lang"),
    ]:
        val = _extract_balanced_pdf_str(raw, pdf_key)
        if val:
            metadata[key_label] = val

    # 3. URIs / Hyperlinks
    uris = list(dict.fromkeys(
        u.decode("latin1", errors="ignore").strip()
        for u in re.findall(rb'/URI\s*\(([^)]+)\)', raw)
    ))

    # 4. Stream Text Extraction & Vector vs Text Diagnosis
    cmap = _parse_tounicode_cmaps(raw)
    streams = re.findall(rb'stream[\r\n]+(.*?)endstream', raw, re.DOTALL)
    extracted_text_chunks: List[str] = []
    has_vector_paths = False

    for st in streams:
        st_clean = st.rstrip(b'\r\n')
        try:
            dec = zlib.decompress(st_clean)
        except Exception:
            dec = st_clean

        if re.search(rb'\b(re\s+f|cm\s+q|c\s+l|m\s+l)\b', dec):
            has_vector_paths = True

        # Extract BT ... ET text blocks
        for bt in re.findall(rb'BT(.*?)ET', dec, re.DOTALL):
            # Literal strings: (Hello) Tj
            for lit in re.findall(rb'\(([^)\\]*(?:\\.[^)\\]*)*)\)\s*Tj', bt):
                txt = lit.replace(b'\\(', b'(').replace(b'\\)', b')').decode("latin1", errors="ignore").strip()
                if txt and re.search(r'[A-Za-z0-9]', txt):
                    extracted_text_chunks.append(txt)
            # TJ arrays: [(H) -10 (ello)] TJ
            for arr in re.findall(rb'\[([^\]]+)\]\s*TJ', bt):
                pieces = re.findall(rb'\(([^)\\]*(?:\\.[^)\\]*)*)\)', arr)
                joined = b"".join(pieces).replace(b'\\(', b'(').replace(b'\\)', b')').decode("latin1", errors="ignore").strip()
                if joined and re.search(r'[A-Za-z0-9]', joined):
                    extracted_text_chunks.append(joined)
            # Hex strings mapped via ToUnicode CMap
            if cmap:
                for hex_s in re.findall(rb'<([0-9a-fA-F]{4,})>\s*(?:Tj|TJ)', bt):
                    h_str = hex_s.decode("ascii").upper()
                    # Try 4-hex-digit chunks first
                    decoded_chars = []
                    for i in range(0, len(h_str), 4):
                        chunk = h_str[i:i+4]
                        if chunk in cmap:
                            decoded_chars.append(cmap[chunk])
                    if decoded_chars:
                        extracted_text_chunks.append("".join(decoded_chars).strip())

    # 5. Embedded Raster Images Reconstruction
    os.makedirs(assets_dir, exist_ok=True)
    embedded_images: List[Dict[str, Any]] = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_prefix = re.sub(r'[^A-Za-z0-9_\-]', '_', base_name)

    obj_pattern = rb'(\d+)\s+\d+\s+obj\s*<<(.*?)>>\s*stream[\r\n]+(.*?)endstream'
    pdf_objs: Dict[int, Tuple[str, bytes]] = {}
    for m in re.finditer(obj_pattern, raw, re.DOTALL):
        oid = int(m.group(1))
        hdr = m.group(2).decode("latin1", errors="ignore")
        stm = m.group(3).rstrip(b'\r\n')
        pdf_objs[oid] = (hdr, stm)

    try:
        from PIL import Image
        img_counter = 0
        for oid, (hdr, stm) in pdf_objs.items():
            if "/Subtype /Image" not in hdr and "/Subtype/Image" not in hdr:
                continue
            w_m = re.search(r'/Width\s+(\d+)', hdr)
            h_m = re.search(r'/Height\s+(\d+)', hdr)
            cs_m = re.search(r'/ColorSpace\s*/(\w+)', hdr)
            flt_m = re.search(r'/Filter\s*/(\w+)', hdr)
            smask_m = re.search(r'/SMask\s+(\d+)\s+\d+\s+R', hdr)
            if not (w_m and h_m):
                continue
            w_val, h_val = int(w_m.group(1)), int(h_m.group(1))
            cs_val = cs_m.group(1) if cs_m else "Unknown"
            flt_val = flt_m.group(1) if flt_m else "None"

            # Skip standalone SMask grayscale masks from separate saving if already paired
            if cs_val == "DeviceGray" and any(f"/SMask {oid} " in h2 for h2, _ in pdf_objs.values()):
                continue

            img_counter += 1
            asset_path = os.path.join(assets_dir, f"{safe_prefix}_asset_{img_counter}.png").replace("\\", "/")

            if flt_val == "DCTDecode":
                jpg_path = os.path.join(assets_dir, f"{safe_prefix}_asset_{img_counter}.jpg").replace("\\", "/")
                with open(jpg_path, "wb") as jf:
                    jf.write(stm)
                try:
                    with Image.open(jpg_path) as im:
                        im.save(asset_path, format="PNG")
                    os.remove(jpg_path)
                    embedded_images.append({
                        "object_id": oid,
                        "width": w_val,
                        "height": h_val,
                        "color_space": cs_val,
                        "encoding": "JPEG (DCTDecode)",
                        "path": asset_path,
                    })
                except Exception:
                    pass
            elif flt_val == "FlateDecode" and cs_val == "DeviceRGB":
                try:
                    rgb_bytes = zlib.decompress(stm)
                    if len(rgb_bytes) >= w_val * h_val * 3:
                        im = Image.frombytes("RGB", (w_val, h_val), rgb_bytes[:w_val * h_val * 3])
                        mode_lbl = "RGB"
                        if smask_m:
                            sm_id = int(smask_m.group(1))
                            if sm_id in pdf_objs:
                                try:
                                    alpha_bytes = zlib.decompress(pdf_objs[sm_id][1])
                                    if len(alpha_bytes) >= w_val * h_val:
                                        a_im = Image.frombytes("L", (w_val, h_val), alpha_bytes[:w_val * h_val])
                                        im.putalpha(a_im)
                                        mode_lbl = "RGBA (with Transparency Mask)"
                                except Exception:
                                    pass
                        im.save(asset_path, format="PNG")
                        embedded_images.append({
                            "object_id": oid,
                            "width": w_val,
                            "height": h_val,
                            "color_space": mode_lbl,
                            "encoding": "FlateDecode Raster Layer",
                            "path": asset_path,
                        })
                except Exception:
                    pass
    except ImportError:
        pass

    return {
        "pdf_version": pdf_version,
        "page_count": page_count,
        "page_dimensions": mediaboxes[0] if mediaboxes else "Standard",
        "metadata": metadata,
        "uris": uris,
        "native_text_lines": extracted_text_chunks,
        "has_vector_paths": has_vector_paths,
        "embedded_images": embedded_images,
    }


# ============================================================================
# 2. Cross-Platform WinRT High-Res PDF Rendering & Offline OCR (Windows)
# ============================================================================

_PS_RENDER_PDF = r"""
param([string]$PdfPath, [string]$OutImgDir, [int]$MaxPages = 3)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncAction' })[0]

function AwaitOp($WinRtTask, $ResultType) {
    $netTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}
function AwaitAct($WinRtTask) {
    $netTask = $asTaskAction.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
}

New-Item -ItemType Directory -Force -Path $OutImgDir | Out-Null
$absPath = (Resolve-Path $PdfPath).Path
$fs = [System.IO.File]::OpenRead($absPath)
$ras = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($fs)
$pdfDoc = AwaitOp ([Windows.Data.Pdf.PdfDocument]::LoadFromStreamAsync($ras)) ([Windows.Data.Pdf.PdfDocument])

$limit = [Math]::Min($pdfDoc.PageCount, $MaxPages)
for ($i = 0; $i -lt $limit; $i++) {
    $page = $pdfDoc.GetPage($i)
    $outPng = Join-Path $OutImgDir ("rendered_p" + ($i + 1) + ".png")
    $outFs = [System.IO.File]::Create($outPng)
    $outRas = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($outFs)
    $opts = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $targetW = [Math]::Max($page.Size.Width * 3.0, 2000.0)
    $scale = $targetW / $page.Size.Width
    $opts.DestinationWidth = [uint32]$targetW
    $opts.DestinationHeight = [uint32]($page.Size.Height * $scale)
    AwaitAct ($page.RenderToStreamAsync($outRas, $opts))
    try { $outRas.Dispose() } catch {}
    try { $outFs.Dispose() } catch {}
    try { $page.Dispose() } catch {}
    Write-Output ("PAGE_PNG=" + $outPng)
}
try { $ras.Dispose() } catch {}
try { $fs.Dispose() } catch {}
[Environment]::Exit(0)
"""

_PS_OCR_IMAGE = r"""
param([string]$ImgPath)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function AwaitOp($WinRtTask, $ResultType) {
    $netTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}

$absPath = (Resolve-Path $ImgPath).Path
$fs = [System.IO.File]::OpenRead($absPath)
$ras = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($fs)
$decoder = AwaitOp ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($ras)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = AwaitOp ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

$ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if (-not $ocrEngine) {
    $lang = New-Object Windows.Globalization.Language("en-US")
    $ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
}
$ocrRes = AwaitOp ($ocrEngine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

$outLines = @()
foreach ($line in $ocrRes.Lines) {
    $words = @()
    foreach ($w in $line.Words) {
        $words += @{
            text = $w.Text
            x = [Math]::Round($w.BoundingRect.X, 1)
            y = [Math]::Round($w.BoundingRect.Y, 1)
            w = [Math]::Round($w.BoundingRect.Width, 1)
            h = [Math]::Round($w.BoundingRect.Height, 1)
        }
    }
    $outLines += @{ text = $line.Text; words = $words }
}
@{ text = $ocrRes.Text; lines = $outLines } | ConvertTo-Json -Depth 5 -Compress
try { $ras.Dispose() } catch {}
try { $fs.Dispose() } catch {}
[Environment]::Exit(0)
"""


def render_pdf_to_images(pdf_path: str, out_dir: str, max_pages: int = 3) -> List[str]:
    """Renders PDF pages on Windows or through PyMuPDF on Linux/Render."""
    os.makedirs(out_dir, exist_ok=True)

    if sys.platform == "win32":
        script_path = os.path.join(tempfile.gettempdir(), "vision_agent_render_pdf.ps1")
        with open(script_path, "w", encoding="utf-8") as sf:
            sf.write(_PS_RENDER_PDF)

        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path,
                 "-PdfPath", os.path.abspath(pdf_path), "-OutImgDir", os.path.abspath(out_dir),
                 "-MaxPages", str(max_pages)],
                capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=25
            )
            rendered = []
            for line in proc.stdout.splitlines():
                if line.startswith("PAGE_PNG="):
                    p = line.split("=", 1)[1].strip().replace("\\", "/")
                    if os.path.exists(p):
                        rendered.append(p)
            if rendered:
                return rendered
        except Exception:
            pass

    # Cross-platform fallback for Linux/Render and as a Windows fallback.
    try:
        import fitz

        rendered: List[str] = []
        with fitz.open(pdf_path) as doc:
            for page_index in range(min(doc.page_count, max_pages)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                output_path = os.path.join(out_dir, f"page_{page_index + 1}.png").replace("\\", "/")
                pix.save(output_path)
                rendered.append(output_path)
        return rendered
    except Exception:
        return []


def run_local_ocr(image_path: str) -> Dict[str, Any]:
    """Runs Windows OCR locally; hosted Linux uses searchable text or Vision APIs."""
    if not os.path.exists(image_path):
        return {"text": "", "lines": []}

    if sys.platform == "win32":
        script_path = os.path.join(tempfile.gettempdir(), "vision_agent_ocr_img.ps1")
        with open(script_path, "w", encoding="utf-8") as sf:
            sf.write(_PS_OCR_IMAGE)

        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path,
                 "-ImgPath", os.path.abspath(image_path)],
                capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=25
            )
            stdout = proc.stdout.strip()
            if stdout.startswith("{"):
                result = json.loads(stdout)
                if result.get("text"):
                    return result
        except Exception:
            pass

    # Hosted Linux: PyMuPDF text extraction and Vision APIs handle OCR without loading
    # a large ONNX model that can exhaust a free Render instance.
    return {"text": "", "lines": []}

def reconstruct_spatial_rows(ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Groups OCR word bounding boxes into true horizontal rows sorted left-to-right.
    Solves fragmented form fields (e.g. 'Registration No :' + 'GGAZSAI011')
    and aligned multi-column tables.
    """
    words = []
    for line in ocr_data.get("lines", []):
        for w in line.get("words", []):
            txt = str(w.get("text", "")).strip()
            if txt:
                words.append({
                    "text": txt,
                    "x": float(w.get("x", 0)),
                    "y": float(w.get("y", 0)),
                    "w": float(w.get("w", 0)),
                    "h": max(float(w.get("h", 12)), 8.0),
                    "yc": float(w.get("y", 0)) + float(w.get("h", 12)) / 2.0,
                })
    if not words:
        return []

    words.sort(key=lambda item: (item["yc"], item["x"]))
    rows: List[List[Dict[str, Any]]] = []
    for w in words:
        if not rows:
            rows.append([w])
            continue
        last_row = rows[-1]
        avg_yc = sum(item["yc"] for item in last_row) / len(last_row)
        avg_h = sum(item["h"] for item in last_row) / len(last_row)
        if abs(w["yc"] - avg_yc) <= max(avg_h * 0.65, 10.0):
            last_row.append(w)
        else:
            rows.append([w])

    reconstructed = []
    for r in rows:
        r.sort(key=lambda item: item["x"])
        row_text = " ".join(item["text"] for item in r)
        # Compute segments separated by large horizontal gaps (column cells / Key vs Value)
        cells: List[Dict[str, Any]] = []
        current_cell_words = [r[0]]
        for idx in range(1, len(r)):
            prev = r[idx - 1]
            curr = r[idx]
            gap = curr["x"] - (prev["x"] + prev["w"])
            gap_thresh = max(prev["h"] * 1.6, 32.0)
            if gap > gap_thresh:
                cells.append({
                    "text": " ".join(item["text"] for item in current_cell_words),
                    "x": current_cell_words[0]["x"],
                    "x_end": current_cell_words[-1]["x"] + current_cell_words[-1]["w"],
                })
                current_cell_words = [curr]
            else:
                current_cell_words.append(curr)
        cells.append({
            "text": " ".join(item["text"] for item in current_cell_words),
            "x": current_cell_words[0]["x"],
            "x_end": current_cell_words[-1]["x"] + current_cell_words[-1]["w"],
        })

        reconstructed.append({
            "y": round(sum(item["yc"] for item in r) / len(r), 1),
            "text": row_text,
            "cells": cells,
            "words": r,
        })
    return reconstructed


# ============================================================================
# 3. Optional Live Multimodal Vision API Extraction (Gemini / OpenAI)
# ============================================================================

def extract_with_vision_api(
    file_path: str,
    enhanced_image_paths: List[str],
    api_key: Optional[str] = None,
    provider: str = "auto",
) -> Optional[Dict[str, Any]]:
    """
    If a Gemini or OpenAI API key is configured, sends the enhanced document directly
    to a multimodal model with strict instructions to extract ONLY authentic details
    and sort them into suitable categories. Returns None if no key is provided.
    """
    gemini_key = api_key if (api_key and (provider == "gemini" or (provider == "auto" and api_key.startswith("AIza")))) else os.environ.get("GEMINI_API_KEY")
    openai_key = api_key if (api_key and (provider == "openai" or (provider == "auto" and api_key.startswith("sk-")))) else os.environ.get("OPENAI_API_KEY")

    if not gemini_key and not openai_key:
        return None

    system_prompt = (
        "You are a precision document scanner. Analyze the attached document carefully.\n"
        "RULES:\n"
        "1. Extract ONLY details that are genuinely visible in this document. NEVER invent, guess, or assume fields that are not shown.\n"
        "2. Enhance/clean any blurry, faint, skewed, or noisy text readings, and list each quality enhancement or correction made in 'enhancements_applied'.\n"
        "3. Sort all extracted details into suitable, domain-appropriate categories (e.g., Institution/Organization, Candidate/Customer, Identifiers, Dates/Timeline, Financial/Academic Scores, Line Items/Subjects, Design/Visual Details).\n"
        "4. Return STRICT JSON ONLY matching this schema:\n"
        "{\n"
        '  "document_domain": "<accurate domain label>",\n'
        '  "summary": "<1-2 sentence accurate summary of document>",\n'
        '  "enhancements_applied": ["<list of clarity/OCR fixes applied>"],\n'
        '  "categorized_details": {\n'
        '    "<Category Name with Emoji>": { "<Field Name>": "<Extracted Value>" }\n'
        "  }\n"
        "}"
    )

    ext = os.path.splitext(file_path)[1].lower()

    if gemini_key:
        parts: List[Dict[str, Any]] = [{"text": system_prompt}]
        target_mime = "application/pdf" if ext == ".pdf" else "image/png"
        target_file = file_path if ext == ".pdf" else (enhanced_image_paths[0] if enhanced_image_paths else file_path)
        if os.path.exists(target_file):
            with open(target_file, "rb") as tf:
                b64 = base64.b64encode(tf.read()).decode("ascii")
            parts.append({"inline_data": {"mime_type": target_mime, "data": b64}})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        gemini_models = ("gemini-2.5-flash", "gemini-flash-latest")
        for model_name in gemini_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    raw_txt = body["candidates"][0]["content"]["parts"][0]["text"]
                    result = json.loads(raw_txt)
                    if isinstance(result.get("categorized_details"), dict):
                        print(f"Vision extraction succeeded with Gemini model: {model_name}")
                        return result
            except urllib.error.HTTPError as exc:
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    error_body = ""
                print(f"Gemini model {model_name} HTTP {exc.code}: {error_body}")
            except Exception as exc:
                print(f"Gemini model {model_name} failed: {type(exc).__name__}: {exc}")

    if openai_key and enhanced_image_paths:
        content_items: List[Dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        for img_p in enhanced_image_paths[:2]:
            if os.path.exists(img_p):
                with open(img_p, "rb") as imf:
                    b64 = base64.b64encode(imf.read()).decode("ascii")
                content_items.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content_items}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_txt = body["choices"][0]["message"]["content"]
                result = json.loads(raw_txt)
                if isinstance(result.get("categorized_details"), dict):
                    print("Vision extraction succeeded with OpenAI")
                    return result
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                error_body = ""
            print(f"OpenAI vision HTTP {exc.code}: {error_body}")
        except Exception as exc:
            print(f"OpenAI vision failed: {type(exc).__name__}: {exc}")

    if not gemini_key and not openai_key:
        print("No GEMINI_API_KEY or OPENAI_API_KEY is configured")
    return None

