# pdf_extract.py
# -*- coding: utf-8 -*-
"""
Extractor de texto de PDFs con fallback OCR y utilidades de imágenes.

Características:
- Extracción directa (PyMuPDF) + OCR (Tesseract vía pytesseract) por página.
- Selección automática del binario de Tesseract (PATH / TESSERACT_CMD / rutas típicas / --tess-cmd).
- Parámetros Tesseract configurables y autotune de --psm por página (4/6/11).
- Guardado de capturas de páginas (all/ocr/none) y extracción de imágenes incrustadas.
- Export de layouts HTML por página con imágenes recortadas ubicadas en su posición.
"""

import argparse
import sys
import os
import pathlib
import logging
import shutil
import json
from typing import Optional, List

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: falta 'pymupdf'. Instala dependencias con: pip install -r requirements.txt")
    sys.exit(1)

# OCR opcional
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# --------------------------- Logging ---------------------------

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s"
    )


# ------------------ Tesseract binary selection -----------------

def configure_tesseract(explicit_cmd: Optional[str] = None) -> Optional[str]:
    """
    Selecciona el ejecutable de Tesseract en este orden:
    1) explicit_cmd (CLI)
    2) $TESSERACT_CMD (variable de entorno)
    3) lo que haya en PATH (shutil.which)
    4) rutas típicas en Windows
    """
    if not OCR_AVAILABLE:
        logging.debug("pytesseract/Pillow no instalados; se omite configuración de Tesseract.")
        return None

    candidates: List[str] = []

    if explicit_cmd:
        candidates.append(explicit_cmd)

    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        candidates.append(env_cmd)

    which_cmd = shutil.which("tesseract")
    if which_cmd:
        candidates.append(which_cmd)

    candidates += [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]

    for c in candidates:
        if c and os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            logging.info(f"Usando Tesseract: {c}")
            return c

    logging.error("No se encontró Tesseract. Define $TESSERACT_CMD, usa --tess-cmd o añade Tesseract al PATH.")
    return None


# --------------------------- Utilities -------------------------

def find_pdfs(input_path: pathlib.Path) -> List[pathlib.Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    elif input_path.is_dir():
        return sorted([p for p in input_path.rglob("*.pdf") if p.is_file()])
    else:
        return []


def ensure_outdir(outdir: Optional[pathlib.Path]) -> pathlib.Path:
    if outdir is None:
        outdir = pathlib.Path.cwd() / "output_txt"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def page_needs_ocr(text: str, threshold_chars: int = 20) -> bool:
    """
    Heurística simple: si la página devuelve menos de 'threshold_chars'
    la consideramos "escaneada" y hacemos OCR.
    """
    if text is None:
        return True
    stripped = "".join(text.split())
    return len(stripped) < threshold_chars


def make_images_outdir(base_outdir: pathlib.Path, pdf_stem: str) -> pathlib.Path:
    outdir = base_outdir / pdf_stem
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def ensure_dir(p: pathlib.Path) -> pathlib.Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_page_capture(page, outdir: pathlib.Path, page_num: int, dpi: int = 300) -> pathlib.Path:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path = outdir / f"page_{page_num:04d}.png"
    pix.save(out_path.as_posix())
    return out_path


def extract_embedded_images_from_page(doc, page, outdir: pathlib.Path, page_num: int) -> int:
    """
    Extrae imágenes incrustadas (no renders) de una página.
    Devuelve cuántas extrajo.
    """
    images = page.get_images(full=True)
    count = 0
    if not images:
        return 0
    emb_dir = outdir / "embedded"
    emb_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images, start=1):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
            ext = base.get("ext", "png")
            img_bytes = base["image"]
            out_path = emb_dir / f"p{page_num:04d}_img{idx}.{ext}"
            with open(out_path, "wb") as fh:
                fh.write(img_bytes)
            count += 1
        except Exception as e:
            logging.debug(f"Error extrayendo imagen incrustada p{page_num} xref {xref}: {e}")
    return count


# --------- Layout: localizar y colocar imágenes en HTML --------

def get_image_blocks(page) -> list:
    """
    Devuelve lista de dicts con bloques de imagen de la página:
    [{'bbox': (x0,y0,x1,y1), 'number': idx}, ...]
    """
    info = page.get_text("dict")
    blocks = info.get("blocks", [])
    imgs = []
    for b in blocks:
        if b.get("type") == 1:  # 1 = imagen
            bbox = b.get("bbox")
            if bbox and len(bbox) == 4:
                imgs.append({
                    "bbox": tuple(bbox),
                    "number": b.get("number", None)
                })
    return imgs


def render_full_page_png(page, outdir: pathlib.Path, page_num: int, dpi: int) -> pathlib.Path:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_path = outdir / f"page_{page_num:04d}.png"
    pix.save(out_path.as_posix())
    return out_path


def crop_image_region(page, bbox, dpi: int) -> "fitz.Pixmap":
    """
    Recorta la región bbox de la página al DPI indicado y devuelve un Pixmap.
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    rect = fitz.Rect(*bbox)
    return page.get_pixmap(matrix=mat, clip=rect, alpha=False)


def export_page_layout_with_images(doc, page, outdir: pathlib.Path, page_num: int, dpi: int):
    """
    Genera por página:
      - PNG de página completa
      - Recortes PNG de cada imagen (según bbox)
      - HTML con posicionamiento absoluto
      - manifest.json (metadatos)
    """
    page_dir = ensure_dir(outdir / f"page_{page_num:04d}")
    full_png = render_full_page_png(page, page_dir, page_num, dpi)

    # Tamaño de página en puntos y en px al dpi elegido
    page_rect = page.rect  # en puntos (72 dpi)
    zoom = dpi / 72.0
    page_px_w = int(page_rect.width * zoom)
    page_px_h = int(page_rect.height * zoom)

    # Localiza bloques de imagen
    img_blocks = get_image_blocks(page)

    manifest = {
        "page": page_num,
        "dpi": dpi,
        "page_points": [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1],
        "page_pixels": [page_px_w, page_px_h],
        "images": []
    }

    html_parts = []
    html_parts.append(f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Página {page_num:04d}</title>
<style>
  .canvas {{
    position: relative;
    width: {page_px_w}px;
    height: {page_px_h}px;
    background: url("./page_{page_num:04d}.png") top left no-repeat;
    background-size: contain;
    border: 1px solid #ddd;
    box-sizing: content-box;
  }}
  .imgblock {{
    position: absolute;
    box-shadow: 0 0 0 1px rgba(0,0,0,.12);
  }}
</style>
</head>
<body>
<div class="canvas">""")

    crops_dir = ensure_dir(page_dir / "crops")

    for idx, blk in enumerate(img_blocks, start=1):
        bbox = blk["bbox"]  # x0,y0,x1,y1 en puntos
        x0, y0, x1, y1 = bbox
        w_pt = x1 - x0
        h_pt = y1 - y0

        # Escala a píxeles
        left_px = int(x0 * (dpi / 72.0))
        top_px  = int(y0 * (dpi / 72.0))
        w_px    = int(w_pt * (dpi / 72.0))
        h_px    = int(h_pt * (dpi / 72.0))

        # Recorte de la región
        try:
            pix = crop_image_region(page, bbox, dpi=dpi)
            crop_name = f"img_{idx:03d}.png"
            crop_path = crops_dir / crop_name
            pix.save(crop_path.as_posix())
        except Exception as e:
            logging.debug(f"p{page_num} img{idx}: error al recortar: {e}")
            continue

        manifest["images"].append({
            "index": idx,
            "bbox_points": [x0, y0, x1, y1],
            "bbox_pixels": [left_px, top_px, left_px + w_px, top_px + h_px],
            "size_pixels": [w_px, h_px],
            "file": f"./crops/{crop_name}"
        })

        # IMG posicionada en el HTML
        html_parts.append(
            f'<img class="imgblock" src="./crops/{crop_name}" '
            f'style="left:{left_px}px; top:{top_px}px; width:{w_px}px; height:{h_px}px;" '
            f'alt="img{idx:03d}">'
        )

    html_parts.append("</div>\n</body>\n</html>")
    html_html = "\n".join(html_parts)

    # Escribe HTML y manifest
    html_path = page_dir / f"page_{page_num:04d}.html"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html_html)

    manifest_path = page_dir / f"page_{page_num:04d}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    logging.info(f"Layout exportado: {html_path}")


# ---------------------- Heurística de layout -------------------

def _count_text_blocks(page) -> int:
    """Cuenta bloques de texto (aproximado) para decidir layout."""
    try:
        d = page.get_text("dict")
        blocks = d.get("blocks", [])
        text_blocks = 0
        for b in blocks:
            if b.get("type") == 0:  # texto
                lines = b.get("lines", [])
                if any(len(l.get("spans", [])) > 0 for l in lines):
                    text_blocks += 1
        return text_blocks
    except Exception:
        return 0


def pick_psm_for_page(page) -> int:
    """
    Heurística simple:
      - Muchos bloques (>=6) o disposición fragmentada -> psm 4 (columnas/layout libre)
      - Pocos bloques (2–5) -> psm 6 (bloque uniforme)
      - Muy escaso texto (<=1) -> psm 11 (texto disperso/sin orden claro)
    """
    blocks = _count_text_blocks(page)
    if blocks >= 6:
        return 4
    elif blocks <= 1:
        return 11
    else:
        return 6


# ----------------------------- OCR -----------------------------

def ocr_page(page,
             dpi: int = 300,
             lang: str = "spa",
             tess_config: str = "--oem 1 --psm 6",
             override_psm: Optional[int] = None) -> str:
    if not OCR_AVAILABLE:
        logging.warning("OCR no disponible (pytesseract/Pillow no instalados).")
        return ""

    # Render a imagen en alta resolución
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    # Convertir a PIL.Image sin pasar por disco
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Construir config final (sustituir --psm si override_psm está definido)
    cfg = tess_config.strip()
    if override_psm is not None:
        parts = cfg.split()
        cleaned = []
        skip_next = False
        for i, p in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
            if p == "--psm":
                skip_next = True
                continue
            if p.startswith("--psm"):
                continue
            cleaned.append(p)
        cfg = " ".join(cleaned).strip()
        if cfg:
            cfg += f" --psm {override_psm}"
        else:
            cfg = f"--psm {override_psm}"

    # OCR
    try:
        text = pytesseract.image_to_string(img, lang=lang, config=cfg)
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract no encontrado. Configura TESSERACT_CMD, usa --tess-cmd o ajusta el PATH.")
        return ""
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""
    return text


# ---------------------- Core por documento ---------------------

def extract_text_from_pdf(pdf_path: pathlib.Path,
                          outdir: pathlib.Path,
                          force_ocr: bool = False,
                          disable_ocr: bool = False,
                          ocr_lang: str = "spa",
                          dpi: int = 300,
                          min_chars_threshold: int = 20,
                          tess_config: str = "--oem 1 --psm 6",
                          tess_autotune: bool = False,
                          save_pages: str = "none",               # none|ocr|all
                          images_outdir: Optional[pathlib.Path] = None,
                          extract_embedded: bool = False,
                          place_images: bool = False) -> pathlib.Path:
    logging.info(f"Procesando: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"No se pudo abrir {pdf_path.name}: {e}")
        return pathlib.Path()

    img_base_dir = images_outdir if images_outdir else (pathlib.Path.cwd() / "output_images")
    pdf_img_root = make_images_outdir(img_base_dir, pdf_path.stem)

    all_text: List[str] = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text") if not force_ocr else ""
        except Exception as e:
            logging.debug(f"get_text falló en página {i}: {e}")
            text = ""

        did_ocr = False
        if force_ocr or page_needs_ocr(text, min_chars_threshold):
            if disable_ocr:
                logging.debug(f"Pág {i}: OCR deshabilitado; se mantiene texto extraído ({len(text)} chars).")
            else:
                override_psm = pick_psm_for_page(page) if tess_autotune else None
                if override_psm is not None:
                    logging.debug(f"Pág {i}: autotune PSM -> {override_psm}")
                logging.debug(f"Pág {i}: aplicando OCR...")
                ocr_text = ocr_page(page, dpi=dpi, lang=ocr_lang,
                                    tess_config=tess_config,
                                    override_psm=override_psm)
                if ocr_text.strip():
                    text = ocr_text
                did_ocr = True

        # Guardar captura de página según política
        if save_pages == "all" or (save_pages == "ocr" and did_ocr):
            try:
                save_page_capture(page, pdf_img_root, i, dpi=dpi)
                logging.debug(f"Pág {i}: captura guardada.")
            except Exception as e:
                logging.debug(f"Pág {i}: no se pudo guardar captura: {e}")

        # Extraer imágenes incrustadas si se pide
        if extract_embedded:
            try:
                n = extract_embedded_images_from_page(doc, page, pdf_img_root, i)
                if n:
                    logging.debug(f"Pág {i}: {n} imagen(es) incrustada(s) extraída(s).")
            except Exception as e:
                logging.debug(f"Pág {i}: error extrayendo incrustadas: {e}")

        # Exportar HTML con imágenes posicionadas si se pide
        if place_images:
            try:
                export_page_layout_with_images(doc, page, pdf_img_root, i, dpi=dpi)
            except Exception as e:
                logging.debug(f"Pág {i}: error generando layout con imágenes: {e}")

        if text is None:
            text = ""
        all_text.append(text.strip())

    out_file = outdir / (pdf_path.stem + ".txt")
    try:
        with open(out_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n\n".join(all_text).strip() + "\n")
        logging.info(f"OK → {out_file}")
    except Exception as e:
        logging.error(f"No se pudo escribir {out_file}: {e}")
        out_file = pathlib.Path()

    try:
        doc.close()
    except Exception:
        pass

    return out_file


# ------------------------------ CLI ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extractor de texto de PDFs (con fallback OCR) y utilidades de imágenes."
    )
    parser.add_argument("input", type=str,
                        help="Ruta a un archivo .pdf o a una carpeta con PDFs.")
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="Carpeta de salida para .txt (por defecto ./output_txt).")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Forzar OCR en todas las páginas.")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Deshabilitar OCR (solo intenta extraer texto).")
    parser.add_argument("--ocr-lang", type=str, default="spa",
                        help="Idioma OCR Tesseract (ej. 'spa', 'eng', 'spa+eng').")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolución DPI para OCR y capturas (por defecto 300).")
    parser.add_argument("--min-chars", type=int, default=20,
                        help="Umbral mínimo de caracteres para considerar que una página tiene texto.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Modo verboso.")

    # Tesseract
    parser.add_argument("--tess-cmd", type=str, default=None,
                        help="Ruta explícita a tesseract.exe (sobrescribe PATH y TESSERACT_CMD).")
    parser.add_argument("--tess-config", type=str, default="--oem 1 --psm 6",
                        help="Parámetros extra para Tesseract (por defecto '--oem 1 --psm 6').")
    parser.add_argument("--tess-autotune", action="store_true",
                        help="Ajusta automáticamente --psm por página (4/6/11) según el layout.")

    # Imágenes / HTML
    parser.add_argument("--save-pages", choices=["none", "ocr", "all"], default="none",
                        help="Guardar capturas PNG de páginas: 'none' (por defecto), 'ocr' (solo si se aplicó OCR), 'all' (todas).")
    parser.add_argument("--img-dir", type=str, default=None,
                        help="Carpeta base donde guardar imágenes (por defecto ./output_images/<pdf_stem>/).")
    parser.add_argument("--extract-embedded", action="store_true",
                        help="Extraer imágenes incrustadas del PDF (no renders).")
    parser.add_argument("--place-images", action="store_true",
                        help="Genera HTML por página con las imágenes colocadas en su posición.")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Configurar Tesseract (si hay OCR disponible)
    configure_tesseract(args.tess_cmd)

    input_path = pathlib.Path(args.input)
    outdir = ensure_outdir(pathlib.Path(args.outdir) if args.outdir else None)

    pdfs = find_pdfs(input_path)
    if not pdfs:
        logging.error("No se encontraron PDFs en la ruta indicada.")
        sys.exit(2)

    images_outdir = pathlib.Path(args.img_dir) if args.img_dir else None

    for pdf in pdfs:
        extract_text_from_pdf(
            pdf_path=pdf,
            outdir=outdir,
            force_ocr=args.force_ocr,
            disable_ocr=args.no_ocr,
            ocr_lang=args.ocr_lang,
            dpi=args.dpi,
            min_chars_threshold=args.min_chars,
            tess_config=args.tess_config,
            tess_autotune=args.tess_autotune,
            save_pages=args.save_pages,
            images_outdir=images_outdir,
            extract_embedded=args.extract_embedded,
            place_images=args.place_images
        )


if __name__ == "__main__":
    main()
