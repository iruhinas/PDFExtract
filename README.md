# PDFExtract
# USAGE: PDFExtract (ejemplos prácticos)

Guía rápida con **comandos listos para pegar** en PowerShell para `pdf_extract.py`.

> Asumo que ya activaste el venv:
> ```powershell
> E:\Phyton\PDFExtract\venv\Scripts\Activate.ps1
> ```

---

## 1) OCR y extracción básica

### Un único PDF
```powershell
python pdf_extract.py "E:\docs\archivo.pdf" -v
```

### Carpeta completa
```powershell
python pdf_extract.py "E:\docs\lote" -v
```

### Cambiar carpeta de salida de textos
```powershell
python pdf_extract.py "E:\docs\lote" -o "E:\salidas_txt" -v
```

---

## 2) Control fino del OCR (Tesseract)

### Fijar PSM recomendado para layouts complejos (tu caso)
```powershell
python pdf_extract.py "E:\docs\escaneados" --tess-config "--oem 1 --psm 4" -v
```

### Layout uniforme (párrafos/columna única)
```powershell
python pdf_extract.py "E:\docs\mono_columna" --tess-config "--oem 1 --psm 6" -v
```

### Autotune por página (elige 4/6/11 según el layout detectado)
```powershell
python pdf_extract.py "E:\docs\mixto" --tess-autotune -v
```

### Idiomas combinados (español + inglés)
```powershell
python pdf_extract.py "E:\docs\tech" --ocr-lang "spa+eng" --tess-autotune -v
```

### Aumentar DPI para mejorar OCR (más lento, mejor en escaneos finos)
```powershell
python pdf_extract.py "E:\docs\escaneados" --dpi 400 --tess-autotune -v
```

### Forzar OCR en todas las páginas (ignora texto embebido)
```powershell
python pdf_extract.py "E:\docs\solo_imagenes" --force-ocr --tess-config "--oem 1 --psm 4" -v
```

### Deshabilitar OCR (cuando sabes que hay texto embebido)
```powershell
python pdf_extract.py "E:\docs\buscables" --no-ocr -v
```

### Elegir binario de Tesseract explícito
```powershell
python pdf_extract.py "E:\docs\lote" --tess-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" -v
```

---

## 3) Imágenes y layouts

### Guardar capturas PNG de todas las páginas (400 DPI)
```powershell
python pdf_extract.py "E:\docs\lote" --save-pages all --dpi 400 -v
```

### Solo guardar capturas cuando se aplicó OCR
```powershell
python pdf_extract.py "E:\docs\escaneados" --save-pages ocr --dpi 300 -v
```

### Extraer imágenes incrustadas
```powershell
python pdf_extract.py "E:\docs\revista" --extract-embedded -v
```

### Generar HTML por página colocando cada imagen en su sitio
```powershell
python pdf_extract.py "E:\docs\manual" --place-images --dpi 300 -v
```

### Elegir carpeta base para imágenes/HTML
```powershell
python pdf_extract.py "E:\docs\lote" --img-dir "E:\salidas_layout" --place-images --extract-embedded --dpi 300 -v
```

**Estructura de salida por PDF (cuando usas `--place-images`):**
```
output_images\
  <NombrePDF>\
    page_0001\
      page_0001.png
      page_0001.html
      page_0001_manifest.json
      crops\
        img_001.png
        img_002.png
    page_0002\
      ...
```

Abre `page_0001\page_0001.html` en tu navegador para ver la página con **las imágenes recortadas y posicionadas**.

---

## 4) Rendimiento y calidad

- **DPI**: 300 es buen equilibrio; 400 mejora OCR/recortes a costa de tiempo.
- **PSM**: 
  - `4` multi-columna / layout libre, 
  - `6` bloque uniforme, 
  - `11` texto disperso.
- **Autotune**: útil en lotes heterogéneos, evita forzar un PSM único.

---

## 5) Variables útiles (opcional)

Agregar a `Activate.ps1` del venv para fijar Tesseract:
```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:PATH = "C:\Program Files\Tesseract-OCR;$env:PATH"
```

---

## 6) Solución de problemas
- **TesseractNotFoundError** → revisa `--tess-cmd` o variables de entorno.
- **Resultados pobres** → sube `--dpi`, prueba `--psm 4/6`, usa `--ocr-lang "spa+eng"`.
- **OCR innecesario** → baja `--min-chars` o usa `--no-ocr` si el PDF es buscable.
- **Imágenes no aparecen en HTML** → asegúrate de abrir el `page_XXXX.html` junto a su carpeta `crops` y `page_XXXX.png`.

---

## 7) Recordatorio de ayuda
```powershell
python pdf_extract.py -h
```
