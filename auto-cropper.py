import fitz              # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os

# ─── Color Scheme ──────────────────────────────────────────────────────────────
# Hex -> BGR for OpenCV
BANNER_COLOR   = (0x98, 0x3f, 0x7f)  # #7f3f98 → (152,63,127)
GRID_COLOR     = (0xe1, 0xaa, 0x27)  # #27aae1 → (225,170,39)
SELECTION_FILL = (0xe1, 0xaa, 0x27)  # semi-transparent
SELECTION_BORDER = BANNER_COLOR

# ─── Font Setup ────────────────────────────────────────────────────────────────
try:
    FONT = ImageFont.truetype("Gotham-Medium.otf", 24)
except IOError:
    FONT = ImageFont.load_default()  # PIL default font fallback

def auto_detect_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > h*w*0.005]
    if not rects:
        return None, None
    heights = [rh for (_,_,_,rh) in rects]
    widths = [rw for (_,_,rw,_) in rects]
    try:
        rows = max(1, round(h / np.median(heights)))
        cols = max(1, round(w / np.median(widths)))
    except Exception:
        return None, None
    return rows, cols

def gather_settings():
    root = tk.Tk()
    root.withdraw()

    pdf_path = filedialog.askopenfilename(
        title="Select your imposed PDF",
        filetypes=[("PDF files","*.pdf")])
    if not pdf_path:
        return None

    # Render page for detection
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    img = np.array(Image.frombytes("RGB",[pix.width,pix.height],pix.samples))

    # Auto‑detect
    auto_r, auto_c = auto_detect_grid(img)
    if auto_r and auto_c:
        ok = messagebox.askyesno(
            "Confirm Grid",
            f"Auto‑detected {auto_r} rows × {auto_c} cols.\nIs that correct?")
        if ok:
            rows, cols = auto_r, auto_c
        else:
            rows = simpledialog.askinteger("Rows","Enter number of rows:",minvalue=1)
            cols = simpledialog.askinteger("Columns","Enter number of columns:",minvalue=1)
    else:
        messagebox.showwarning("Detection Failed",
                               "Could not auto‑detect grid. Please enter manually.")
        rows = simpledialog.askinteger("Rows","Enter number of rows:",minvalue=1)
        cols = simpledialog.askinteger("Columns","Enter number of columns:",minvalue=1)

    root.destroy()
    if rows is None or cols is None:
        return None
    return pdf_path, rows, cols

def preview_and_crop(pdf_path, rows, cols):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    base = np.array(Image.frombytes("RGB",[pix.width,pix.height],pix.samples))
    h, w = base.shape[:2]
    pw, ph = page.rect.width, page.rect.height
    cw, ch = w/cols, h/rows

    selected = set()
    banner_h = 60
    instr = "L-click: select | R-click: deselect | S: save & exit | Q: cancel"

    def draw_frame():
        img = base.copy()
        # grid lines
        for r in range(1, rows):
            y = int(r*ch)
            cv2.line(img, (0,y), (w,y), GRID_COLOR, 1)
        for c in range(1, cols):
            x = int(c*cw)
            cv2.line(img, (x,0), (x,h), GRID_COLOR, 1)
        # selected overlays
        for (r,c) in selected:
            x0, y0 = int(c*cw), int(r*ch)
            x1, y1 = int((c+1)*cw), int((r+1)*ch)
            overlay = img.copy()
            cv2.rectangle(overlay, (x0,y0), (x1,y1), SELECTION_FILL, -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.rectangle(img, (x0,y0), (x1,y1), SELECTION_BORDER, 2)
        # banner
        cv2.rectangle(img, (0,0), (w,banner_h), BANNER_COLOR, -1)
        # instructions text
        if FONT:
            # PIL draw for Gotham Medium
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            # white text for contrast
            draw.text((10, 7), instr, font=FONT, fill=(255,255,255))
            img = np.array(pil)
        else:
            # fallback Hershey
            cv2.putText(img, instr, (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return img

    def on_mouse(evt, x, y, flags, param):
        if y < banner_h: 
            return  # ignore clicks on banner
        r = int(y//ch); c = int(x//cw)
        if 0 <= r < rows and 0 <= c < cols:
            if evt == cv2.EVENT_LBUTTONDOWN:
                selected.add((r,c))
            elif evt == cv2.EVENT_RBUTTONDOWN:
                selected.discard((r,c))

    cv2.namedWindow("Auto-Cropper", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Auto-Cropper", on_mouse)

    while True:
        cv2.imshow("Auto-Cropper", draw_frame())
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s') or key == ord('q'):
            break
    cv2.destroyAllWindows()

    if not selected:
        messagebox.showinfo("No Selection", "No selections — nothing saved.")
        return

    out_dir = os.path.dirname(pdf_path)
    stem   = os.path.splitext(os.path.basename(pdf_path))[0]
    for idx, (r,c) in enumerate(selected, 1):
        x0 = c*(pw/cols); y0 = r*(ph/rows)
        rect = fitz.Rect(x0,y0,x0+(pw/cols),y0+(ph/rows))
        new = fitz.open()
        pg  = new.new_page(width=rect.width, height=rect.height)
        pg.show_pdf_page(fitz.Rect(0,0,rect.width,rect.height),
                        doc, 0, clip=rect)
        name = f"{stem}_r{r}c{c}_cropped.pdf"
        new.save(os.path.join(out_dir, name))
        print(f"Saved ({idx}/{len(selected)}): {name}")

def main():
    settings = gather_settings()
    if not settings:
        print("Cancelled.")
        return
    pdf_path, rows, cols = settings
    preview_and_crop(pdf_path, rows, cols)

if __name__ == "__main__":
    main()
