import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# ─── CustomTkinter Theme Setup ────────────────────────────────────────────────
ctk.set_appearance_mode("System")  # "Light", "Dark", "System"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# ─── Color Scheme for OpenCV Window ──────────────────────────────────────────
BANNER_COLOR    = (152, 63, 127)   # #7f3f98 (BGR)
RECT_BORDER     = (152, 63, 127)   # same as banner for border
RECT_FILL       = (39, 170, 225)   # semi-transparent fill
TEXT_COLOR      = (255, 255, 255)
BANNER_HEIGHT   = 40               # pixels

# ─── Font Setup ────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(__file__)
font_path = os.path.join(BASE_PATH, "Gotham-Medium.otf")
try:
    FONT = ImageFont.truetype(font_path, 18)
except IOError:
    FONT = None

class AutoCropperApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PDF Auto-Cropper 📐")
        self.geometry("400x200")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        frame = ctk.CTkFrame(self, corner_radius=10)
        frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Welcome to Auto-Cropper!",
                     font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, pady=(10,10))
        ctk.CTkButton(frame, text="Select PDF & Crop",
                      command=self.select_and_crop, height=40).grid(
            row=1, column=0, padx=40, pady=(0,10), sticky="ew")

    def select_and_crop(self):
        pdf_path = filedialog.askopenfilename(
            title="Select imposed PDF",
            filetypes=[("PDF files","*.pdf")]
        )
        if not pdf_path:
            return

        try:
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open or render PDF: {e}")
            return

        self.withdraw()
        # Now returns selections and display size
        selections, disp_w, disp_h = self.run_crop_ui(img)
        self.deiconify()

        if selections:
            out_dir = os.path.dirname(pdf_path)
            stem = os.path.splitext(os.path.basename(pdf_path))[0]
            pw, ph = page.rect.width, page.rect.height
            for idx, (y0, x0, y1, x1) in enumerate(selections, 1):
                # Map display coords back to PDF points
                x0_pdf = x0 / disp_w * pw
                y0_pdf = y0 / disp_h * ph
                x1_pdf = x1 / disp_w * pw
                y1_pdf = y1 / disp_h * ph
                rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)

                new_doc = fitz.open()
                new_page = new_doc.new_page(width=rect.width, height=rect.height)
                new_page.show_pdf_page(
                    fitz.Rect(0,0,rect.width,rect.height), doc, 0, clip=rect
                )
                out_name = f"{stem}_crop{idx}.pdf"
                new_doc.save(os.path.join(out_dir, out_name))

            messagebox.showinfo("Done",
                                f"Saved {len(selections)} crops to:\n{out_dir}")
        else:
            messagebox.showinfo("Cancelled", "No crops were made.")
        doc.close()

    def run_crop_ui(self, img):
        # Determine display size (90% of screen)
        root = tk.Tk(); root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        max_w = int(sw * 0.9)
        max_h = int(sh * 0.9) - BANNER_HEIGHT
        orig_h, orig_w = img.shape[:2]
        scale = min(max_w / orig_w, max_h / orig_h, 1.0)
        disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)

        # Resize PDF image
        base = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        selections = []
        current = None
        drawing = False
        mx = my = 0

        def draw_all():
            # Canvas: PDF region + banner
            disp = np.zeros((disp_h + BANNER_HEIGHT, disp_w, 3), dtype=np.uint8)
            disp[BANNER_HEIGHT:, :] = base
            # PDF border and banner
            cv2.rectangle(disp, (0, BANNER_HEIGHT), (disp_w-1, disp_h+BANNER_HEIGHT-1), RECT_BORDER, 2)
            cv2.rectangle(disp, (0,0), (disp_w, BANNER_HEIGHT), BANNER_COLOR, -1)
            instr = "Drag to draw | Right-click undo | S save | Q cancel"
            if FONT:
                pil = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                draw.text((10, 10), instr, font=FONT, fill=TEXT_COLOR)
                disp = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(disp, instr, (10, BANNER_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            # Draw selections
            for y0, x0, y1, x1 in selections:
                overlay = disp.copy()
                y0b, y1b = y0 + BANNER_HEIGHT, y1 + BANNER_HEIGHT
                cv2.rectangle(overlay, (x0, y0b), (x1, y1b), RECT_FILL, -1)
                cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
                cv2.rectangle(disp, (x0, y0b), (x1, y1b), RECT_BORDER, 2)
            # Draw current rectangle
            if drawing and current:
                y0, x0 = current
                cv2.rectangle(disp, (x0, y0+BANNER_HEIGHT), (mx, my+BANNER_HEIGHT), RECT_BORDER, 1)
            return disp

        def mouse_cb(event, x, y, flags, param):
            nonlocal drawing, current, mx, my
            y_ = y - BANNER_HEIGHT
            x_ = x
            if y_ < 0 or x_ < 0 or x_ >= disp_w or y_ >= disp_h:
                return
            mx, my = x_, y_
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                current = (y_, x_)
            elif event == cv2.EVENT_LBUTTONUP and drawing:
                drawing = False
                y0, x0 = current
                selections.append((min(y0, my), min(x0, mx), max(y0, my), max(x0, mx)))
                current = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                if selections:
                    selections.pop()

        win = "Crop Selector"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, disp_h + BANNER_HEIGHT)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            frame = draw_all()
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('s'), ord('q')):
                break
        cv2.destroyAllWindows()
        return selections, disp_w, disp_h

if __name__ == "__main__":
    app = AutoCropperApp()
    app.mainloop()
