import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Pillow for image manipulation
import os

import customtkinter as ctk  # For modern UI widgets and dialogs
from tkinter import filedialog  # Kept for askopenfilename as CTk does not have its own
from tkinter import Toplevel  # For creating custom top-level windows if needed

# ─── CustomTkinter Theme Setup ────────────────────────────────────────────────
ctk.set_appearance_mode("System")  # "Light", "Dark", "System"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# ─── Color Scheme for OpenCV Window ──────────────────────────────────────────
 # Hex -> BGR for OpenCV
# Note: CustomTkinter will handle the colors for its own widgets
BANNER_COLOR = (152, 63, 127)  # #7f3f98 (BGR)
GRID_COLOR = (39, 170, 225)  # #27aae1 (BGR)
SELECTION_FILL = (39, 170, 225)  # semi-transparent
SELECTION_BORDER = (152, 63, 127)  # BANNER_COLOR

# ─── Font Setup ────────────────────────────────────────────────────────────────
try:
    # Ensure Gotham-Medium.otf is in the same directory as the script or system path
    FONT = ImageFont.truetype("Gotham-Medium.otf", 24)
except IOError:
    FONT = ImageFont.load_default()  # PIL default font fallback
    print("Warning: 'Gotham-Medium.otf' not found. Using default font.")

# ─── Auto Grid Detection ───────────────────────────────────────────────────────
def auto_detect_grid(img):
    """
    Attempts to auto-detect grid rows and columns based on image contours.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert colors for thresholding to make grid lines white on black
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Added OTSU for better adaptiveness

    # Morphological operations to enhance lines (optional, but can help)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]

    # Filter out very small contours and get bounding rectangles
    # Use a more robust area threshold, e.g., 0.1% of total image area
    min_contour_area = h * w * 0.001
    rects = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > min_contour_area]

    if not rects:
        return None, None

    # Attempt to find common heights and widths
    heights = [rh for (_, _, _, rh) in rects]
    widths = [rw for (_, _, rw, _) in rects]

    if not heights or not widths:
        return None, None

    try:
        # Using clustering or more robust statistics might be better for noisy data
        # For now, median can give a good estimate
        median_height = np.median(heights)
        median_width = np.median(widths)

        if median_height < 5 or median_width < 5:  # Minimum reasonable cell size
            return None, None

        # Estimate rows and columns based on image dimensions and median cell size
        rows = max(1, round(h / median_height))
        cols = max(1, round(w / median_width))

        # Basic sanity check: ensure rows/cols are within a reasonable range (e.g., 1 to 50)
        if not (1 <= rows <= 50 and 1 <= cols <= 50):
            return None, None

    except Exception as e:
        print(f"Error during auto-detection: {e}")
        return None, None
    return rows, cols

# ─── Main Application Class ────────────────────────────────────────────────────
class AutoCropperApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PDF Auto-Cropper 📐")
        self.geometry("400x300") # Smaller initial window for settings gathering
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.settings_frame = ctk.CTkFrame(self, corner_radius=10)
        self.settings_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.settings_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.settings_frame, text="Welcome to Auto-Cropper!",
                    font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, pady=(20, 10))

        ctk.CTkButton(self.settings_frame, text="Select PDF", command=self.gather_settings_gui,
                     height=40).grid(row=1, column=0, pady=20, padx=40, sticky="ew")

        self.pdf_path = None
        self.rows = None
        self.cols = None
        self.doc = None # Store FitZ document

    def gather_settings_gui(self):
        self.pdf_path = filedialog.askopenfilename(
            title="Select your imposed PDF",
            filetypes=[("PDF files","*.pdf")])
        
        if not self.pdf_path:
            ctk.CTkMessagebox(title="Cancelled", message="PDF selection cancelled.", icon="info")
            return

        try:
            self.doc = fitz.open(self.pdf_path)
            page = self.doc[0]
            pix = page.get_pixmap(dpi=150) # Use a higher DPI for better detection
            # Convert Pixmap to NumPy array
            img = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        except Exception as e:
            ctk.CTkMessagebox(title="Error", message=f"Failed to open PDF or render page: {e}", icon="cancel")
            return

        # Auto-detect grid
        auto_r, auto_c = auto_detect_grid(img)

        if auto_r and auto_c:
            response = ctk.CTkMessagebox(
                title="Confirm Grid",
                message=f"Auto-detected {auto_r} rows × {auto_c} cols.\nIs that correct?",
                icon="question", option_1="Yes", option_2="No")
            
            if response.get() == "Yes":
                self.rows, self.cols = auto_r, auto_c
            else:
                self.manual_grid_input()
        else:
            ctk.CTkMessagebox(title="Detection Failed",
                                message="Could not auto-detect grid. Please enter manually.", icon="warning")
            self.manual_grid_input()

        if self.rows is not None and self.cols is not None:
            self.withdraw() # Hide the main CustomTkinter window
            self.preview_and_crop_opencv() # Proceed to OpenCV preview

    def manual_grid_input(self):
        rows_dialog = ctk.CTkInputDialog(text="Enter number of rows:", title="Manual Input")
        rows_str = rows_dialog.get_input()
        try:
            self.rows = int(rows_str) if rows_str else None
            if self.rows is not None and self.rows < 1:
                raise ValueError
        except (ValueError, TypeError):
            ctk.CTkMessagebox(title="Invalid Input", message="Please enter a valid positive integer for rows.", icon="warning")
            self.rows = None
            return # Exit if rows is invalid

        cols_dialog = ctk.CTkInputDialog(text="Enter number of columns:", title="Manual Input")
        cols_str = cols_dialog.get_input()
        try:
            self.cols = int(cols_str) if cols_str else None
            if self.cols is not None and self.cols < 1:
                raise ValueError
        except (ValueError, TypeError):
            ctk.CTkMessagebox(title="Invalid Input", message="Please enter a valid positive integer for columns.", icon="warning")
            self.cols = None

    def preview_and_crop_opencv(self):
        doc = self.doc # Use the already opened document
        if not doc: return # Should not happen if logic is followed

        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        # Convert to BGR for OpenCV
        base = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

        h, w = base.shape[:2]
        pw, ph = page.rect.width, page.rect.height # Page dimensions in PDF points

        scale = 1.0  # initial scale for OpenCV preview
        min_scale = 0.2
        max_scale = 3.0
        selected = set() # Stores (row, col) tuples of selected cells
        banner_h = 60
        instr = "L-click: select | R-click: deselect | S: save & exit | Q: cancel | +/-: zoom"

        def draw_frame():
            # Scale the base image for display in OpenCV window
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            img_display = cv2.resize(base, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            cw, ch = scaled_w / self.cols, scaled_h / self.rows # Cell width/height on display

            # Draw grid lines
            for r_idx in range(1, self.rows):
                y = int(r_idx * ch)
                cv2.line(img_display, (0, y), (scaled_w, y), GRID_COLOR, 1)
            for c_idx in range(1, self.cols):
                x = int(c_idx * cw)
                cv2.line(img_display, (x, 0), (x, scaled_h), GRID_COLOR, 1)

            # Draw selected overlays
            for (r, c) in selected:
                x0, y0 = int(c * cw), int(r * ch)
                x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
                overlay = img_display.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), SELECTION_FILL, -1)
                cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display) # Blend with transparency
                cv2.rectangle(img_display, (x0, y0), (x1, y1), SELECTION_BORDER, 2) # Draw border

            # Draw banner at the top
            cv2.rectangle(img_display, (0, 0), (scaled_w, banner_h), BANNER_COLOR, -1)
            
            # Add instructions text to banner using PIL for better font rendering
            pil_img = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 15), instr, font=FONT, fill=(255, 255, 255)) # White text
            img_display = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            return img_display, cw, ch, scaled_w, scaled_h

        # Mouse callback function for OpenCV window
        def on_mouse(event, x, y, flags, param):
            non_local_scale = param['scale_ref'][0] # Access current scale
            non_local_rows = param['rows_ref'][0]
            non_local_cols = param['cols_ref'][0]

            if y < banner_h: # Ignore clicks on the banner area
                return

            # Calculate cell indices based on scaled coordinates
            current_w_display = int(w * non_local_scale)
            current_h_display = int(h * non_local_scale)
            
            cell_w_display = current_w_display / non_local_cols
            cell_h_display = current_h_display / non_local_rows

            r = int((y - banner_h) // cell_h_display) if (y - banner_h) >= 0 else int(y // cell_h_display)
            c = int(x // cell_w_display)
            
            if 0 <= r < non_local_rows and 0 <= c < non_local_cols:
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected.add((r, c))
                elif event == cv2.EVENT_RBUTTONDOWN:
                    selected.discard((r, c))

        cv2.namedWindow("Auto-Cropper Preview", cv2.WINDOW_NORMAL)
        # Pass mutable objects to the callback to allow updating scale/rows/cols
        mouse_param = {'scale_ref': [scale], 'rows_ref': [self.rows], 'cols_ref': [self.cols]}
        cv2.setMouseCallback("Auto-Cropper Preview", on_mouse, mouse_param)

        while True:
            img_display, cw, ch, scaled_w, scaled_h = draw_frame()
            
            # Update mouse_param with current frame's dimensions and scale
            mouse_param['scale_ref'][0] = scale
            
            cv2.imshow("Auto-Cropper Preview", img_display)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('s'): # Save and exit
                break
            elif key == ord('q'): # Cancel and exit
                selected.clear() # Clear selections if cancelled
                break
            elif key == ord('+') or key == ord('='):
                scale = min(max_scale, scale + 0.1)
            elif key == ord('-'):
                scale = max(min_scale, scale - 0.1)
        cv2.destroyAllWindows()
        self.deiconify() # Show the main CustomTkinter window again

        if not selected:
            ctk.CTkMessagebox(title="No Selection", message="No selections made. No files saved.", icon="info")
            return

        out_dir = os.path.dirname(self.pdf_path)
        stem = os.path.splitext(os.path.basename(self.pdf_path))[0]
        
        for idx, (r, c) in enumerate(selected, 1):
            # Calculate crop rectangle in PDF points
            x0_pdf = c * (pw / self.cols)
            y0_pdf = r * (ph / self.rows)
            x1_pdf = x0_pdf + (pw / self.cols)
            y1_pdf = y0_pdf + (ph / self.rows)
            rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)

            new_pdf = fitz.open()
            # Create a new page with the dimensions of the cropped area
            pg = new_pdf.new_page(width=rect.width, height=rect.height)
            
            # Place the content from the original page's clipped rectangle onto the new page
            pg.show_pdf_page(fitz.Rect(0, 0, rect.width, rect.height), # Destination rectangle on new page
                            doc, 0, # Source document and page number (0 for first page)
                            clip=rect) # Clip to the calculated rectangle on the source page
            
            name = f"{stem}_r{r}c{c}_cropped.pdf"
            save_path = os.path.join(out_dir, name)
            new_pdf.save(save_path)
            new_pdf.close() # Close the new PDF document
            print(f"Saved ({idx}/{len(selected)}): {name}")
        
        ctk.CTkMessagebox(title="Cropping Complete", message=f"Successfully saved {len(selected)} cropped PDFs to:\n{out_dir}", icon="check")
        doc.close() # Close the original PDF document

# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # For auto-detection placeholder, if you don't have a file.
    # import numpy as np # Already imported
    # import os # Already imported

    app = AutoCropperApp()
    app.mainloop()