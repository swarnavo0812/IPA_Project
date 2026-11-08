import os
import csv
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from image_processing import apply_technique
from image_metrics import calculate_metrics


class ImageEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Enhancement and Restoration")
        self.root.geometry("1180x760")
        self.root.minsize(950, 650)

        self.original = None
        self.processed = None
        self.reference = None
        self.input_path = None
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.project_dir, "output_images")
        os.makedirs(self.output_dir, exist_ok=True)

        self.create_interface()

    def create_interface(self):
        ttk.Label(self.root, text="Image Enhancement and Restoration",
                  font=("Arial", 20, "bold")).pack(pady=12)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Load Reference", command=self.load_reference).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Reset", command=self.reset_image).pack(side="left", padx=5)

        area = ttk.Frame(self.root, padding=10)
        area.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(area, text="Processing Options", padding=12)
        controls.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(controls, text="Select technique:").pack(anchor="w")
        self.technique = tk.StringVar(value="CLAHE")
        options = ["Histogram Equalization", "CLAHE", "Gamma Correction",
                   "Median Filter", "Gaussian Filter", "Unsharp Mask",
                   "DFT Low-Pass Filter", "DFT High-Pass Filter"]
        ttk.Combobox(controls, textvariable=self.technique, values=options,
                     state="readonly", width=25).pack(fill="x", pady=(4, 12))

        ttk.Label(controls, text="Gamma value:").pack(anchor="w")
        self.gamma = tk.DoubleVar(value=1.4)
        ttk.Scale(controls, from_=0.2, to=3.0, variable=self.gamma,
                  orient="horizontal").pack(fill="x")
        self.gamma_label = ttk.Label(controls, text="1.40")
        self.gamma_label.pack(anchor="e")
        self.gamma.trace_add("write", self.update_gamma_label)

        ttk.Label(controls, text="Kernel size:").pack(anchor="w", pady=(12, 3))
        self.kernel = tk.IntVar(value=5)
        ttk.Combobox(controls, textvariable=self.kernel, values=[3, 5, 7, 9, 11],
                     state="readonly", width=10).pack(anchor="w")

        ttk.Button(controls, text="Apply Technique", command=self.process_image).pack(fill="x", pady=(18, 6))
        ttk.Button(controls, text="Show Histogram", command=self.show_histogram).pack(fill="x", pady=5)
        ttk.Button(controls, text="Calculate Metrics", command=self.show_metrics).pack(fill="x", pady=5)

        instructions = ("Steps:\n1. Load an image.\n2. Select a technique.\n"
                        "3. Apply the technique.\n4. Save the result.\n\n"
                        "For MSE, PSNR and SSIM, load a clean reference image "
                        "of the same scene and dimensions.")
        ttk.Label(controls, text=instructions, justify="left", wraplength=220).pack(anchor="w", pady=18)

        images = ttk.Frame(area)
        images.pack(side="left", fill="both", expand=True)
        original_box = ttk.LabelFrame(images, text="Original Image", padding=8)
        original_box.pack(side="left", fill="both", expand=True, padx=5)
        result_box = ttk.LabelFrame(images, text="Processed Image", padding=8)
        result_box.pack(side="left", fill="both", expand=True, padx=5)

        self.original_view = ttk.Label(original_box, text="No image loaded")
        self.original_view.pack(fill="both", expand=True)
        self.result_view = ttk.Label(result_box, text="No result yet")
        self.result_view.pack(fill="both", expand=True)

        self.status = tk.StringVar(value="Ready. Load an image to begin.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w", padding=6).pack(fill="x", side="bottom")

    def update_gamma_label(self, *_):
        self.gamma_label.config(text=f"{self.gamma.get():.2f}")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not open this image.")
            return
        self.original = image
        self.processed = image.copy()
        self.input_path = path
        self.display_image(image, self.original_view)
        self.display_image(self.processed, self.result_view)
        self.status.set(f"Loaded: {os.path.basename(path)}")

    def load_reference(self):
        path = filedialog.askopenfilename(
            title="Choose a clean reference image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        reference = cv2.imread(path)
        if reference is None:
            messagebox.showerror("Error", "Could not open the reference image.")
            return
        self.reference = reference
        self.status.set(f"Reference loaded: {os.path.basename(path)}")

    def display_image(self, image, label):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(pil_image)
        label.configure(image=photo, text="")
        label.image = photo

    def process_image(self):
        if self.original is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        try:
            self.processed = apply_technique(
                self.original, self.technique.get(), self.gamma.get(), self.kernel.get())
            self.display_image(self.processed, self.result_view)
            self.status.set(f"Applied: {self.technique.get()}")
            self.write_experiment_log()
        except Exception as error:
            messagebox.showerror("Processing error", str(error))

    def save_result(self):
        if self.processed is None:
            messagebox.showwarning("No result", "Load and process an image first.")
            return
        path = filedialog.asksaveasfilename(
            initialdir=self.output_dir, initialfile="enhanced_image.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if path:
            if cv2.imwrite(path, self.processed):
                self.status.set(f"Saved: {path}")
                messagebox.showinfo("Saved", "Image saved successfully.")
            else:
                messagebox.showerror("Save error", "Could not save the image.")

    def reset_image(self):
        if self.original is not None:
            self.processed = self.original.copy()
            self.display_image(self.processed, self.result_view)
            self.status.set("Image reset to original.")

    def show_histogram(self):
        if self.original is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        window = tk.Toplevel(self.root)
        window.title("Grayscale Histogram")
        canvas = tk.Canvas(window, width=570, height=330, bg="white")
        canvas.pack(padx=10, pady=10)
        highest = max(float(hist.max()), 1.0)
        points = []
        for i, value in enumerate(hist):
            x = 35 + i * (500 / 255)
            y = 275 - (float(value) / highest) * 230
            points.extend((x, y))
        canvas.create_line(35, 275, 535, 275, fill="black")
        canvas.create_line(35, 35, 35, 275, fill="black")
        canvas.create_line(*points, fill="blue", width=1)
        canvas.create_text(285, 300, text="Pixel intensity (0-255)")
        canvas.create_text(12, 150, text="Frequency", angle=90)

    def show_metrics(self):
        if self.processed is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        try:
            values = calculate_metrics(self.processed, self.reference)
            message = [f"Image size: {values['width']} x {values['height']}",
                       f"Mean intensity: {values['mean']:.2f}",
                       f"Standard deviation: {values['std_dev']:.2f}"]
            if "mse" in values:
                psnr_text = "Infinity" if values["psnr"] == float("inf") else f"{values['psnr']:.4f}"
                message += ["", "Reference-based metrics:",
                            f"MSE: {values['mse']:.4f}",
                            f"PSNR: {psnr_text} dB",
                            f"SSIM: {values['ssim']:.4f}"]
            else:
                message += ["", "No clean reference image loaded.",
                            "MSE, PSNR and SSIM were not calculated."]
            messagebox.showinfo("Image Metrics", "\n".join(message))
        except ValueError as error:
            messagebox.showwarning("Metric warning", str(error))

    def write_experiment_log(self):
        log_file = os.path.join(self.project_dir, "experiment_log.csv")
        exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["Date and Time", "Input Image", "Technique", "Gamma", "Kernel"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             os.path.basename(self.input_path), self.technique.get(),
                             f"{self.gamma.get():.2f}", self.kernel.get()])


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancementApp(root)
    root.mainloop()
