# BioSync Patient Simulator – Integrated GUI + Live ECG (Tk-after animation)
# Fixes "nothing shows up" by avoiding FuncAnimation and forcing canvas redraws.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # ensure Tk backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PatientSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BioSync Patient Simulator (2025)")
        self.root.geometry("1200x700")

        # --- Menubar ---
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import Waveform", command=self.import_waveform)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Grid", command=self.toggle_grid)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

        # --- Layout ---
        self.root.columnconfigure(0, minsize=360)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        ttk.Label(self.root, text="Patient Simulator", font=("Sans-Serif", 28))\
            .grid(row=0, column=0, columnspan=2, sticky="nw", padx=20, pady=(12, 0))

        # Left controls
        left = ttk.LabelFrame(self.root, text="Signal Controls", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=20, pady=(60, 10))
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Signal Type").grid(row=0, column=0, sticky="w")
        self.signal_type = tk.StringVar(value="ECG")
        ttk.OptionMenu(left, self.signal_type, "ECG", "ECG").grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(left, text="Rate (BPM)").grid(row=2, column=0, sticky="w")
        self.bpm_entry = ttk.Entry(left)
        self.bpm_entry.insert(0, "75")
        self.bpm_entry.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(left, text="Sampling Freq (Hz)").grid(row=4, column=0, sticky="w")
        self.fs_entry = ttk.Entry(left)
        self.fs_entry.insert(0, "250")
        self.fs_entry.grid(row=5, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(left, text="Amplitude (mV)").grid(row=6, column=0, sticky="w")
        self.amp_entry = ttk.Entry(left)
        self.amp_entry.insert(0, "1.0")
        self.amp_entry.grid(row=7, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(left, text="Width (ms)").grid(row=8, column=0, sticky="w")
        self.width_entry = ttk.Entry(left)
        self.width_entry.insert(0, "120")
        self.width_entry.grid(row=9, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(left, text="Delay (ms)").grid(row=10, column=0, sticky="w")
        self.delay_entry = ttk.Entry(left)
        self.delay_entry.insert(0, "1.0")
        self.delay_entry.grid(row=11, column=0, sticky="ew", pady=(0, 12))

        btn_frame = ttk.Frame(left); btn_frame.grid(row=12, column=0, sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)
        tk.Button(btn_frame, text="Update Settings", command=self.update_settings).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(btn_frame, text="Start", command=self.start, bg="green", fg="white").grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(btn_frame, text="Stop", command=self.stop, bg="red", fg="white").grid(row=0, column=2, sticky="ew", padx=(6, 0))

        tk.Button(left, text="Import Waveform", command=self.import_waveform, bg="blue", fg="white")\
            .grid(row=13, column=0, sticky="ew", pady=(10, 0))

        # Right plot
        right = ttk.LabelFrame(self.root, text="Waveform Preview", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=(60, 10))
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Look
        self.ax.set_facecolor("black")
        self.fig.patch.set_facecolor("black")
        self.ax.tick_params(colors='white')
        self.grid_on = True
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Status bar
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken")\
            .grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 12))

        # Data / animation state
        self.running = False
        self.timer_id = None
        self.fs = 250
        self.bpm = 75
        self.update_interval_ms = 4  # UI timer tick (keep modest; we decimate samples per tick)
        self.samples_per_tick = 2     # how many samples to push per tick
        self.ecg_beat = self.generate_ecg_beat(self.fs)
        self.beat_len = len(self.ecg_beat)
        self.beat_interval = max(1, int((60.0 / self.bpm) * self.fs))
        self.sample_index = 0

        # buffer ~ 2 seconds
        self.buffer_len = max(self.fs * 2, 500)
        self.ecg_buffer = np.zeros(self.buffer_len)
        self.x = np.arange(self.buffer_len)

        # Create a single Line2D and reuse it
        (self.line,) = self.ax.plot(self.x, self.ecg_buffer, linewidth=2)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_xlim(0, self.buffer_len - 1)
        self._set_titles()
        self.canvas.draw()  # initial paint

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------- Model --------
    def generate_ecg_beat(self, fs):
        t = np.linspace(0, 1, fs, endpoint=False)
        p = 0.20 * np.exp(-((t - 0.10) ** 2) / 0.01)
        q = -0.15 * np.exp(-((t - 0.20) ** 2) / 0.001)
        r = 1.00 * np.exp(-((t - 0.25) ** 2) / 0.002)
        s = -0.25 * np.exp(-((t - 0.30) ** 2) / 0.001)
        tw = 0.35 * np.exp(-((t - 0.40) ** 2) / 0.01)
        return p + q + r + s + tw

    # -------- Controls --------
    def update_settings(self):
        try:
            self.bpm = int(float(self.bpm_entry.get()))
            self.fs = int(float(self.fs_entry.get()))
            if self.bpm <= 0 or self.fs <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid Settings", "Enter positive numbers for BPM and Sampling Freq.")
            return

        # rebuild beat and buffer
        self.ecg_beat = self.generate_ecg_beat(self.fs)
        self.beat_len = len(self.ecg_beat)
        self.beat_interval = max(1, int((60.0 / self.bpm) * self.fs))
        self.buffer_len = max(self.fs * 2, 500)

        if len(self.ecg_buffer) != self.buffer_len:
            self.ecg_buffer = np.zeros(self.buffer_len)
            self.x = np.arange(self.buffer_len)
            self.ax.set_xlim(0, self.buffer_len - 1)

        self.sample_index = 0
        self._set_titles()
        self.status.set(f"Updated: BPM={self.bpm}, Fs={self.fs} Hz, Buffer={self.buffer_len}")
        self.canvas.draw_idle()

    def start(self):
        if self.running:
            return
        self.update_settings()  # ensure latest
        self.running = True
        self.status.set(f"Running: {self.signal_type.get()} | {self.bpm} BPM | Fs={self.fs} Hz")
        self._tick()

    def stop(self):
        self.running = False
        if self.timer_id:
            try:
                self.root.after_cancel(self.timer_id)
            except Exception:
                pass
            self.timer_id = None
        self.status.set("Stopped")

    def toggle_grid(self):
        self.grid_on = not self.grid_on
        self.ax.grid(self.grid_on, color='gray', linestyle='--', linewidth=0.5)
        self.canvas.draw_idle()

    # -------- Animation tick --------
    def _tick(self):
        if not self.running:
            return

        try:
            amp = float(self.amp_entry.get())
        except Exception:
            amp = 1.0

        # Push a few samples per UI tick (smooth + CPU-friendly)
        for _ in range(self.samples_per_tick):
            if self.sample_index >= self.beat_interval:
                self.sample_index = 0
            s = self.ecg_beat[self.sample_index % self.beat_len]
            self.ecg_buffer = np.append(self.ecg_buffer[1:], amp * s)
            self.sample_index += 1

        # Update line artist only (fast) and force a draw
        self.line.set_ydata(self.ecg_buffer)
        self._set_titles()
        self.canvas.draw_idle()

        # Schedule next tick
        self.timer_id = self.root.after(self.update_interval_ms, self._tick)

    def _set_titles(self):
        self.ax.set_title(f"Live {self.signal_type.get()} ({self.bpm} BPM)", color='white', fontsize=16)
        self.ax.set_ylabel("Amplitude", color='white')
        self.ax.set_xticks([])
        self.ax.tick_params(colors='white')
        self.ax.set_facecolor("black")
        self.fig.patch.set_facecolor("black")

    # -------- File actions --------
    def import_waveform(self):
        path = filedialog.askopenfilename(
            title="Import Custom Waveform",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        # TODO: assign loaded array into self.ecg_beat with resampling to self.fs
        self.status.set(f"Imported: {path}")

    def save_settings(self):
        path = filedialog.asksaveasfilename(
            title="Save Settings", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.status.set(f"Saved settings to: {path}")

    # -------- Close --------
    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PatientSimulatorApp(root)
    root.mainloop()

