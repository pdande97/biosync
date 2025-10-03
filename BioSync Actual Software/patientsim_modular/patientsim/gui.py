import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PatientSimulatorGUI:
    def __init__(self, root, signal_source):
        self.root = root
        self.src = signal_source
        self.root.title("BioSync Patient Simulator")
        self.root.geometry("1260x740")

        # --- Menu ---
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
        self.root.columnconfigure(0, minsize=460)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        ttk.Label(self.root, text="Patient Simulator", font=("Sans-Serif", 28))            .grid(row=0, column=0, columnspan=2, sticky="nw", padx=20, pady=(12, 0))

        # Left controls
        left = ttk.LabelFrame(self.root, text="ECG Controls", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=20, pady=(60, 10))
        for r in range(24): left.rowconfigure(r, weight=0)
        left.columnconfigure(0, weight=1); left.columnconfigure(1, weight=1)

        # Rate & Fs
        ttk.Label(left, text="Rate (BPM)").grid(row=0, column=0, sticky="w")
        self.rate_entry = ttk.Entry(left); self.rate_entry.insert(0, "60")
        self.rate_entry.grid(row=1, column=0, sticky="ew", pady=(0,10))

        ttk.Label(left, text="Sampling Freq (Hz)").grid(row=0, column=1, sticky="w")
        self.fs_entry = ttk.Entry(left); self.fs_entry.insert(0, "250")
        self.fs_entry.grid(row=1, column=1, sticky="ew", pady=(0,10))

        # Amplitudes (mV) P/R/T
        ttk.Label(left, text="P-wave amp (mV)").grid(row=2, column=0, sticky="w")
        self.p_amp = ttk.Entry(left); self.p_amp.insert(0, "2.0")
        self.p_amp.grid(row=3, column=0, sticky="ew", pady=(0,10))

        ttk.Label(left, text="QRS-wave amp (mV)").grid(row=2, column=1, sticky="w")
        self.r_amp = ttk.Entry(left); self.r_amp.insert(0, "5.0")
        self.r_amp.grid(row=3, column=1, sticky="ew", pady=(0,10))

        ttk.Label(left, text="T-wave amp (mV)").grid(row=4, column=0, sticky="w")
        self.t_amp = ttk.Entry(left); self.t_amp.insert(0, "3.0")
        self.t_amp.grid(row=5, column=0, sticky="ew", pady=(0,10))

        # Delays (ms)
        ttk.Label(left, text="t1 delay (ms)").grid(row=4, column=1, sticky="w")
        self.t1_ms = ttk.Entry(left); self.t1_ms.insert(0, "80")
        self.t1_ms.grid(row=5, column=1, sticky="ew", pady=(0,10))

        ttk.Label(left, text="t2 delay (ms)").grid(row=6, column=0, sticky="w")
        self.t2_ms = ttk.Entry(left); self.t2_ms.insert(0, "120")
        self.t2_ms.grid(row=7, column=1, sticky="ew", pady=(0,10))

        # Durations
        ttk.Label(left, text="P-wave duration (ms)").grid(row=6, column=1, sticky="w")
        self.p_ms = ttk.Entry(left); self.p_ms.insert(0, "80")
        self.p_ms.grid(row=7, column=0, sticky="ew", pady=(0,10))

        ttk.Label(left, text="QRS-wave duration (ms)").grid(row=8, column=0, sticky="w")
        self.qrs_ms = ttk.Entry(left); self.qrs_ms.insert(0, "100")
        self.qrs_ms.grid(row=9, column=0, sticky="ew", pady=(0,10))

        ttk.Label(left, text="T-wave duration (ms)").grid(row=8, column=1, sticky="w")
        self.t_ms = ttk.Entry(left); self.t_ms.insert(0, "160")
        self.t_ms.grid(row=9, column=1, sticky="ew", pady=(0,10))

        # Buttons
        btns = ttk.Frame(left); btns.grid(row=12, column=0, columnspan=2, sticky="ew")
        btns.columnconfigure((0,1,2), weight=1)
        tk.Button(btns, text="Update Settings", command=self.update_settings).grid(row=0, column=0, sticky="ew", padx=(0,6))
        tk.Button(btns, text="Start", command=self.start, bg="green", fg="white").grid(row=0, column=1, sticky="ew", padx=6)
        tk.Button(btns, text="Stop", command=self.stop, bg="red", fg="white").grid(row=0, column=2, sticky="ew", padx=(6,0))

        # Right plot
        right = ttk.LabelFrame(self.root, text="Waveform Preview", padding=8)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=(60, 10))
        right.rowconfigure(0, weight=1); right.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Style
        self.ax.set_facecolor("black"); self.fig.patch.set_facecolor("black")
        self.ax.tick_params(colors='white')
        self.grid_on = True; self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Status
        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.root, textvariable=self.status, anchor="w", relief="sunken")            .grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 12))

        # Animation state
        self.running = False
        self.timer_id = None
        self.update_interval_ms = 4
        self.samples_per_tick = 2

        # Buffer ~2 seconds
        self.buffer_len = 500
        self.y = np.zeros(self.buffer_len)
        self.x = np.arange(self.buffer_len)
        (self.line,) = self.ax.plot(self.x, self.y, linewidth=2)

        # Fixed y-scale –5..+5 mV
        self.ax.set_ylim(-5.0, 5.0)
        self.ax.set_xlim(0, self.buffer_len-1)
        self._set_titles()
        self.canvas.draw()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------- Controls -------
    def update_settings(self):
        try:
            rate   = float(self.rate_entry.get())
            fs     = int(float(self.fs_entry.get()))
            amp_p  = float(self.p_amp.get())
            amp_r  = float(self.r_amp.get())
            amp_t  = float(self.t_amp.get())
            p_ms   = float(self.p_ms.get())
            qrs_ms = float(self.qrs_ms.get())
            t_ms   = float(self.t_ms.get())
            t1_ms  = float(self.t1_ms.get())
            t2_ms  = float(self.t2_ms.get())
            if rate <= 0 or fs <= 0: raise ValueError
        except Exception:
            messagebox.showerror("Invalid Settings", "Enter positive Rate/Fs and numeric amplitudes/durations.")
            return

        self.src.configure(
            fs=fs, rate=rate, amplitude=1.0,
            amp_p=amp_p, amp_r=amp_r, amp_t=amp_t,
            p_ms=p_ms, qrs_ms=qrs_ms, t_ms=t_ms,
            t1_ms=t1_ms, t2_ms=t2_ms
        )

        # resize buffer to ~2 s
        target = max(500, fs*2)
        if target != self.buffer_len:
            self.buffer_len = target
            self.y = np.zeros(self.buffer_len)
            self.x = np.arange(self.buffer_len)
            self.line.set_xdata(self.x)
            self.ax.set_xlim(0, self.buffer_len-1)

        self.ax.set_ylim(-5.0, 5.0)
        self._set_titles()
        self.status.set(f"Updated: Rate={rate} BPM, Fs={fs} Hz, Buffer={self.buffer_len}")
        self.canvas.draw_idle()

    def start(self):
        if self.running: return
        self.update_settings()
        self.running = True
        self.status.set("Running")
        self._tick()

    def stop(self):
        self.running = False
        if self.timer_id:
            try: self.root.after_cancel(self.timer_id)
            except Exception: pass
            self.timer_id = None
        self.status.set("Stopped")

    def toggle_grid(self):
        self.grid_on = not self.grid_on
        self.ax.grid(self.grid_on, color='gray', linestyle='--', linewidth=0.5)
        self.canvas.draw_idle()

    # ------- Animation -------
    def _tick(self):
        if not self.running: return
        self.y = np.append(self.y[self.samples_per_tick:], self.src.step(self.samples_per_tick))
        self.line.set_ydata(self.y)
        self._set_titles()
        self.canvas.draw_idle()
        self.timer_id = self.root.after(self.update_interval_ms, self._tick)

    def _set_titles(self):
        self.ax.set_title("Live ECG", color='white', fontsize=14)
        self.ax.set_ylabel("mV", color='white'); self.ax.set_xticks([]); self.ax.tick_params(colors='white')

    # ------- File actions -------
    def import_waveform(self):
        path = filedialog.askopenfilename(title="Import Custom Waveform",
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path: return
        self.status.set(f"Imported: {path} (CSVSignal not yet wired)")

    def save_settings(self):
        path = filedialog.asksaveasfilename(title="Save Settings", defaultextension=".json",
                                            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path: return
        cfg = dict(
            rate=self.rate_entry.get(), fs=self.fs_entry.get(),
            amp_p=self.p_amp.get(), amp_r=self.r_amp.get(), amp_t=self.t_amp.get(),
            p_ms=self.p_ms.get(), qrs_ms=self.qrs_ms.get(), t_ms=self.t_ms.get(),
            t1_ms=self.t1_ms.get(), t2_ms=self.t2_ms.get()
        )
        import json
        with open(path, "w") as f: json.dump(cfg, f, indent=2)
        self.status.set(f"Saved settings to: {path}")

    def on_close(self):
        self.stop()
        self.root.destroy()
