# PatientSim (Tkinter + Matplotlib)

Modular GUI that visualizes synthetic ECG (and pluggable signals) in real time.
Uses Tk's `.after()` loop (no FuncAnimation) for reliable embedded updates.

## Run
```bash
python app.py
```

## Layout
```
patientsim/
├─ app.py                # entry point
├─ gui.py                # Tkinter UI (no signal math)
├─ signals/
│  ├─ __init__.py
│  ├─ base.py            # SignalSource interface
│  ├─ ecg.py             # ECG implementation
│  └─ respiration.py     # Respiration signal (optional demo)
├─ io/
│  └─ dac.py             # placeholder for future hardware I/O
└─ utils/
   └─ resample.py        # (placeholder) helpers
```
