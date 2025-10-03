import tkinter as tk
from gui import PatientSimulatorGUI
from signals.ecg import ECG

def main():
    root = tk.Tk()
    gui = PatientSimulatorGUI(root, ECG())
    root.mainloop()

if __name__ == "__main__":
    main()
