"""
color_config.py
---------------
GUI tool to visually configure class colors for predict.py.

- Click any color swatch → opens system color picker
- Adjust alpha slider for mask classes
- Click Save → rewrites CLASS_CONFIG block in predict.py automatically

Run this before inference to set your preferred colors.
"""

import tkinter as tk
from tkinter import colorchooser, messagebox
import re
import os

# ─────────────────────────────────────────────
#  PATH TO predict.py  — edit if needed
# ─────────────────────────────────────────────
PREDICT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict.py")

# ─────────────────────────────────────────────
#  DEFAULT CLASS CONFIG  (mirrors predict.py)
# ─────────────────────────────────────────────
DEFAULT_CONFIG = [
    {"name": "SHOULDER_LANE",        "type": "mask", "color_rgb": (128,   0, 128), "alpha": 0.45},
    {"name": "CENTRE_LANE",          "type": "mask", "color_rgb": (255, 165,   0), "alpha": 0.45},
    {"name": "KERB",                 "type": "mask", "color_rgb": (144, 238, 144), "alpha": 0.45},
    {"name": "RIGID_CRASH_BARRIER",  "type": "mask", "color_rgb": (  0,  80,   0), "alpha": 0.45},
    {"name": "W-BEAM_CRASH_BARRIER", "type": "mask", "color_rgb": (  0, 255, 255), "alpha": 0.45},
    {"name": "CHEVRON_SIGNS",        "type": "box",  "color_rgb": (255, 200,   0), "alpha": 1.0 },
    {"name": "INFORMATIVE_SIGNS",    "type": "box",  "color_rgb": (  0,   0, 255), "alpha": 1.0 },
    {"name": "MANDATORY_SIGNS",      "type": "box",  "color_rgb": (255,   0,   0), "alpha": 1.0 },
    {"name": "WARNING_SIGNS",        "type": "box",  "color_rgb": (255, 128,   0), "alpha": 1.0 },
]


def rgb_to_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_bgr(r, g, b):
    return (b, g, r)


class ColorConfigApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Class Color Config — Road Safety Audit")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        self.entries = []  # list of dicts per class: color_rgb, alpha_var, swatch_btn

        self._build_ui()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg="#1e1e2e")
        header.pack(fill="x", padx=20, pady=(18, 4))

        tk.Label(header, text="Class Color Config", font=("Segoe UI", 15, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(side="left")
        tk.Label(header, text="Click a swatch to change color", font=("Segoe UI", 9),
                 bg="#1e1e2e", fg="#6c7086").pack(side="left", padx=12)

        # ── Column headers ──
        cols = tk.Frame(self.root, bg="#1e1e2e")
        cols.pack(fill="x", padx=20, pady=(6, 2))
        for text, w in [("Class", 22), ("Type", 7), ("Color", 10), ("Alpha / Opacity", 20)]:
            tk.Label(cols, text=text, font=("Segoe UI", 9, "bold"),
                     bg="#1e1e2e", fg="#6c7086", width=w, anchor="w").pack(side="left")

        # ── Separator ──
        tk.Frame(self.root, bg="#313244", height=1).pack(fill="x", padx=20, pady=2)

        # ── Class rows ──
        for cfg in DEFAULT_CONFIG:
            self._build_row(cfg)

        # ── Separator ──
        tk.Frame(self.root, bg="#313244", height=1).pack(fill="x", padx=20, pady=10)

        # ── Buttons ──
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=(0, 18), padx=20, fill="x")

        tk.Button(btn_frame, text="Reset to Defaults", font=("Segoe UI", 10),
                  bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                  relief="flat", padx=14, pady=6, cursor="hand2",
                  command=self._reset_defaults).pack(side="left")

        tk.Button(btn_frame, text="  Save to predict.py  ", font=("Segoe UI", 10, "bold"),
                  bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec",
                  relief="flat", padx=14, pady=6, cursor="hand2",
                  command=self._save).pack(side="right")

    def _build_row(self, cfg):
        row = tk.Frame(self.root, bg="#1e1e2e")
        row.pack(fill="x", padx=20, pady=3)

        entry = {
            "name":      cfg["name"],
            "type":      cfg["type"],
            "color_rgb": list(cfg["color_rgb"]),
            "alpha_var": tk.DoubleVar(value=cfg["alpha"]),
        }

        # Class name
        tk.Label(row, text=cfg["name"].replace("_", " "), font=("Segoe UI", 10),
                 bg="#1e1e2e", fg="#cdd6f4", width=22, anchor="w").pack(side="left")

        # Type badge
        badge_color = "#a6e3a1" if cfg["type"] == "mask" else "#f38ba8"
        badge_bg    = "#1e3a1e" if cfg["type"] == "mask" else "#3a1e1e"
        tk.Label(row, text=cfg["type"].upper(), font=("Segoe UI", 8, "bold"),
                 bg=badge_bg, fg=badge_color, width=6, pady=2, relief="flat").pack(side="left", padx=(0, 8))

        # Color swatch button
        init_hex = rgb_to_hex(*cfg["color_rgb"])
        swatch = tk.Button(row, bg=init_hex, width=5, height=1,
                           relief="solid", bd=1, cursor="hand2",
                           activebackground=init_hex)
        swatch.pack(side="left", padx=(0, 16))

        entry["swatch"] = swatch
        swatch.configure(command=lambda e=entry: self._pick_color(e))

        # Alpha slider (only meaningful for masks; disabled for box)
        alpha_frame = tk.Frame(row, bg="#1e1e2e")
        alpha_frame.pack(side="left", fill="x", expand=True)

        slider_state = "normal" if cfg["type"] == "mask" else "disabled"
        slider = tk.Scale(alpha_frame, from_=0.0, to=1.0, resolution=0.05,
                          orient="horizontal", variable=entry["alpha_var"],
                          bg="#1e1e2e", fg="#cdd6f4", troughcolor="#313244",
                          highlightthickness=0, sliderrelief="flat",
                          length=180, state=slider_state, showvalue=True,
                          font=("Segoe UI", 8))
        slider.pack(side="left")

        if cfg["type"] == "box":
            tk.Label(alpha_frame, text="(box — N/A)", font=("Segoe UI", 8),
                     bg="#1e1e2e", fg="#45475a").pack(side="left", padx=4)

        self.entries.append(entry)

    def _pick_color(self, entry):
        current_hex = rgb_to_hex(*entry["color_rgb"])
        result = colorchooser.askcolor(color=current_hex,
                                       title=f"Pick color for {entry['name']}")
        if result and result[0]:
            r, g, b = (int(x) for x in result[0])
            entry["color_rgb"] = [r, g, b]
            new_hex = rgb_to_hex(r, g, b)
            entry["swatch"].configure(bg=new_hex, activebackground=new_hex)

    def _reset_defaults(self):
        for entry, cfg in zip(self.entries, DEFAULT_CONFIG):
            entry["color_rgb"] = list(cfg["color_rgb"])
            entry["alpha_var"].set(cfg["alpha"])
            hex_col = rgb_to_hex(*cfg["color_rgb"])
            entry["swatch"].configure(bg=hex_col, activebackground=hex_col)

    def _build_config_block(self):
        lines = ["CLASS_CONFIG = {\n"]
        lines.append(f"    #  class name                     type      color (BGR)          alpha\n")
        for entry in self.entries:
            r, g, b = entry["color_rgb"]
            bgr     = rgb_to_bgr(r, g, b)
            alpha   = round(entry["alpha_var"].get(), 2)
            name    = entry["name"]
            typ     = entry["type"]
            padding = max(1, 25 - len(name))
            lines.append(
                f'    "{name}"{" " * padding}: '
                f'{{"type": "{typ}", '
                f'"color": ({bgr[0]:3d}, {bgr[1]:3d}, {bgr[2]:3d}), '
                f'"alpha": {alpha}}},\n'
            )
        lines.append("}\n")
        return "".join(lines)

    def _save(self):
        if not os.path.exists(PREDICT_SCRIPT):
            messagebox.showerror("Error", f"predict.py not found at:\n{PREDICT_SCRIPT}")
            return

        with open(PREDICT_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace CLASS_CONFIG block using regex
        pattern = r"CLASS_CONFIG\s*=\s*\{.*?\n\}"
        new_block = self._build_config_block().rstrip("\n")

        if not re.search(pattern, content, re.DOTALL):
            messagebox.showerror("Error", "Could not find CLASS_CONFIG block in predict.py.\n"
                                          "Make sure the block exists and ends with a lone '}'.")
            return

        new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)

        with open(PREDICT_SCRIPT, "w", encoding="utf-8") as f:
            f.write(new_content)

        messagebox.showinfo("Saved", f"CLASS_CONFIG updated in:\n{PREDICT_SCRIPT}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorConfigApp(root)
    root.mainloop()
