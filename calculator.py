import tkinter as tk
from tkinter import ttk

class Calculator:
    def __init__(self, root):
        self.root = root
        root.title("计算器")
        root.geometry("320x480")
        root.resizable(False, False)

        self.expression = ""
        self.result_var = tk.StringVar(value="0")

        self._create_style()
        self._create_display()
        self._create_buttons()

    def _create_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 14), padding=8)

    def _create_display(self):
        display_frame = ttk.Frame(self.root)
        display_frame.pack(fill="x", padx=10, pady=(20, 10))

        entry = ttk.Entry(
            display_frame,
            textvariable=self.result_var,
            font=("Segoe UI", 28),
            justify="right",
            state="readonly",
        )
        entry.pack(fill="x")

    def _create_buttons(self):
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        buttons = [
            ("C", 0, 0), ("±", 0, 1), ("%", 0, 2), ("÷", 0, 3),
            ("7", 1, 0), ("8", 1, 1), ("9", 1, 2), ("×", 1, 3),
            ("4", 2, 0), ("5", 2, 1), ("6", 2, 2), ("-", 2, 3),
            ("1", 3, 0), ("2", 3, 1), ("3", 3, 2), ("+", 3, 3),
            ("0", 4, 0), (".", 4, 2), ("=", 4, 3),
        ]

        for text, row, col in buttons:
            if text == "0":
                btn = ttk.Button(btn_frame, text=text, command=lambda t=text: self._press(t))
                btn.grid(row=row, column=col, columnspan=2, sticky="nsew", padx=2, pady=2)
            else:
                btn = ttk.Button(btn_frame, text=text, command=lambda t=text: self._press(t))
                btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

        for i in range(5):
            btn_frame.rowconfigure(i, weight=1)
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)

    def _press(self, key):
        ops = {"+", "-", "×", "÷", "%"}

        if key == "C":
            self.expression = ""
            self.result_var.set("0")
            return

        if key == "±":
            if self.expression and self.expression[0] == "-":
                self.expression = self.expression[1:]
            else:
                self.expression = "-" + self.expression
            self.result_var.set(self.expression or "0")
            return

        if key == "=":
            try:
                expr = self.expression.replace("×", "*").replace("÷", "/")
                result = eval(expr)
                self.result_var.set(str(result))
                self.expression = str(result)
            except Exception:
                self.result_var.set("错误")
                self.expression = ""
            return

        if key in ops:
            if self.expression and self.expression[-1] in ops:
                self.expression = self.expression[:-1]
        elif key == ".":
            last_num = self._get_last_number()
            if "." in last_num:
                return

        self.expression += key
        self.result_var.set(self.expression)

    def _get_last_number(self):
        for op in ["+", "-", "×", "÷", "%"]:
            if op in self.expression:
                parts = self.expression.split(op)
                return parts[-1]
        return self.expression


if __name__ == "__main__":
    root = tk.Tk()
    Calculator(root)
    root.mainloop()
