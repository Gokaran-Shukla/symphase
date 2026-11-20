# -*- coding: utf-8 -*-
"""
symphase.quiz — self-checking quizzes for the SymPhase tutorials

    from symphase.quiz import Quiz
    q = Quiz("symmetry")          # or "hull"
    q.show(1)                     # print question 1
    q.answer(1, "b")              # check an answer (letter, number or text)
    q.hint(1)
    q.score()                     # running score
    q.widget(1)                   # clickable version (needs ipywidgets)

Answers are stored lightly encoded so that they are not visible by accident.
"""

import base64
import json
import math
import os

_GS_NOTICE = "© 2026 Gokaran Shukla · github.com/Gokaran-Shukla/symphase · MIT License"  # gs-watermark


_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _enc(x):
    return base64.b64encode(json.dumps(x).encode()[::-1]).decode()


def _dec(s):
    return json.loads(base64.b64decode(s.encode())[::-1].decode())


def list_quizzes():
    for f in sorted(os.listdir(_DATA)):
        if f.startswith("quiz_") and f.endswith(".json"):
            with open(os.path.join(_DATA, f)) as fh:
                d = json.load(fh)
            print(f"  {f[5:-5]:<10s} {d['title']}  ({len(d['questions'])} questions)")


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except Exception:  # noqa: BLE001
        return False


class Quiz:
    def __init__(self, name):
        path = name if os.path.isfile(name) else os.path.join(_DATA, f"quiz_{name}.json")
        with open(path) as fh:
            d = json.load(fh)
        self.title = d["title"]
        self.questions = {int(q["id"]): q for q in d["questions"]}
        self.results = {}

    def __len__(self):
        return len(self.questions)

    # ------------------------------------------------------------------
    def _text(self, i):
        q = self.questions[i]
        s = f"Q{i}. {q['question']}"
        if q.get("options"):
            s += "\n" + "\n".join(f"   ({k}) {v}" for k, v in q["options"].items())
        if q["type"] == "num":
            s += f"\n   [numerical answer{', unit: ' + q['unit'] if q.get('unit') else ''}]"
        return s

    def show(self, i=None):
        print(_GS_NOTICE + "\n")
        for k in ([i] if i else sorted(self.questions)):
            print(self._text(k) + "\n")

    def hint(self, i):
        print(f"Hint Q{i}: {self.questions[i].get('hint', 'no hint for this one')}")

    def _correct(self, i, ans):
        q = self.questions[i]
        key = _dec(q["key"])
        if q["type"] == "mc":
            return str(ans).strip().lower().strip("()") == str(key).lower()
        if q["type"] == "num":
            try:
                v = float(ans)
            except (TypeError, ValueError):
                return False
            tol = q.get("tol", 1e-6)
            return math.isclose(v, float(key), abs_tol=tol)
        norm = lambda s: "".join(str(s).lower().split()).replace("’", "'").replace("‘", "'")
        return norm(ans) in [norm(k) for k in (key if isinstance(key, list) else [key])]

    def answer(self, i, ans):
        if ans is None or ans == "" or ans == ...:
            print(f"Q{i}: replace the placeholder with your answer, e.g. q.answer({i}, 'b')")
            return None
        ok = self._correct(i, ans)
        self.results[i] = ok
        q = self.questions[i]
        print(("✅ Correct!" if ok else "❌ Not quite.") + f"   (Q{i}, score {self._score_str()})")
        if ok or self.results.get(f"_tries{i}", 0) >= 1:
            print("   " + q["explanation"])
        else:
            print("   Try again, or call q.hint(%d) / q.reveal(%d)." % (i, i))
        self.results[f"_tries{i}"] = self.results.get(f"_tries{i}", 0) + 1
        return ok

    def reveal(self, i):
        q = self.questions[i]
        key = _dec(q["key"])
        if q["type"] == "mc":
            key = f"({key}) {q['options'][key]}"
        print(f"Answer Q{i}: {key}\n   {q['explanation']}")

    def _score_str(self):
        done = [k for k in self.results if isinstance(k, int)]
        return f"{sum(self.results[k] for k in done)}/{len(self.questions)}"

    def score(self):
        done = sorted(k for k in self.results if isinstance(k, int))
        good = [k for k in done if self.results[k]]
        print(f"{self.title}: {len(good)}/{len(self.questions)} correct "
              f"({len(done)} attempted)")
        missing = [k for k in sorted(self.questions) if k not in done]
        wrong = [k for k in done if not self.results[k]]
        if wrong:
            print("  revisit:", ", ".join(map(str, wrong)))
        if missing:
            print("  not yet answered:", ", ".join(map(str, missing)))
        return len(good)

    def widget(self, i):
        """clickable question (multiple choice) or text box (numeric/text)"""
        try:
            import ipywidgets as W
            from IPython.display import display
        except ImportError:
            self.show(i)
            print("(install ipywidgets for clickable quizzes; meanwhile use q.answer)")
            return
        q = self.questions[i]
        out = W.Output()
        if q["type"] == "mc":
            sel = W.RadioButtons(options=[(f"({k}) {v}", k) for k, v in q["options"].items()],
                                 layout=W.Layout(width="auto"))
        else:
            sel = W.Text(placeholder="your answer")
        btn = W.Button(description="Check", button_style="primary")

        def go(_):
            with out:
                out.clear_output()
                self.answer(i, sel.value)
        btn.on_click(go)
        display(W.VBox([W.HTML(f"<b>Q{i}.</b> {q['question']}"), sel, btn, out]))


def build_bank(path, title, questions):
    """write a quiz bank; questions: list of dicts with plain 'answer' (encoded on save)"""
    qs = []
    for k, q in enumerate(questions, 1):
        q = dict(q)
        q["id"] = k
        q["key"] = _enc(q.pop("answer"))
        qs.append(q)
    with open(path, "w") as fh:
        json.dump(dict(title=title, questions=qs), fh, indent=1, ensure_ascii=False)
