"""arXiv LaTeX 源码 → 纯文本。

**这是全库质量的关键一环。** RL 论文的公式密度极高——贝尔曼方程、策略梯度推导、
收敛性证明——PDF 抽取会把它们毁成 ``r(s,a) + γ max Q(s ,a )`` 这类无法检索的
碎片。所以源码优先，PDF 只作回退（见 :mod:`rllib.extract.text`）。

三条设计取舍：

1. **数学环境原样保留**，不做任何"清理"。保留 ``$...$`` 与 ``\\begin{equation}``
   的 LaTeX 形式，比试图转成 Unicode 再出错要好。数学**宏**只特判了
   ``\\frac`` 和 ``\\sqrt``——它们有结构（分子分母），扁平成 ``ab`` 会丢掉
   分式关系；而 ``x^{2}`` 这类上下标不动，因为花括号原样保留就已经可读。
2. **符号表是有选择地加**，不是全量 Unicode 化。只映射检索时真正会用到的
   （``\\alpha`` → α、``\\nabla`` → ∇），这样用户搜"policy gradient α"能命中。
3. **参考文献整块丢弃**。它动辄占全文 15–25% 字符，且全是人名和期刊缩写，
   在语义检索里是纯噪声——查询"actor critic"会被参考文献里的
   "Actor-Critic" 标题刷屏。需要引用关系时从缓存源码重新解析，不必留在正文里。

输入来自 :mod:`rllib.fetch` 下载的 arXiv ``/src/`` 包，可能是 tar.gz、单个
gzip 压缩的 .tex、zip，也可能直接就是一个 .tex（老论文）。
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
import zipfile
from typing import Callable, Iterable

log = logging.getLogger(__name__)

#: 抽取器版本。改了这个文件就该 +1——质量报告靠它识别哪些文本是旧规则产出的。
#: 不重抽不会自己变好，但至少能查出来。
VERSION = "1"

#: 整个丢弃的环境。thebibliography 是噪声（见模块 docstring）；
#: tikz/pgf 是绘图代码；verbatim/lstlisting 是代码块。
_DROP_ENVS = (
    "thebibliography", "tikzpicture", "pgfpicture", "verbatim", "verbatim*",
    "lstlisting", "filecontents", "comment", "pspicture", "picture",
)

#: 只保留 caption 的浮动体。图表的语义几乎全在 caption 里，
#: 而坐标表、子图引用是纯噪音。
#:
#: **algorithm 不在其中**：伪代码是论文的实质内容（算法 1 往往就是全文核心），
#: 只留标题会把它整个抹掉。
_CAPTION_ENVS = (
    "figure", "figure*", "table", "table*", "wrapfigure", "sidewaysfigure",
    "sidewaystable", "subfigure",
)

#: 展开成 markdown 标题的章节命令，按层级
_SECTION_MACROS = {
    "part": "#", "chapter": "#", "section": "##",
    "subsection": "###", "subsubsection": "####", "paragraph": "#####",
}

#: 丢弃命令**及其参数**：只剩引用编号/标签，没有可读内容
_STRIP_ARG_MACROS = (
    "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear",
    "citeyearpar", "ref", "eqref", "pageref", "autoref", "cref", "Cref", "cpageref",
    "label", "index", "glossary", "bibliography", "bibliographystyle", "nocite",
    "hypersetup", "includegraphics", "vspace", "hspace", "vskip", "hskip",
    "documentclass", "usepackage", "setlength", "addtolength", "newcommand",
    "renewcommand", "providecommand", "def", "DeclareMathOperator", "newtheorem",
    "graphicspath", "captionsetup", "numberwithin", "pagestyle", "thispagestyle",
)

#: 拆掉命令、**保留参数**的格式宏
_UNWRAP_MACROS = (
    "textbf", "textit", "texttt", "textsc", "textrm", "textsf", "textsl",
    "textup", "textmd", "textnormal", "textsuperscript", "textsubscript",
    "emph", "mbox", "hbox", "text", "ensuremath", "mathrm", "mathbf", "mathcal",
    "mathbb", "mathit", "mathsf", "mathtt", "operatorname", "bm", "boldsymbol",
    "vec", "hat", "bar", "tilde", "dot", "ddot", "overline", "underline",
    "widehat", "widetilde", "url", "href", "footnote", "footnotetext", "thanks",
    "caption", "texorpdfstring", "protect",
    # \left \right \big \Big 这类**不带花括号参数**的命令：_apply_macro 取不到
    # 参数就直接删掉命令本身，留下来的 ( ) 定界符正好是可读的。
    "left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr",
    # 同样无参的排版命令。不删的话 \maketitle、\centering 这类会以字面形式
    # 混进正文，做检索时就是噪声词。
    "maketitle", "centering", "raggedright", "noindent", "newpage", "clearpage",
    "small", "footnotesize", "scriptsize", "normalsize", "large", "Large",
    "LARGE", "huge", "Huge", "tiny", "bfseries", "itshape", "ttfamily",
    "sffamily", "rmfamily", "scshape", "upshape", "vfill", "hfill", "hline",
    "toprule", "midrule", "bottomrule", "appendix", "tableofcontents",
)

#: 数学符号 → Unicode。检索时真正用得上的那些，不做全量 Unicode 化。
_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π",
    r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\sum": "Σ", r"\prod": "Π", r"\int": "∫", r"\oint": "∮",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\propto": "∝", r"\ll": "≪", r"\gg": "≫",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\circ": "∘", r"\otimes": "⊗",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\subseteq": "⊆",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅", r"\varnothing": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\neg": "¬", r"\land": "∧", r"\lor": "∨",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\leftrightarrow": "↔", r"\mapsto": "↦",
    r"\mathbb{R}": "ℝ", r"\mathbb{N}": "ℕ", r"\mathbb{E}": "𝔼",
    r"\mathbb{Z}": "ℤ", r"\mathbb{P}": "ℙ",
    # 算子：留着反斜杠会让 "max Q" 搜不到 "max"，而 \max_a' 这种写法在
    # Bellman 方程里几乎每篇都有
    r"\max": "max", r"\min": "min", r"\log": "log", r"\exp": "exp",
    r"\lim": "lim", r"\sup": "sup", r"\inf": "inf", r"\det": "det",
    r"\arg": "arg", r"\tr": "tr", r"\diag": "diag", r"\Var": "Var",
    r"\softmax": "softmax", r"\KL": "KL", r"\std": "std",
    # \mid 是条件概率的竖线（P(s' \mid s,a)）。不映射的话条件分布的写法
    # 会带着反斜杠留在正文里，"P(s'|s,a)" 这种检索词一条都命中不了。
    r"\mid": "|",
    # 最常见的一批自定义宏。理论上无法穷举作者的 \newcommand，
    # 但这几个在 RL 论文里的含义基本没有歧义（\E 期望、\R 实数域）。
    # 带 (?![a-zA-Z]) 断言，所以 \Re、\Rho 之类不会被误伤。
    r"\E": "E", r"\R": "ℝ",
}

#: 符号替换**必须带"后面不接字母"的断言**，不能直接用 str.replace。
#:
#: 反例（实测会踩）：``\le`` 是 ``\left`` 的前缀，``\in`` 是 ``\infty`` 的前缀，
#: ``\to`` 是 ``\top`` 的前缀。裸替换会把 ``\left(`` 变成 ``≤ft(``，
#: 把 ``\infty`` 变成 ``∈fty``——而且这种破坏是静默的，抽完根本看不出。
#: 按长度倒序排列只解决 ``\in``/``\infty`` 这一类，解决不了 ``\le``/``\left``
#: （后者压根不在符号表里）。
_SYMBOL_RE = re.compile(
    "(?:" + "|".join(re.escape(k) for k in sorted(_SYMBOLS, key=len, reverse=True)) + r")(?![a-zA-Z])"
)

_MAX_INCLUDE_DEPTH = 8


# --------------------------------------------------------------------------- #
# 解包
# --------------------------------------------------------------------------- #

#: arXiv 源码包里只关心这些扩展名的内容，图片字体一律忽略
_TEXT_EXTS = (".tex", ".ltx", ".bbl")

#: 花括号配对求值的起点。所有需要"取一个 {...} 参数"的地方都走它，
#: 因为正则数不清嵌套——``\textbf{a \emph{b} c}`` 用 ``[^{}]*`` 只能取到 ``a \emph{b``。
_BRACE_START = "{"


def _decode(raw: bytes) -> str:
    """arXiv 源码里 latin-1 和 utf-8 混杂，utf-8 失败时退 latin-1 而不是丢弃。

    丢弃坏字节会让公式里的重音符号变成替换符，进而让标题匹配失效；
    latin-1 至少是双射，不会丢信息。
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def unpack(data: bytes) -> dict[str, str]:
    """arXiv 源码字节 → ``{路径: 文本}``（只含 .tex/.ltx/.bbl）。

    支持 tar.gz / tar / zip / 单文件 gzip / 裸 .tex 五种形态，都是 arXiv
    实际会返回的。识别靠魔数而不是扩展名——arXiv 的 URL 里根本没有扩展名。
    """
    out: dict[str, str] = {}

    def _add(name: str, raw: bytes) -> None:
        if name.lower().endswith(_TEXT_EXTS) and len(raw) < 8 << 20:
            out[name] = _decode(raw)

    # zip
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        _add(info.filename, zf.read(info))
            return out
        except (zipfile.BadZipFile, RuntimeError) as exc:
            log.debug("zip 解包失败: %s", exc)
            return out

    # gzip：可能是 tar.gz，也可能是单个被 gzip 的 .tex
    if data[:2] == b"\x1f\x8b":
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        fh = tf.extractfile(member)
                        if fh is not None:
                            _add(member.name, fh.read())
            if out:
                return out
        except (tarfile.TarError, EOFError):
            pass
        try:
            return {"main.tex": _decode(gzip.decompress(data))}
        except (OSError, EOFError) as exc:
            log.debug("gzip 解包失败: %s", exc)
            return out

    # 未压缩的 tar
    if len(data) > 262 and data[257:262] == b"ustar":
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        fh = tf.extractfile(member)
                        if fh is not None:
                            _add(member.name, fh.read())
            return out
        except (tarfile.TarError, EOFError) as exc:
            log.debug("tar 解包失败: %s", exc)
            return out

    # 裸 .tex（老论文常见）
    if data[:5] != b"%PDF-":
        text = _decode(data)
        if "\\documentclass" in text or "\\begin{document}" in text:
            return {"main.tex": text}
    return out


def find_main(files: dict[str, str]) -> str | None:
    """挑出主 .tex 的文件名。

    判据分三档，因为 arXiv 上三种情况都真实存在：

    1. 既有 ``\\documentclass`` 又有 ``\\begin{document}`` —— 标准情况。
       有多个时取最大的那个（小的往往是模板/附录）。
    2. 只有 ``\\begin{document}`` —— 正文里 ``\\documentclass`` 被拆到了
       ``\\input`` 的文件里。
    3. 都不满足 —— 退而取最大的 .tex，至少不会完全空手而归。
    """
    if not files:
        return None
    tex = {k: v for k, v in files.items() if k.lower().endswith((".tex", ".ltx"))}
    if not tex:
        return None

    def pick(pool: dict[str, str]) -> str | None:
        if not pool:
            return None
        return max(pool, key=lambda k: len(pool[k]))

    both = {k: v for k, v in tex.items() if "\\documentclass" in v and "\\begin{document}" in v}
    if both:
        # main.tex / paper.tex 之类名字明确优先于"最大的"
        named = {k: v for k, v in both.items()
                 if re.search(r"(^|/)(main|paper|ms|article)\.(tex|ltx)$", k, re.I)}
        return pick(named) or pick(both)

    doc = {k: v for k, v in tex.items() if "\\begin{document}" in v}
    return pick(doc) or pick(tex)


# --------------------------------------------------------------------------- #
# 展开 \input / \include
# --------------------------------------------------------------------------- #

_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^{}]+)\}")


def _lookup(files: dict[str, str], name: str) -> str | None:
    """按 ``\\input{sec/intro}`` 找文件。作者写的路径和包内实际路径经常对不上，
    所以依次试：原样、补 ``.tex``、按 basename 唯一匹配。"""
    name = name.strip().replace("\\", "/")
    if name in files:
        return name
    for cand in (f"{name}.tex", f"{name}.ltx"):
        if cand in files:
            return cand
    base = name.rsplit("/", 1)[-1].lower()
    hits = [k for k in files if k.rsplit("/", 1)[-1].lower() in (base, f"{base}.tex")]
    return hits[0] if len(hits) == 1 else None


def _expand(tex: str, files: dict[str, str], seen: set[str], depth: int = 0) -> str:
    if depth >= _MAX_INCLUDE_DEPTH:
        return tex

    def repl(m: re.Match[str]) -> str:
        key = _lookup(files, m.group(1))
        if key is None or key in seen:
            # 找不到就留空而不是留原文：``\input{sec1}`` 出现在正文里是可读性噪声。
            # 已在 seen 里的直接跳过，防 A→B→A 的环。
            return ""
        seen.add(key)
        return _expand(files[key], files, seen, depth + 1)

    return _INCLUDE_RE.sub(repl, tex)


# --------------------------------------------------------------------------- #
# 花括号感知的宏处理
# --------------------------------------------------------------------------- #

def _take_arg(s: str, i: int) -> tuple[str, int]:
    """``s[i] == '{'``，返回 ``(括号内内容, 结束后的下标)``。嵌套安全。

    被转义的 ``\\{`` ``\\}`` 不计入深度——它们在 LaTeX 里是字面花括号。
    括号不闭合（坏源码）时按"到结尾"处理，不抛异常：13000 篇里总有几篇坏的，
    为此中断整批不值得。
    """
    depth = 0
    start = i + 1
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], n


def _transform_braced(
    text: str,
    pattern: re.Pattern[str],
    fn: Callable[[re.Match[str], list[str], list[str]], str | None],
) -> str:
    """按 ``pattern`` 扫描，把命令**及其花括号参数**一并消费掉再替换。

    为什么不能用 ``re.sub``：它只替换**匹配到的那一段**。``\\section{X}`` 这种
    "命令 + 参数"构造，正则没法匹配完整——参数里的花括号是嵌套的（``\\section{The
    $\\alpha$-divergence}``），正则数不清。若正则只匹配 ``\\section{``，替换之后
    参数 ``X}`` 会原样留在正文里，表现为**标题重复一次**、``\\frac`` 的分子分母
    各残留一份。实测就是这样，而且抽完后正文看着还挺正常，只有读到具体公式才
    发现每个分式都被复制了一遍。

    ``fn(m, opt, args)``：``opt`` 是 ``[...]`` 可选参数，``args`` 是紧随的
    ``{...}`` 内容。返回 ``None`` 表示放弃替换、原样保留——坏源码不要瞎猜。
    """
    out: list[str] = []
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:m.start()])
        j = m.end()
        opt: list[str] = []
        while j < len(text) and text[j] == "[":
            depth = 0
            k = j
            while k < len(text):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            opt.append(text[j + 1:k])
            j = _skip_ws(text, k + 1)
        args: list[str] = []
        while j < len(text) and text[j] == "{":
            arg, j = _take_arg(text, j)
            args.append(arg)
            j = _skip_ws(text, j)
        repl = fn(m, opt, args)
        out.append(text[m.start():j] if repl is None else repl)
        pos = j


def _apply_macro(text: str, names: Iterable[str], *, keep_arg: bool) -> str:
    """对 ``\\name{...}`` 做统一处理。

    ``keep_arg=True`` 拆掉命令保留内容（``\\textbf{x}`` → ``x``）；
    ``keep_arg=False`` 连内容一起丢（``\\cite{x}`` → 空）。
    """
    pattern = re.compile(r"\\(?:" + "|".join(sorted(set(names), key=len, reverse=True)) + r")\*?(?![a-zA-Z])")
    out: list[str] = []
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:m.start()])
        j = m.end()
        # 宏后面可能有可选参数 ``\caption[短标题]{长标题}``，跳过它
        while j < len(text) and text[j] == "[":
            depth = 0
            k = j
            while k < len(text):
                if text[k] == "[":
                    depth += 1
                elif text[k] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            j = k + 1
        # 可能连着多个参数 ``\href{url}{text}``：保留最后一个（可读的那个）
        args: list[str] = []
        while j < len(text) and text[j] == "{":
            arg, j = _take_arg(text, j)
            args.append(arg)
        if not args:
            pos = m.end()          # ``\left(`` 这类无参命令，原地继续
            continue
        if keep_arg:
            out.append(args[-1])
        pos = j


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def _frac(m: re.Match[str], opt: list[str], args: list[str]) -> str | None:
    """``\\frac{A}{B}`` → ``(A)/(B)``。

    这是唯一不能被简单扁平化的数学宏：丢掉命令会得到 ``AB``，
    分子分母糊在一起，贝尔曼方程里的折扣因子和重要性采样比就全废了。
    """
    if len(args) < 2:
        return None
    return f"({args[0]})/({args[1]})"


def _sqrt(m: re.Match[str], opt: list[str], args: list[str]) -> str | None:
    """``\\sqrt[n]{x}`` → ``sqrt[n](x)``，``\\sqrt{x}`` → ``sqrt(x)``。"""
    if not args:
        return None
    return f"sqrt[{opt[0]}]({args[0]})" if opt else f"sqrt({args[0]})"


def _section(m: re.Match[str], opt: list[str], args: list[str]) -> str | None:
    """``\\section{The $\\alpha$-divergence}`` → ``## The $\\alpha$-divergence``。"""
    if not args:
        return None
    return f"\n\n{_SECTION_MACROS[m.group(1)]} {args[0]}\n\n"


#: ``\frac`` 与它的同构体 ``\dfrac`` ``\tfrac`` 共用一套处理。
#: 只匹配命令名本身，参数由 _transform_braced 消费——把 ``{`` 写进正则
#: 会让替换范围对不上参数的结尾。
_FRAC_RE = re.compile(r"\\[dt]?frac(?![a-zA-Z])")
_SQRT_RE = re.compile(r"\\sqrt(?![a-zA-Z])")
_SECTION_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph)\*?(?![a-zA-Z])"
)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _strip_comments(tex: str) -> str:
    """去掉 ``%`` 注释。``\\%``（字面百分号）不算注释。

    已知不完美：``\\url{http://x/a%20b}`` 里的 ``%20`` 会被误当注释截掉。
    在 ML 论文里极少见，且损失只是 URL 后半段，不值得为此写一个逐 token 解析器。
    """
    return re.sub(r"(?<!\\)%.*", "", tex)


#: ``\caption[短标题]{长标题}`` 的命令部分
_CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?(?![a-zA-Z])")


def _captions(fragment: str, limit: int = 3) -> list[str]:
    """取出片段里所有 ``\\caption{...}`` 的参数。

    必须**主动提取**，不能用 ``_apply_macro`` 去"转换"：后者是变换器，会原样
    保留未匹配的文本，于是 ``\\begin{figure}``…``\\end{figure}`` 整块都被带回来——
    包括 ``\\includegraphics`` 和表格里的数字。实测就是这么漏出去的，
    而且漏出来的内容看着还挺像正文，只有读到具体段落才觉得不对。
    """
    out: list[str] = []
    for m in _CAPTION_RE.finditer(fragment):
        j = _skip_ws(fragment, m.end())
        if j < len(fragment) and fragment[j] == "{":
            arg, _ = _take_arg(fragment, j)
            arg = re.sub(r"\s+", " ", arg).strip()
            if arg:
                out.append(arg)
        if len(out) >= limit:      # 防御性上限：嵌套浮动体不该无限取下去
            break
    return out


def _caption_only(m: re.Match[str]) -> str:
    """浮动体整个换成它的 caption。图表的语义几乎全在 caption 里，
    剩下的坐标表 / tikz 代码 / 数字网格在语义检索里是纯噪声。"""
    caps = _captions(m.group(0))
    return f"\n{' '.join(caps)}\n" if caps else "\n"


def _drop_env(tex: str, env: str) -> str:
    return re.sub(
        rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
        "\n",
        tex,
        flags=re.S,
    )


def flatten(tex: str, files: dict[str, str]) -> str:
    """LaTeX 正文 → 纯文本。``files`` 用于展开 ``\\input``。"""
    tex = _strip_comments(tex)
    tex = _expand(tex, files, seen=set())

    for env in _DROP_ENVS:
        tex = _drop_env(tex, env)
    for env in _CAPTION_ENVS:
        tex = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
            _caption_only,
            tex,
            flags=re.S,
        )

    # 切正文。没有 \begin{document} 时（片段式源码）取全文，
    # 总比返回空字符串好——质量报告会把过短的标出来。
    if "\\begin{document}" in tex:
        tex = tex.split("\\begin{document}", 1)[1]
    if "\\end{document}" in tex:
        tex = tex.rsplit("\\end{document}", 1)[0]

    # 顺序有讲究，下面四组不能随意调换：
    #
    # 1. 章节 → 标题。必须在切正文之后，否则前言的 \section 也会被算进来。
    # 2. 符号表。必须在 _UNWRAP_MACROS **之前**——\mathbb 在拆宏名单里，
    #    先拆就把 \mathbb{R} 变成 R，后面 ℝ 的规则永远轮不上。
    # 3. \frac/\sqrt 特判。必须在符号表之前，因为 \frac 的输出含 () 与 /，
    #    而符号表里没有这些；放后面也一样，但先做更省事。
    # 4. 拆宏/丢参。放最后，前面三步产出的文本就不再被宏规则触碰。
    tex = _transform_braced(tex, _SECTION_RE, _section)
    tex = _transform_braced(tex, _FRAC_RE, _frac)
    tex = _transform_braced(tex, _SQRT_RE, _sqrt)
    tex = _SYMBOL_RE.sub(lambda m: _SYMBOLS[m.group(0)], tex)

    tex = re.sub(r"\\item\b", "\n- ", tex)
    tex = _apply_macro(tex, _STRIP_ARG_MACROS, keep_arg=False)
    tex = _apply_macro(tex, _UNWRAP_MACROS, keep_arg=True)

    # 环境首尾标记、间距命令、转义字符。
    # 开始标记要连可选参数一起吃（``\begin{figure}[t]``）——只吃 ``\begin{figure}``
    # 的话 ``[t]`` 会作为字面文本留在正文里。
    tex = re.sub(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?|\\end\{[^}]*\}", "\n", tex)
    tex = re.sub(r"\\(?:quad|qquad|enspace|thinspace|medspace|,|;|!|:)(?![a-zA-Z])", " ", tex)
    tex = re.sub(r"\\\\(?:\[[^\]]*\])?", "\n", tex)
    tex = re.sub(r"\\([%&$#_{}])", r"\1", tex)      # \% \& \$ \_ 等还原成字面字符
    tex = tex.replace("~", " ")
    # 收掉残留花括号。上下标要先剥（``\sum_{i=1}^{n}`` → ``Σ_i=1^n``），
    # 否则 ``{}`` 一删就剩 ``Σ_i=1^n`` 这种断在半截的形态。
    tex = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", tex)
    tex = re.sub(r"[{}]", "", tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\s+([.,;:)\]])", r"\1", tex)   # 删宏之后留下的 "word ." 空格
    tex = re.sub(r"([(\[])\s+", r"\1", tex)       # "\left( x" 拆完留下的 "( x"
    tex = re.sub(r"\n\s*\n\s*\n+", "\n\n", tex)
    tex = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", tex)
    return tex.strip()


def source_to_text(data: bytes) -> str | None:
    """arXiv 源码字节 → 正文。拿不到可用正文时返回 ``None``。

    返回 ``None`` 不是失败——有些论文的 ``/src/`` 其实返回 PDF
    （arXiv 对早期投稿如此），调用方会转而用 PyMuPDF 处理同一份字节。
    """
    files = unpack(data)
    if not files:
        return None
    main = find_main(files)
    if main is None:
        return None
    text = flatten(files[main], files)
    return text or None


# --------------------------------------------------------------------------- #
# 按章节切段（周报用）
# --------------------------------------------------------------------------- #

#: 标题行：``## Title``。**要求井号后有空白**——这不是吹毛求疵：
#: LaTeX 宏参数在扁平化后会留下 ``#1`` 这类以井号开头的行，
#: 不要求空白就会把它们当成标题。
_HEADING_RE = re.compile(r"^(#{1,5})[ \t]+(\S.*?)[ \t]*$", re.M)

#: 「作者自己说的缺口」的取用优先级，**从最有信息量到最兜底**。
#:
#: 这个顺序不是随手排的：``limitation`` / ``future work`` 是作者**主动**写下的
#: 没做的事；``discussion`` 常含局限但混着结果讨论；``conclusion`` 兜底——
#: 它覆盖面最广（主库实测 85%）但**常常只是复述摘要**，所以排最后。
#:
#: ⚠️ 2026-09-17 晚**撤回了** ``concluding remark``——加它和撤它间隔两小时，
#: 而**这两小时里我只检查了「改了哪几篇」，没检查「改出来的东西是什么」**。
#:
#: 加它的理由（「词表里只缺一个同义词」）听着对，但它在 47 篇里**只命中
#: ``2609.19074`` 一篇**，而那篇的 ``Concluding Remarks`` 正文是纯总结 +
#: 教学感想（结尾是 "We hope RLLBC-Lib is useful for the broader community"），
#: **既没有缺口也没有展望**。那是篇教学代码库论文，本来就没有研究缺口——
#: **v1 报 ``no_section`` 才是对的**，是我把正确的「没有」改成了错误的「有」，
#: 还把它记进了「修好的 5 篇」和「零回归」。
#:
#: 教训不是「词表该加还是该撤」，而是：**词表一动，抽出来的东西就整体变了，
#: 得重新过一遍人眼**——而当时没有任何检查覆盖「抽到的这段有没有信息量」。
#:
#: 撤回的代价：本语料里以 ``Concluding Remarks`` 收尾**且真有缺口**的一篇都没有
#: （47 篇里只有 ``2609.19074`` 命中过它），所以成本是 0。**但这是本语料的结论**，
#: 换个语料要重测。
GAP_SECTIONS: tuple[str, ...] = (
    "limitation", "future work", "future direction", "open problem",
    "discussion", "conclusion",
)

#: 段落长度的**合理区间**。这两个数是 47 篇试跑**之前**就写死的判据
#: （原文在 ``scripts/probe_sections.py`` 的模块 docstring 里），此处只是把它们
#: 从"报警线"提升为**挑选依据**——**不是照着试跑的数据回头拟合出来的阈值**。
#:
#: 它们要挡的是两种**根本不是段落**的东西：
#:
#: * 过短：``## Conclusion`` 底下只剩一句 ``\linespread0.88\selectfont``。实测
#:   ``2609.18333`` 只有 26 字——作者交的 ``sections/5_conclusion.tex``
#:   **整个文件才 20 字节**，也就是说正文压根不存在，不是我们切短了。
#: * 过长：``## Discussion`` 底下挂着一串下级小节没有收敛。实测 ``2609.18286``
#:   拿到 25,137 字，而它自己的 ``Conclusion``（1,581 字）才是真结论。
#:
#: ⚠️ ``scripts/probe_sections.py`` 里有一份同名同值的副本（``TOO_SHORT`` /
#: ``TOO_LONG``）。**那一份已经冻结、不要跟着改**——它产出的
#: ``section_trial.jsonl`` 是那次试跑的原始证据，改了就对不上账了。复制是
#: 有意为之，不是漏改：**证据脚本要能被重新执行出同一个数**，就得把它自己
#: 依赖的每个常量都钉在自己文件里。
BODY_MIN = 200
BODY_MAX = 6000


def _norm_heading(heading: str) -> str:
    """``"Limitations of Our Approach"`` → ``"limitations of our approach"``。

    去掉标点（真实标题里有 ``Conclusion:`` / ``Limitations.``）和**开头的序号**
    （``"5 Limitations"`` / ``"VI. Conclusion"``）——后者是手工写的编号，
    留着会让"按开头匹配"失效。
    """
    s = re.sub(r"[^a-z0-9 ]+", " ", heading.lower()).strip()
    return re.sub(r"^(?:\d+|[ivxlcdm]+) +", "", s)


#: ``Discussion of Robustness`` / ``Discussion: X`` / ``Discussion on Y`` ——
#: 讨论某个**具体话题**的小节，是结果讨论，不是作者自述的缺口。
#:
#: 实测（2026-09-17，``2609.18964``）：挑中的 ``Discussion of Robustness.``
#: 正文是 "We continue to evaluate robustness of our method within different
#: environmental setups…"，而这篇的 ``Conclusions``（671 字）才是结论。
#: 这正是 :func:`extract_sections` docstring 里预告过的那个风险键。
_TOPIC_DISCUSSION_RE = re.compile(
    r"\bdiscussion\b\s*(?:of|on|about|:|—|-)\s*\S", re.I
)

#: 带这些词的 ``Discussion of X`` **不算**话题讨论——``Discussion of Limitations``
#: 是货真价实的缺口节，不能连它一起排掉。
_GAP_WORDS = ("limitation", "future", "open problem")


def _matches_key(heading: str, key: str) -> bool:
    """**原标题** ``heading`` 是否命中优先键 ``key``。

    这是全库唯一一处"某个标题算不算缺口节"的判定，:func:`extract_sections`
    与 :func:`pick_gap_section` 共用它——两处各写一份的话，切出来的候选与
    挑中的那一段就会悄悄地不是同一套口径。
    """
    norm = _norm_heading(heading)
    if not norm.startswith(key):
        return False
    if key == "discussion" and _TOPIC_DISCUSSION_RE.search(heading):
        # 话题限定的 Discussion 一律排掉，**除非**它同时带着缺口词
        # （``Discussion of Limitations`` 那种——它是个真的缺口节）。
        return any(w in norm for w in _GAP_WORDS)
    return True


def has_gap_word(heading: str) -> bool:
    """这个标题里**有没有**缺口/展望词（``limitation`` / ``future`` / ``open problem``）。

    ⚠️ 它**不是判据**，别拿它当闸门——这是本库栽过好几回的形状（一张小清单
    被当成判据，然后它的洞不报错）。它只回答"标题里有没有这几个词"：

    ==========================  ============
    ``Conclusion and limitations``  有（``limitations``）
    ``Conclusions``                 没有
    ==========================  ============

    而后者照样可能在正文里写 future work。它的**唯一**用途是**排序**：周报人眼
    验收页把"标题里没有这些词"的那批排在前面（挑错的概率更高，先看）。

    ⚠️ 用词表就是 :data:`_GAP_WORDS` 那一份（与 :func:`_matches_key` 同一份），
    **不另抄一张**——两处各写一份的话，"哪些词算缺口词"这句判断就有了两个答案。
    """
    norm = _norm_heading(heading)
    return any(w in norm for w in _GAP_WORDS)


def list_headings(text: str) -> list[str]:
    """正文里的**全部**章节标题，按出现顺序。

    存在的唯一理由是回答「真没有 vs 没读到」：没抽到段落时，得能看见
    这篇论文**实际有哪些标题**，否则分不清"它确实没有 Limitations 一节"
    与"我们没读出来"——本库栽过好几次的那类病。
    """
    return [m.group(2).strip() for m in _HEADING_RE.finditer(text)]


def extract_sections(text: str, names: Iterable[str]) -> dict[str, str]:
    """扁平化正文 → ``{原标题: 段落正文}``，只留标题命中 ``names`` 的。

    **匹配按标题开头，不是包含**。实测踩过：按包含匹配时 ``discussion``
    命中了 ``## Results and discussion``——那是结果节，不是缺口节，
    于是整篇挑中的是 30,776 字的结果讨论。开头匹配把这一类切干净，
    同时仍然收得下 ``Limitations of Our Approach``、``Conclusion and Future Work``。

    一段的范围到**下一个同级或更高级标题**为止——不切下一级标题，
    因为 ``## Limitations`` 底下的 ``### Setup`` 属于它。

    ⚠️ **``discussion`` 是风险键**：``## Discussion: Relationship to X`` 这种
    "讨论某个具体关系"的小节也会被开头匹配收下。**2026-09-17 已收口**——
    交给 :func:`_matches_key` 统一排掉"讨论具体话题"的形态（带缺口词的除外）。
    **其余不确定的形态一律保留原标题交给下游/人眼判，不在代码里猜。**
    """
    keys = tuple(_norm_heading(n) for n in names)
    hits = [
        m for m in _HEADING_RE.finditer(text)
        if any(_matches_key(m.group(2), k) for k in keys)
    ]
    out: dict[str, str] = {}
    for m in hits:
        level, title = len(m.group(1)), m.group(2).strip()
        # 找本段终点：下一个 level <= 本级的标题
        end = len(text)
        for nxt in _HEADING_RE.finditer(text, m.end()):
            if len(nxt.group(1)) <= level:
                end = nxt.start()
                break
        body = text[m.end():end].strip()
        # 同名标题出现多次时保留**最长**的那一段：短的那次多半是
        # 一句 ``See Section 5 (Limitations)`` 式的交叉引用。
        if title not in out or len(body) > len(out[title]):
            out[title] = body
    return out


def pick_gap_section(sections: dict[str, str]) -> tuple[str, str] | None:
    """按 :data:`GAP_SECTIONS` 的优先级挑一段，返回 ``(原标题, 正文)``。

    同一优先级内有多段时取**最长**的（假阳性一般更短）。

    **但长度落在 :data:`BODY_MIN` … :data:`BODY_MAX` 之外的段落要先让路**，
    哪怕它排在更前的优先级上：那种长度说明它根本不是一段完整陈述，而不是
    说明它内容更丰富。实测（2026-09-17，47 篇）：

    ============  =========================================================
    ``2602.00408`` 优先级 5 的 ``Discussion`` 6,719 字 让路给 ``Conclusion`` 2,750 字
    ``2605.16692`` 优先级 5 的 ``Discussion`` **112 字** 让路给 ``Conclusion and limitations`` 1,302 字
    ``2609.18286`` 优先级 5 的 ``Discussion`` 25,137 字 让路给 ``Conclusion`` 1,581 字
    ============  =========================================================

    反过来，三篇长度正常的 ``Discussion``（1,060 / 1,120 / 2,563 字）**一篇都没被
    改动**——让路的是"畸形"，不是"优先级 5"。所以这不是把 ``conclusion`` 提到
    ``discussion`` 前面（那样会修好 3 篇、弄坏 3 篇），而是**只对畸形的让路**。

    全部命中都是畸形段落时退回按优先级取最长——**宁可交出一段可疑的原文，
    也不要交出一个空的**：空的那种会退化成"这篇没有内容可挖"，而它其实有。

    全都不命中返回 ``None``——**调用方必须把这个 None 计进一个显式的桶**，
    不能让它和"抽到了空字符串"混在一起（本库反复栽的「没有 vs 没读到」）。
    """
    if not sections:
        return None
    degenerate: tuple[str, str] | None = None
    for key in GAP_SECTIONS:
        cand = [(t, b) for t, b in sections.items() if _matches_key(t, key)]
        if not cand:
            continue
        good = [tb for tb in cand if BODY_MIN <= len(tb[1]) <= BODY_MAX]
        if good:
            return max(good, key=lambda tb: len(tb[1]))
        if degenerate is None:
            degenerate = max(cand, key=lambda tb: len(tb[1]))
    return degenerate
