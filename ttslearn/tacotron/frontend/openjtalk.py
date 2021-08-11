import re

# 音素 (+pau/sil)
phonemes = [
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
    "pau",
    "sil",
]

extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS> (通常)
    "?",  # 文の末尾を表す特殊記号 <EOS> (疑問系)
    "_",  # ポーズ
    "#",  # アクセント句境界
    "[",  # ピッチの上がり位置
    "]",  # ピッチの下がり位置
]

_pad = "~"

# NOTE: 0 をパディングを表す数値とする
symbols = [_pad] + extra_symbols + phonemes


_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def pp_symbols(labels, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels

    The algorithm is based on [Kurihara 2021] [1]_ with some tweaks.

    Args:
        labels (HTSLabelFile): List of labels
        drop_unvoiced_vowels (bool): Drop unvoiced vowels. Defaults to True.

    Returns:
        list: List of phoneme + prosody symbols

    .. ipython::

        In [11]: import ttslearn

        In [12]: from nnmnkwii.io import hts

        In [13]: from ttslearn.tacotron.frontend.openjtalk import pp_symbols

        In [14]: labels = hts.load(ttslearn.util.example_label_file())

        In [15]: " ".join(pp_symbols(labels.contexts))
        Out[15]: '^ m i [ z u o # m a [ r e ] e sh i a k a r a ... $'

    .. [1] K. Kurihara, N. Seiyama, and T. Kumano, “Prosodic features control by
        symbols as input of sequence-to-sequence acoustic modeling for neural tts,”
        IEICE Transactions on Information and Systems, vol. E104.D, no. 2,
        pp. 302–311, 2021.
    """
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)  # type: ignore

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1:
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")

    return PP


def num_vocab():
    """Get number of vocabraries

    Returns:
        int: Number of vocabraries

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import num_vocab
        >>> num_vocab()
        >>> 52
    """
    return len(symbols)


def text_to_sequence(text):
    """Convert phoneme + prosody symbols to sequence of numbers

    Args:
        text (list): text as a list of phoneme + prosody symbols

    Returns:
        list: List of numbers

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import text_to_sequence
        >>> text_to_sequence(["^", "m", "i", "[", "z","o", "$"])
        >>> [1, 31, 27, 6, 49, 35, 2]
    """
    return [_symbol_to_id[s] for s in text]


def sequence_to_text(seq):
    """Convert sequence of numbers to phoneme + prosody symbols

    Args:
        seq (list): Input sequence of numbers

    Returns:
        list: List of phoneme + prosody symbols

    Examples:

        >>> from ttslearn.tacotron.frontend.openjtalk import sequence_to_text
        >>> sequence_to_text([1, 31, 27, 6, 49, 35, 2])
        >>> ['^', 'm', 'i', '[', 'z', 'o', '$']
    """
    return [_id_to_symbol[s] for s in seq]
