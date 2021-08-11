# 語彙の定義
characters = "abcdefghijklmnopqrstuvwxyz!'(),-.:;? "
# その他特殊記号
extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS>
]
_pad = "~"

# NOTE: パディングを 0 番目に配置
symbols = [_pad] + extra_symbols + list(characters)

# 文字列⇔数値の相互変換のための辞書
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def num_vocab():
    """Get number of vocabraries

    Returns:
        int: Number of vocabraries

    Examples:

        >>> from ttslearn.tacotron.frontend.text import num_vocab
        >>> num_vocab()
        >>> 40
    """
    return len(symbols)


def text_to_sequence(text):
    """Convert text to sequence of numbers

    Args:
        text (str): Input text

    Returns:
        list: List of numbers

    Examples:

        >>> from ttslearn.tacotron.frontend.text import text_to_sequence
        >>> text_to_sequence("Hello world")
        >>> [1, 10, 7, 14, 14, 17, 39, 25, 17, 20, 14, 6, 2]

    """
    # 簡易のため、大文字と小文字を区別せず、全ての大文字を小文字に変換
    text = text.lower()

    # 文頭を表す<SOS>
    seq = [_symbol_to_id["^"]]

    # 本文
    seq += [_symbol_to_id[s] for s in text]

    # 文末を表す<EOS>
    seq.append(_symbol_to_id["$"])

    return seq


def sequence_to_text(seq):
    """Convert sequence of numbers to text

    Args:
        seq (list): Input sequence of numbers

    Returns:
        str: Text

    Examples:

        >>> from ttslearn.tacotron.frontend.text import sequence_to_text
        >>> sequence_to_text([1, 10, 7, 14, 14, 17, 39, 25, 17, 20, 14, 6, 2])
        >>> ['^', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '$']

    """
    return [_id_to_symbol[s] for s in seq]
