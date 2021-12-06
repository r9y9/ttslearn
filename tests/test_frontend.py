import pyopenjtalk
from ttslearn.tacotron.frontend.openjtalk import pp_symbols


def test_pp_symbols_kurihara():
    # 参考文献に載っている例
    # Ref: Prosodic Features Control by Symbols as Input of Sequence-to-Sequence
    # Acoustic Modeling for Neural TTS
    for text, expected in [
        # NOTE: 参考文献では、「お伝えします」が一つのアクセント句になっているが、
        # pyopenjtalkでは、「お伝え」「します」の二つのアクセント句に分かれているので注意
        ("続いて、ニュースをお伝えします", "^tsu[zuite_nyu]usuo#o[tsutae#shi[ma]su$"),
        # NOTE: 参考文献の Table 4 のPP表記は、「横ですか」のアクセント核の位置が間違っていると思われます
        # 誤：「ヨ^コデス!カ」
        # 正：「ヨ^コデ!スカ」
        ("私の席は、あの婦人の横ですか。", "^wa[tashino#se]kiwa_a[no#fu[jiNno#yo[kode]suka$"),
    ]:
        actual = "".join(pp_symbols(pyopenjtalk.extract_fullcontext(text)))
        assert actual == expected


def test_pp_symbols_python_book():
    # Pythonで学ぶ音声合成に載せた例
    for text, expected in [
        # 10章
        ("一貫学習にチャレンジしましょう！", "^i[clkaNga]kushuuni#cha[re]Nji#shi[masho]o$"),
        ("端が", "^ha[shiga$"),
        ("箸が", "^ha]shiga$"),
        ("橋が", "^ha[shi]ga$"),
        ("今日の天気は？", "^kyo]ono#te]Nkiwa?"),
        # 4章
        ("日本語音声合成のデモです。", "^ni[hoNgooNseego]oseeno#de]modesu$"),
        # 6章
        (
            "深層学習に基づく音声合成システムです。",
            "^shi[Nsooga]kushuuni#mo[tozu]ku#o[Nseegooseeshi]sutemudesu$",
        ),
        # 8章
        ("ウェーブネットにチャレンジしましょう！", "^we]ebunecltoni#cha[re]Nji#shi[masho]o$"),
    ]:
        actual = "".join(pp_symbols(pyopenjtalk.extract_fullcontext(text)))
        assert actual == expected


def test_pp_symbols_accent_phrase():
    # https://github.com/r9y9/ttslearn/issues/26
    for text, expected in [
        ("こんにちは", "^ko[Nnichiwa$"),
        ("こ、こんにちは", "^ko_ko[Nnichiwa$"),
        ("ここ、こんにちは", "^ko[ko_ko[Nnichiwa$"),
        ("くっ、こんにちは", "^ku]cl_ko[Nnichiwa$"),
        ("きょ、こんにちは", "^kyo_ko[Nnichiwa$"),
        ("ん、こんにちは", "^N_ko[Nnichiwa$"),
        ("んっ、こんにちは", "^N]cl_ko[Nnichiwa$"),
    ]:
        actual = "".join(pp_symbols(pyopenjtalk.extract_fullcontext(text)))
        assert actual == expected
