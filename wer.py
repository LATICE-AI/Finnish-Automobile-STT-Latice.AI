#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import unicodedata

import pandas as pd
import jiwer
from pydub import AudioSegment

CSV_PATH = "result.csv"
DATASET_PATH = "dataset.json"

DIGIT_TO_WORD = {
    "0": "nolla",
    "1": "yksi",
    "2": "kaksi",
    "3": "kolme",
    "4": "neljä",
    "5": "viisi",
    "6": "kuusi",
    "7": "seitsemän",
    "8": "kahdeksan",
    "9": "yhdeksän",
}

FRACTION_PATTERNS = [
    (re.compile(r"(?i)\b1\s*[-/]\s*2\b"), "puoli"),
    (re.compile(r"(?i)\bpuol(i|en)\b"), "puoli"),
]

CORRECTIONS = {
    "kylla": "kyllä",
    "kylla.": "kyllä.",
    "tehda": "tehdä",
    "haluttaisinko": "haluttaisinko",
    "huolta": "huolto",
    "huollon": "huollon",
    "tanaan": "tänään",
    "huomenna": "huomenna",
    "icao": "icao",
    "icloud": "icloud",
    "iclcoud": "icloud",
    "icluod": "icloud",
    "icloudcom": "icloud com",
    "gmailcom": "gmail com",
    "hotmailcom": "hotmail com",
    "ajettu": "ajettu",
    "tonniä": "tonnia",
    "kaksisataa": "kaksi sata",
    "kolmesataa": "kolme sata",
    "neljasataa": "neljä sata",
    "viisisataa": "viisi sata",
    "kuusisataa": "kuusi sata",
    "seitsemansataa": "seitsemän sata",
    "kahdeksansataa": "kahdeksan sata",
    "yhdeksansataa": "yhdeksän sata",
    "sataviisikymmenta": "sata viisikymmentä",
    "sataviisikymmentä": "sata viisikymmentä",
    "satakaksikymmenta": "sata kaksikymmentä",
    "satakaksikymmentä": "sata kaksikymmentä",
    "satakuuskymmenta": "sata kuusikymmentä",
    "satakuuskymmentä": "sata kuusikymmentä",
    "satayhdeksankymmenta": "sata yhdeksänkymmentä",
    "satayhdeksänkymmentä": "sata yhdeksänkymmentä",
    "kaksituhatta": "kaksi tuhatta",
    "kolmetuhatta": "kolme tuhatta",
    "neljatuhatta": "neljä tuhatta",
    "viisituhatta": "viisi tuhatta",
    "kuusituhatta": "kuusi tuhatta",
    "seitsemantuhatta": "seitsemän tuhatta",
    "kahdeksantuhatta": "kahdeksan tuhatta",
    "yhdeksantuhatta": "yhdeksän tuhatta",
    "nelja": "neljä",
    "neljaa": "neljä",
    "yheksän": "yhdeksän",
    "yksii": "yksi",
    "phonetically": "phonettisesti",
}

NUMBER_WORDS = {
    0: "nolla",
    1: "yksi",
    2: "kaksi",
    3: "kolme",
    4: "neljä",
    5: "viisi",
    6: "kuusi",
    7: "seitsemän",
    8: "kahdeksan",
    9: "yhdeksän",
    10: "kymmenen",
    11: "yksitoista",
    12: "kaksitoista",
    13: "kolmetoista",
    14: "neljätoista",
    15: "viisitoista",
    16: "kuusitoista",
    17: "seitsemäntoista",
    18: "kahdeksantoista",
    19: "yhdeksäntoista",
    20: "kaksikymmentä",
    30: "kolmekymmentä",
    40: "neljäkymmentä",
    50: "viisikymmentä",
    60: "kuusikymmentä",
    70: "seitsemänkymmentä",
    80: "kahdeksankymmentä",
    90: "yhdeksänkymmentä",
}


FILLER_PATTERN = re.compile(
    r"\b(öö+|ää+|aa+|eh+|hm+|hmm+|emm+|mmm+|oo+|ooh+|ah+|noh+|niinku|joo+|juu+)\b"
)

LETTER_WORDS = {
    "aarne": "a",
    "aapo": "a",
    "aarni": "a",
    "aaro": "a",
    "bertta": "b",
    "beeta": "b",
    "celsius": "c",
    "daavid": "d",
    "eemeli": "e",
    "eevert": "e",
    "faarao": "f",
    "felix": "f",
    "gideon": "g",
    "heikki": "h",
    "helena": "h",
    "iivari": "i",
    "iida": "i",
    "jussi": "j",
    "jani": "j",
    "kalle": "k",
    "kari": "k",
    "lauri": "l",
    "leevi": "l",
    "maija": "m",
    "matti": "m",
    "mikko": "m",
    "niilo": "n",
    "nora": "n",
    "otto": "o",
    "paavo": "p",
    "petri": "p",
    "risto": "r",
    "sakari": "s",
    "seppo": "s",
    "tarmo": "t",
    "teppo": "t",
    "tuomo": "t",
    "urho": "u",
    "veikko": "v",
    "vilho": "v",
    "yrjö": "y",
    "yrjo": "y",
    "zeta": "z",
    "seta": "z",
    "viski": "w",
    "äiti": "ä",
    "aiti": "ä",
    "öljy": "ö",
    "oljy": "ö",
}


def _ascii_variant(text: str) -> str:
    return (
        text.replace("ä", "a")
        .replace("ö", "o")
        .replace("å", "a")
    )


def _strip_repetitions(text: str) -> str:
    """Compress elongated onomatopoeic repetitions (e.g. zzzz -> z)."""
    return re.sub(r"([a-zåäö])\1{2,}", r"\1", text)


def _apply_corrections(text: str) -> str:
    for wrong, right in CORRECTIONS.items():
        text = text.replace(wrong, right)
        ascii_wrong = _ascii_variant(wrong)
        if ascii_wrong != wrong:
            text = text.replace(ascii_wrong, right)
    return text


DIGIT_WORD_TO_DIGIT = {
    "nolla": "0",
    "yksi": "1",
    "yks": "1",
    "yksii": "1",
    "yhen": "1",
    "kaksi": "2",
    "kolme": "3",
    "neljä": "4",
    "nelja": "4",
    "viisi": "5",
    "viis": "5",
    "viisii": "5",
    "kuusi": "6",
    "seitsemän": "7",
    "seitsemän": "7",
    "seiska": "7",
    "kahdeksan": "8",
    "kahdeksan": "8",
    "ysi": "9",
    "yhdeksän": "9",
    "yheksan": "9",
}


def _words_to_number_tokens(tokens):
    converted = []
    for token in tokens:
        replacement = DIGIT_WORD_TO_DIGIT.get(token.lower())
        if replacement is not None:
            converted.append(replacement)
        else:
            converted.append(token)
    return converted


def _standardize_numbers(text: str) -> str:
    # unify decimal separators (13.45 -> 13,45) then remove commas later
    text = re.sub(r"(\d+)[\.,](\d+)", r"\1,\2", text)
    # collapse spaced digits (2 400 -> 2400)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return text


def _register_number_word(mapping, word, value):
    mapping[word] = value
    ascii_word = _ascii_variant(word)
    if ascii_word != word:
        mapping[ascii_word] = value


def _register_multiplier(mapping, word, value):
    mapping[word] = value
    ascii_word = _ascii_variant(word)
    if ascii_word != word:
        mapping[ascii_word] = value


WORD_TO_NUMBER = {}
for num, word in NUMBER_WORDS.items():
    _register_number_word(WORD_TO_NUMBER, word, num)

for tens in range(20, 100, 10):
    for ones in range(1, 10):
        combined = NUMBER_WORDS[tens] + NUMBER_WORDS[ones]
        _register_number_word(WORD_TO_NUMBER, combined, tens + ones)

for digit, word in DIGIT_TO_WORD.items():
    _register_number_word(WORD_TO_NUMBER, word, int(digit))

MULTIPLIER_WORDS = {}
for word in ["sata", "sataa", "satainen", "sadatta"]:
    _register_multiplier(MULTIPLIER_WORDS, word, 100)
for word in ["tuhat", "tuhatta", "tuhannen", "tuhansi", "tonni", "tonnia", "tonnin", "tonnit"]:
    _register_multiplier(MULTIPLIER_WORDS, word, 1000)
for word in ["miljoona", "miljoonan", "miljoonaa", "miljoonaan"]:
    _register_multiplier(MULTIPLIER_WORDS, word, 1_000_000)


def _combine_digit_tokens(tokens, start_index):
    combined = tokens[start_index]
    end_index = start_index + 1
    while end_index < len(tokens) and tokens[end_index].isdigit():
        combined += tokens[end_index]
        end_index += 1
    return combined, end_index


def _next_token(tokens, index):
    return tokens[index] if 0 <= index < len(tokens) else None


def _token_to_number(token: str):
    if token is None:
        return None
    if token.isdigit():
        return int(token)
    if token in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[token]
    ascii_token = _ascii_variant(token)
    return WORD_TO_NUMBER.get(ascii_token)


def _get_multiplier(token: str) -> int:
    if token is None:
        return 1
    if token in MULTIPLIER_WORDS:
        return MULTIPLIER_WORDS[token]
    ascii_token = _ascii_variant(token)
    return MULTIPLIER_WORDS.get(ascii_token, 1)


def _letter_from_word(token: str, prev_token: str, next_token: str):
    letter = LETTER_WORDS.get(token)
    if letter is None:
        letter = LETTER_WORDS.get(_ascii_variant(token))
    return letter


def _consume_number_phrase(tokens, start_index):
    total = 0
    current = 0
    consumed = 0
    idx = start_index

    while idx < len(tokens):
        token = tokens[idx]
        if not token:
            break

        number = _token_to_number(token)
        multiplier = _get_multiplier(token)

        if number is not None:
            current += number
            idx += 1
            consumed += 1
            continue

        if multiplier > 1:
            if current == 0:
                current = 1
            current *= multiplier
            idx += 1
            consumed += 1
            total += current
            current = 0
            continue

        break

    total += current
    if consumed == 0:
        return None, 0
    return total, consumed


def _normalize_tokens(tokens):
    normalized = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token:
            i += 1
            continue

        if token.isdigit():
            combined_str, next_index = _combine_digit_tokens(tokens, i)
            value = int(combined_str)
            idx = next_index

            while True:
                multiplier = _get_multiplier(_next_token(tokens, idx))
                if multiplier > 1:
                    value *= multiplier
                    idx += 1
                else:
                    break

            extra_value, consumed_extra = _consume_number_phrase(tokens, idx)
            if consumed_extra:
                value += extra_value
                idx += consumed_extra

            normalized.append(f"num{value}")
            i = idx
            continue

        number_value, consumed = _consume_number_phrase(tokens, i)
        if consumed:
            normalized.append(f"num{number_value}")
            i += consumed
            continue

        letter = _letter_from_word(token, None, None)
        if letter is not None:
            normalized.append(letter)
            i += 1
            continue

        normalized.append(token)
        i += 1
    return normalized


def _collapse_tokens(tokens):
    collapsed = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("num"):
            collapsed.append(token)
            i += 1
            continue

        if len(token) == 1 and token.isalpha():
            letters = [token]
            j = i + 1
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                letters.append(tokens[j])
                j += 1
            collapsed.append("".join(letters))
            i = j
            continue

        collapsed.append(token)
        i += 1
    return collapsed


def _merge_alphanumeric_tokens(tokens):
    merged = []
    for token in tokens:
        if token.startswith("num"):
            digits = token[3:]
            if digits and merged and merged[-1].isalpha():
                merged[-1] = merged[-1] + digits
                continue
        merged.append(token)
    return merged


def _split_segments(text: str):
    if not text:
        return []
    segments = re.split(r"[.!?\n]+", text)
    return [seg.strip() for seg in segments if seg.strip()]


def normalize_for_wer(s: str) -> str:
    """Normalisation adaptée au finnois pour le calcul du WER."""
    if s is None:
        return ""
    s = str(s).replace("⁄", "/")
    s = unicodedata.normalize("NFKC", s).casefold()

    s = _standardize_numbers(s)
    s = _apply_corrections(s)

    for pattern, replacement in FRACTION_PATTERNS:
        s = pattern.sub(replacement, s)

    s = s.replace("@", " at ")
    s = s.replace("&", " ja ")

    s = _strip_repetitions(s)
    s = FILLER_PATTERN.sub(" ", s)

    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"(?<=\d)(?=[a-zåäö])", " ", s)
    s = re.sub(r"(?<=[a-zåäö])(?=\d)", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    tokens = s.split()
    tokens = _words_to_number_tokens(tokens)
    normalized_tokens = _normalize_tokens(tokens)
    normalized_tokens = _collapse_tokens(normalized_tokens)
    normalized_tokens = _merge_alphanumeric_tokens(normalized_tokens)
    normalized_tokens = [
        _ascii_variant(token) if not token.startswith("num") else token
        for token in normalized_tokens
    ]

    return " ".join(normalized_tokens)

def compute_wer(truth_raw: str, hyp_raw: str) -> float:
    """Calcule le WER avec normalisation propre"""
    truth_segments = _split_segments(truth_raw)
    hyp_segments = _split_segments(hyp_raw)

    if truth_segments and hyp_segments and len(truth_segments) == len(hyp_segments):
        segment_wers = []
        for t_seg, h_seg in zip(truth_segments, hyp_segments):
            t_norm = normalize_for_wer(t_seg)
            h_norm = normalize_for_wer(h_seg)
            if t_norm == "" and h_norm == "":
                segment_wers.append(0.0)
            else:
                segment_wers.append(jiwer.wer(t_norm, h_norm))
        if segment_wers:
            return min(1.0, float(sum(segment_wers) / len(segment_wers)))

    truth_normalized = normalize_for_wer(truth_raw)
    hyp_normalized = normalize_for_wer(hyp_raw)
    
    if truth_normalized == "" and hyp_normalized == "":
        return 0.0
    if truth_normalized == "" and hyp_normalized != "":
        return 1.0
    
    # Utiliser jiwer.wer directement
    value = jiwer.wer(truth_normalized, hyp_normalized)
    return min(1.0, float(value))

def pick_col(df, candidates):
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for name in candidates:
        if name in low:
            return low[name]
    for c in cols:
        cl = c.lower()
        if any(name in cl for name in candidates):
            return c
    return None

def load_dataset_map(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    # Handle the structure with "data" array
    if "data" in data:
        for item in data["data"]:
            audio_file = item.get("audio_file") or item.get("audio_path", "")
            transcription = item.get("transcription") or item.get("text", "")
            if audio_file:
                base = os.path.basename(audio_file)
                base_l = base.lower()
                noext = os.path.splitext(base_l)[0]
                m[base_l] = transcription
                m[noext] = transcription
    else:
        # Fallback for other structures
        for k, v in data.items():
            base = os.path.basename(k)
            base_l = base.lower()
            noext = os.path.splitext(base_l)[0]
            m[base_l] = v
            m[noext] = v
    return m

def _sanitize_display(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r", " ").replace("\n", " ")
    s = s.replace("⁄", "/")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii", errors="ignore")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    try:
        if not os.path.exists(audio_path):
            return 0.0
        
        # Try different audio formats
        try:
            audio = AudioSegment.from_mp3(audio_path)
        except:
            try:
                audio = AudioSegment.from_wav(audio_path)
            except:
                try:
                    audio = AudioSegment.from_file(audio_path)
                except:
                    return 0.0
        
        return len(audio) / 1000.0  # pydub returns milliseconds, convert to seconds
    except:
        return 0.0

def main():
    csv_path = "result.csv"  # Utiliser le fichier de résultats STT
    dataset_path = "dataset.json"  # Utiliser notre dataset.json

    df = pd.read_csv(csv_path, encoding='utf-8')
    dataset_map = load_dataset_map(dataset_path)

    audio_col = pick_col(df, ["sample","audio","file","path","audio_path","filepath","wav","wav_path","utterance"])
    stt_col   = pick_col(df, ["stt","model","engine","provider","system","asr"])
    hyp_col   = pick_col(df, ["transcription","text","transcript","prediction","hypothesis","output","result","recognized_text","asr_text"])

    if hyp_col is None:
        print("colonne de transcription introuvable.")
        sys.exit(1)

    ref_col = pick_col(df, ["reference","ref","ground_truth","target","gold","truth","expected"])
    response_time_col = pick_col(df, ["response_time","latency","time","duration"])
    
    # Set to None to allow all models, or set to specific models to filter
    # allowed_models = {"gemini-2.0-flash", "groq-whisper-large-v3", "save_test_py", "elevenlabs", "deepgram-nova-3"}
    allowed_models = None  # uncomment to see all models
    
    wers = []
    stts = []
    hyps = []
    response_times = []
    audio_durations = []
    
    # Group results by reference text for better display
    results_by_ref = {}
    first_non_empty_printed = False

    for _, row in df.iterrows():
        hyp = str(row[hyp_col]) if hasattr(row, '__getitem__') and pd.notna(row[hyp_col]) else ""
        if ref_col is not None and ref_col in row:
            ref = "" if pd.isna(row[ref_col]) else str(row[ref_col])
        else:
            if audio_col is not None and pd.notna(row[audio_col]):
                base = os.path.basename(str(row[audio_col]))
                key = os.path.splitext(base.lower())[0]
                ref = dataset_map.get(key) or dataset_map.get(base.lower()) or ""
                ref = ref.get("text") if isinstance(ref, dict) else ref
            else:
                ref = ""

        # treat raw-empty lines as separators: print blank line and skip scoring
        if (ref.strip() == "") and (hyp.strip() == ""):
            if first_non_empty_printed:
                print("")
            continue
            
        stt_name = str(row[stt_col]) if stt_col and pd.notna(row[stt_col]) else "unknown"
        
        # skip models not in allowed list (if filtering is enabled)
        if allowed_models is not None and stt_name not in allowed_models:
            continue

        # skip if no reference found in dataset
        if ref.strip() == "":
            continue

        w = compute_wer(ref, hyp)
        wers.append(w)
        stts.append(stt_name)
        hyps.append(hyp.strip())
        
        # collect response times
        if response_time_col and pd.notna(row[response_time_col]):
            try:
                rt = float(row[response_time_col])
                response_times.append(rt)
            except (ValueError, TypeError):
                pass
        
        # collect audio duration
        if audio_col and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            # Construct full path to audio file
            audio_path = os.path.join("Audio", audio_filename)
            duration = get_audio_duration(audio_path)
            audio_durations.append(duration)
        
        # Group by reference text + audio type (_raw or _aug)
        disp_ref = _sanitize_display(ref)
        disp_hyp = _sanitize_display(hyp)
        
        # Determine audio type from filename
        audio_type = ""
        if audio_col is not None and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            if "_raw" in audio_filename.lower():
                audio_type = "_raw"
            elif "_aug" in audio_filename.lower():
                audio_type = "_aug"
        
        # Create unique key combining ref text and audio type
        ref_key = f"{disp_ref}{audio_type}"
        
        if ref_key not in results_by_ref:
            results_by_ref[ref_key] = []
        results_by_ref[ref_key].append((stt_name, disp_hyp, w))
    
    # Find STTs with at least one non-empty hyp
    stt_has_non_empty = {}
    for i, stt in enumerate(stts):
        if stt not in stt_has_non_empty:
            stt_has_non_empty[stt] = False
        if hyps[i] != "":
            stt_has_non_empty[stt] = True
    valid_stts = {stt for stt, has_non_empty in stt_has_non_empty.items() if has_non_empty}
    
    # Display results grouped by reference
    for ref_key, results in results_by_ref.items():
        # NOTE: On garde TOUS les échantillons, même si les WER sont identiques
        # (commenté le filtre qui sautait les WER identiques)
        # wer_scores = [wer_score for _, _, wer_score in results]
        # if len(set(wer_scores)) == 1:
        #     continue
            
        if not first_non_empty_printed:
            first_non_empty_printed = True
        else:
            print()
        
        # Extract ref text and audio type from key
        if ref_key.endswith("_raw"):
            ref_text = ref_key[:-4]
            audio_type_label = " (SIMPLE)"
        elif ref_key.endswith("_aug"):
            ref_text = ref_key[:-4]
            audio_type_label = " (HARD)"
        else:
            ref_text = ref_key
            audio_type_label = ""
            
        print(f"resultat attendu : \"{ref_text}\"{audio_type_label}")
        for stt_name, hyp_text, wer_score in results:
            if stt_name in valid_stts:
                print(f"{stt_name} = \"{hyp_text}\" -> {wer_score:.3f}")

    # create dataframe with all data including response times
    data_dict = {"stt": stts, "wer": wers, "hyp": hyps}
    if len(response_times) == len(stts):
        data_dict["response_time"] = response_times
    else:
        # pad with NaN if lengths don't match
        rt_padded = response_times + [float('nan')] * (len(stts) - len(response_times))
        data_dict["response_time"] = rt_padded[:len(stts)]
    
    out = pd.DataFrame(data_dict)
    
    # Filter out STTs where all hyp are empty
    out = out[out["stt"].isin(valid_stts)]
    out = out.drop(columns=["hyp"])
    
    if out["stt"].nunique() > 1 or (out["stt"].nunique()==1 and out["stt"].iloc[0] != "unknown"):
        print("\naverage WER by stt:")
        grp = out.groupby("stt", dropna=False)["wer"].mean().sort_values()
        cnt = out.groupby("stt", dropna=False)["wer"].size()
        best_wer = grp.iloc[0]  # le meilleur (plus petit WER)
        
        for name, mean_wer in grp.items():
            if mean_wer == best_wer:
                print(f"{name} -> {mean_wer:.3f}")
            else:
                if best_wer > 0:
                    diff_pct = ((mean_wer - best_wer) / best_wer) * 100
                    print(f"{name} -> {mean_wer:.3f} (-{diff_pct:.2f}%)")
                else:
                    print(f"{name} -> {mean_wer:.3f}")
            
        # calcul WER=0 et WER=1 par STT
        print("\nnumber of WER=0 and WER=1 by stt:")
        for name in grp.index:
            stt_wers = out[out["stt"] == name]["wer"]
            wer_0_count = (stt_wers == 0.0).sum()
            wer_1_count = (stt_wers == 1.0).sum()
            print(f"{name} -> WER=0: {wer_0_count}, WER=1: {wer_1_count}")
            
        # calcul latence moyenne par STT
        if "response_time" in out.columns:
            print("\naverage latency by stt:")
            rt_grp = out.groupby("stt", dropna=False)["response_time"].mean().sort_values()
            rt_cnt = out.groupby("stt", dropna=False)["response_time"].count()
            best_rt = rt_grp.iloc[0]  # le meilleur (plus rapide)
            
            for name, mean_rt in rt_grp.items():
                if name in rt_cnt and rt_cnt[name] > 0:
                    if mean_rt == best_rt:
                        print(f"{name} -> {mean_rt:.3f}s")
                    else:
                        if best_rt > 0:
                            diff_pct = ((mean_rt - best_rt) / best_rt) * 100
                            print(f"{name} -> {mean_rt:.3f}s (+{diff_pct:.2f}%)")
                        else:
                            print(f"{name} -> {mean_rt:.3f}s")
    else:
        overall = float(out["wer"].mean()) if len(out) > 0 else float('nan')
        print(f"\naverage WER -> {overall:.3f}")
        
        # calcul WER=0 et WER=1 global
        if len(out) > 0:
            wer_0_count = (out["wer"] == 0.0).sum()
            wer_1_count = (out["wer"] == 1.0).sum()
            print(f"WER=0: {wer_0_count}, WER=1: {wer_1_count}")
            
        # calcul latence moyenne global
        if "response_time" in out.columns and len(out) > 0:
            avg_latency = out["response_time"].mean()
            print(f"average latency -> {avg_latency:.3f}s")
    
    # calcul latence moyenne globale si pas déjà affiché
    if out["stt"].nunique() > 1 and "response_time" in out.columns and len(out) > 0:
        avg_latency = out["response_time"].mean()
        print(f"\nglobal average latency -> {avg_latency:.3f}s")
    
    # calcul nombre total d'audios uniques testés et durée totale
    unique_audios = set()
    audio_duration_map = {}
    
    # recalcul pour avoir les vrais audios uniques
    for _, row in df.iterrows():
        stt_name = str(row[stt_col]) if stt_col and pd.notna(row[stt_col]) else "unknown"
        
        # skip models not in allowed list (if filtering is enabled)
        if allowed_models is not None and stt_name not in allowed_models:
            continue
            
        if audio_col and pd.notna(row[audio_col]):
            audio_filename = str(row[audio_col])
            unique_audios.add(audio_filename)
            
            # calculate duration only once per unique audio
            if audio_filename not in audio_duration_map:
                audio_path = os.path.join("Audio", audio_filename)
                duration = get_audio_duration(audio_path)
                audio_duration_map[audio_filename] = duration
    
    total_audios = len(unique_audios)

class TeeOutput:
    """Write to both stdout and a file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

if __name__ == "__main__":
    # Redirect output to both terminal and file
    output_file = "wer_results.txt"
    tee = TeeOutput(output_file)
    sys.stdout = tee
    
    try:
        main()
    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nRésultats sauvegardés dans {output_file}")
