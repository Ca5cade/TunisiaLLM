"""
arabizi_utils.py - Tunisian Arabizi Transliteration and Code-Switching Utilities
Provides:
  1. Lexicon-based and phonetic transliteration from Arabic-script Derja to Arabizi
  2. Common colloquial Arabizi vocabulary & phrases
  3. Augmentation helpers for multi-dialect training and evaluation
"""

import re
import random
from typing import List, Dict, Tuple

LEXICON_AR_TO_ARABIZI = {
    "عسلامة": ["3aslema", "aslema", "3asslema", "ahla"],
    "شنوة": ["chnawa", "chnouwa", "chneya", "chnehiya"],
    "أحوالك": ["a7welek", "ahwelek", "a7welik"],
    "حوالك": ["a7welek", "7welek"],
    "اليوم": ["lyoum", "el youm"],
    "كيفاش": ["kifech", "kifeh", "kifeech"],
    "قداش": ["9adech", "9adeh", "kadech"],
    "شريت": ["chrit", "chrite"],
    "بربي": ["brabi", "bellehi", "yar7em waldik"],
    "باهي": ["bahi", "behi", "mrigel"],
    "يا": ["ya"],
    "غالي": ["ghali"],
    "مريڤل": ["mrigel", "cava", "jawwek behi"],
    "الحمد": ["el 7amdoullah", "elhamdoulilah"],
    "لله": ["lellah", "lillah"],
    "طماطم": ["tmatem"],
    "بطاطا": ["batata", "btata"],
    "كيلو": ["kilo", "kg"],
    "بصل": ["bsal"],
    "مليم": ["mlim", "millimes"],
    "دينار": ["dinar", "dt"],
    "دينارات": ["dinarat", "dinars"],
    "حلقة": ["boucle", "7al9a", "loop"],
    "تكرار": ["loop", "tekrar"],
    "بايثون": ["python"],
    "دالة": ["fonction", "function", "dala"],
    "قائمة": ["liste", "list"],
    "شرط": ["condition", "charth"],
    "تونس": ["tounes", "tunisie"],
    "القيروان": ["kairouan", "9irwan"],
    "قرطاج": ["carthage", "cartage"],
    "كسكسي": ["couscous", "kosksi"],
    "لبلابي": ["lablabi", "leblebi"],
    "كفتاجي": ["kafteji", "kafteji"],
    "عجة": ["ojja", "3ojja"],
    "بريك": ["brik"],
    "جربة": ["djerba", "jerba"],
    "المهدية": ["mahdia"],
    "قليبية": ["kelibia"],
    "طبرقة": ["tabarka"],
    "الحمامات": ["hammamet"],
    "سوسة": ["sousse"],
    "صفاقس": ["sfax"],
    "بنزرت": ["bizerte"],
    "توزر": ["tozeur"],
    "تصييفة": ["tasyifa", "vacances", "tsayef"],
    "بحر": ["b7ar", "plage"],
    "شط": ["chatt", "chott"],
    "احسبلي": ["e7sebli", "a7sebli"],
    "زيد": ["zid"],
    "عليهم": ["3lihom"],
    "مع": ["m3a", "+"],
    "إلا": ["illa", "-"]
}

CHAR_MAP = {
    'ع': '3',
    'ح': '7',
    'خ': '5',
    'ق': '9',
    'غ': '8',
    'ء': '2',
    'أ': 'a',
    'إ': 'e',
    'آ': 'a',
    'ا': 'a',
    'ش': 'ch',
    'ص': 's',
    'ض': 'dh',
    'ط': 't',
    'ظ': 'dh',
    'ث': 'th',
    'ذ': 'dh',
    'ة': 'a',
    'ي': 'i',
    'ى': 'a',
    'و': 'ou',
    'ب': 'b',
    'ت': 't',
    'ج': 'j',
    'د': 'd',
    'ر': 'r',
    'ز': 'z',
    'س': 's',
    'ف': 'f',
    'ك': 'k',
    'ل': 'l',
    'م': 'm',
    'ن': 'n',
    'ه': 'h',
    '؟': '?',
    '،': ','
}

def derja_to_arabizi(text: str, rng: random.Random = None) -> str:
    """Convert an Arabic Derja string to natural Tunisian Arabizi with lexicon substitutions."""
    if rng is None:
        rng = random.Random(42)

    tokens = re.findall(r"[\w']+|[^\w\s]", text, re.UNICODE)
    converted = []

    for t in tokens:
        if t in LEXICON_AR_TO_ARABIZI:
            choice = rng.choice(LEXICON_AR_TO_ARABIZI[t])
            converted.append(choice)
        elif t in CHAR_MAP:
            converted.append(CHAR_MAP[t])
        else:
            w_chars = []
            for c in t:
                w_chars.append(CHAR_MAP.get(c, c))
            converted.append("".join(w_chars))

    result = " ".join(converted)
    # Clean spacing around punctuation
    result = re.sub(r'\s+([?,.!])', r'\1', result)
    return result
