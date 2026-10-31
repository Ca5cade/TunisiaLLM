"""
multiturn_data.py - High-Quality Multi-Turn Tunisian Derja Dialogue Generator
Generates realistic 2-turn and 3-turn conversations:
  1. Shopping calculations with follow-up items
  2. Programming tutorials with follow-up syntax / debugging
  3. Tunisian cooking recipes with follow-up seasoning / steps
  4. Tunisian tourism with follow-up hotel / activity suggestions
  5. Arabizi multi-turn code-switching
"""

import random
from typing import List, Dict

SYSTEM_PROMPTS = [
    "إنت مساعد ذكي ولطيف بالدارجة التونسية، تعاون بوضوح وطلاقة وتجاوب ديما على قد السؤال.",
    "إنت مساعد ذكي بالدارجة التونسية.",
    "Tu es un assistant IA tunisien qui parle couramment la Derja tunisienne avec du code-switching naturel en français et anglais."
]

def generate_multiturn_dialogues(num_dialogues: int = 150) -> List[Dict]:
    rng = random.Random(777)
    dialogues = []

    items = [
        ("طماطم", "tmatem", "كيلو", 1500),
        ("بطاطا", "batata", "كيلو", 1200),
        ("بصل", "bsal", "كيلو", 800),
        ("تفاح", "teffe7", "كيلو", 2500),
        ("حليب", "7lib", "باكوات", 1400),
        ("ياغورت", "yaourt", "كعبات", 500)
    ]

    for _ in range(num_dialogues):
        mode = rng.choice(["shopping", "coding", "food", "travel", "arabizi"])
        sys_p = rng.choice(SYSTEM_PROMPTS)

        if mode == "shopping":
            i1, _, u1, p1 = rng.choice(items)
            q1 = rng.randint(2, 4)
            res1 = q1 * p1
            d1 = res1 // 1000
            m1 = res1 % 1000

            i2, _, u2, p2 = rng.choice([it for it in items if it[0] != i1])
            q2 = rng.randint(1, 3)
            res2 = res1 + (q2 * p2)
            d2 = res2 // 1000
            m2 = res2 % 1000

            dialogues.append({
                "system": sys_p,
                "turns": [
                    {
                        "role": "user",
                        "content": f"شريت {q1} {u1} {i1} بـ {p1} مليم، قداش يطيحو؟"
                    },
                    {
                        "role": "assistant",
                        "content": f"<tool_call>calc: {q1} * {p1}</tool_call><tool_result>{res1}</tool_result>يطيحو {res1} مليم (يعني {d1} دينارات و {m1} مليم) يا باهي."
                    },
                    {
                        "role": "user",
                        "content": f"زيد عليهم {q2} {u2} {i2} سوم الكيلو {p2} مليم، قداش يولي المجموع الكل؟"
                    },
                    {
                        "role": "assistant",
                        "content": f"<tool_call>calc: {res1} + {q2} * {p2}</tool_call><tool_result>{res2}</tool_result>المجموع الكل يولي {res2} مليم (يعني {d2} دينارات و {m2} مليم) يا غالي."
                    }
                ],
                "category": "multiturn_shopping"
            })

        elif mode == "coding":
            topic = rng.choice(["function", "loop", "list"])
            if topic == "function":
                dialogues.append({
                    "system": sys_p,
                    "turns": [
                        {
                            "role": "user",
                            "content": "بربي كيفاش نعمل دالة (function) في بايثون؟"
                        },
                        {
                            "role": "assistant",
                            "content": "باش تصنع دالة في بايثون، تستعمل الكلمة def هكا:\n```python\ndef ejma3(a, b):\n    return a + b\n```\nالكود هذا يصنع دالة تاخذ زوز أرقام وترجع المجموع متاعهم."
                        },
                        {
                            "role": "user",
                            "content": "وكيفاش نعيطلها ونطبع النتيجة؟"
                        },
                        {
                            "role": "assistant",
                            "content": "ساهل برشا، تعيطلها باسمها وتعديلها الأرقام:\n```python\nres = ejma3(15, 20)\nprint(res)\n```\nوهكا تطبعلك 35 في الشاشة يا باهي."
                        }
                    ],
                    "category": "multiturn_coding"
                })
            elif topic == "loop":
                dialogues.append({
                    "system": sys_p,
                    "turns": [
                        {
                            "role": "user",
                            "content": "كيفاش نعمل حلقة تكرار for في بايثون؟"
                        },
                        {
                            "role": "assistant",
                            "content": "باش تعمل حلقة for في بايثون، تستعمل range():\n```python\nfor i in range(5):\n    print(i)\n```\nالكود هذا يدور 5 مرات ويطبع الأرقام من 0 لـ 4."
                        },
                        {
                            "role": "user",
                            "content": "وكيفاش نخليها تطبع كان الأرقام الزوجية؟"
                        },
                        {
                            "role": "assistant",
                            "content": "تنجم تستعمل الخطوة (step) في range:\n```python\nfor i in range(0, 10, 2):\n    print(i)\n```\nأو تحط شرط if داخل الحلقة:\n```python\nif i % 2 == 0:\n    print(i)\n```"
                        }
                    ],
                    "category": "multiturn_coding"
                })
            else:
                dialogues.append({
                    "system": sys_p,
                    "turns": [
                        {
                            "role": "user",
                            "content": "كيفاش نعمل قائمة (list) ونزيد فيها عنصر في بايثون؟"
                        },
                        {
                            "role": "assistant",
                            "content": "تصنع القائمة بـ `[]` وتزيد فيها باستعمال append:\n```python\nqayma = ['تفاح', 'بنان']\nqayma.append('غلة')\n```\nوهكا تولي القائمة فيها 3 عناصر."
                        },
                        {
                            "role": "user",
                            "content": "وكيفاش نفسخ عنصر منها؟"
                        },
                        {
                            "role": "assistant",
                            "content": "باش تفسخ عنصر، تستعمل remove() بالاسم أو pop() بالترتيب:\n```python\nqayma.remove('بنان')\n```\nوهكا يتنحى البنان من القائمة."
                        }
                    ],
                    "category": "multiturn_coding"
                })

        elif mode == "food":
            dialogues.append({
                "system": sys_p,
                "turns": [
                    {
                        "role": "user",
                        "content": "شنوة مكونات اللبلابي التونسي؟"
                    },
                    {
                        "role": "assistant",
                        "content": "اللبلابي التونسي مكوناتو: خبز شايح مقصوص طروف، حمص طايب سخون، كمون، هريسة عربي، زيت زيتونة، عظمة مروبة، وتن وكبار."
                    },
                    {
                        "role": "user",
                        "content": "وكيفاش نحضر الحمص باش يجي طري ومهروس بالباهي؟"
                    },
                    {
                        "role": "assistant",
                        "content": "تنفخ الحمص ليلة كاملة في الماء مع نصف مغرفة كربوناتو، ومن غدوة تغليه في ماء نظيف على نار هادية ساعتين حتى يولي طري زبدة."
                    }
                ],
                "category": "multiturn_food"
            })

        elif mode == "travel":
            dialogues.append({
                "system": sys_p,
                "turns": [
                    {
                        "role": "user",
                        "content": "وين تنصحني نصيف في تونس؟"
                    },
                    {
                        "role": "assistant",
                        "content": "في تونس ننصحك بالمهدية وقليبية لبحر صافي ورملة بيضاء، أو جربة للهدوء والنزل، أو الحمامات وسوسة للجو الشبابي والسهرات."
                    },
                    {
                        "role": "user",
                        "content": "شنوة أحسن بلاصة في جربة بالتحديد؟"
                    },
                    {
                        "role": "assistant",
                        "content": "في جربة ننصحك بسيدي المحرص وميدون للنزل والبحورات الرملية، وحومة السوق للصناعات التقليدية، وقرية قلالة للفخار."
                    }
                ],
                "category": "multiturn_travel"
            })

        else:  # arabizi
            dialogues.append({
                "system": sys_p,
                "turns": [
                    {
                        "role": "user",
                        "content": "3aslema khouya, chnawa a7welek lyoum?"
                    },
                    {
                        "role": "assistant",
                        "content": "Ahla bik ya ghali! Labes el 7amdoullah kol chay mrigel. W inti cava? Chnouwa najjem n3awnek bih lyoum?"
                    },
                    {
                        "role": "user",
                        "content": "brabi kifech na3mel boucle for fi python?"
                    },
                    {
                        "role": "assistant",
                        "content": "Bach ta3mel boucle for fi python, testa3mel range() hakka ya bahi:\n```python\nfor i in range(5):\n    print(i)\n```\nEl code hedha ydour 5 marrat w yetba3 el a3ded men 0 l 4."
                    }
                ],
                "category": "multiturn_arabizi"
            })

    return dialogues

