#!/usr/bin/env python3
"""A long prompt the INDEXER can be wrong about.

`mkprompt.py` repeats one paragraph, which is right for measuring prefill and useless for judging
top-k: every compressed row looks like every other, so any selection scores the same and a broken
indexer looks identical to a working one. This builds varied text instead and plants ONE fact early,
far outside the sliding window, so answering needs the indexer to have admitted the row holding it.
"""
import sys, random

random.seed(7)
COND = ["atrial fibrillation","type 2 diabetes mellitus","stage 3 chronic kidney disease",
        "chronic obstructive pulmonary disease","rheumatoid arthritis","hypothyroidism",
        "peripheral vascular disease","gastro-oesophageal reflux","osteoporosis","gout"]
DRUG = ["apixaban","metformin","bisoprolol","furosemide","atorvastatin","levothyroxine",
        "allopurinol","omeprazole","alendronate","amlodipine"]
CITY = ["Bristol","Leeds","Cardiff","Dundee","Norwich","Exeter","Hull","Derby","Swansea","Preston"]

def para(i):
    return (f"Case {i}: a {random.randint(41,88)}-year-old {random.choice(['woman','man'])} from "
            f"{random.choice(CITY)} with {random.choice(COND)} and {random.choice(COND)} presented "
            f"with a {random.choice(['two','three','five','seven'])}-day history of "
            f"{random.choice(['dyspnoea','palpitations','syncope','epigastric pain','ankle swelling'])}. "
            f"Medications included {random.choice(DRUG)} {random.randint(1,80)} mg and "
            f"{random.choice(DRUG)} {random.randint(1,40)} mg. Blood pressure was "
            f"{random.randint(96,180)}/{random.randint(54,102)} mmHg, heart rate "
            f"{random.randint(48,132)}, creatinine {random.randint(52,240)} umol/L, haemoglobin "
            f"{random.randint(78,161)/10:.1f} g/dL. The working diagnosis was "
            f"{random.choice(['decompensated heart failure','pneumonia','acute kidney injury','pulmonary embolism'])}. ")

NEEDLE = ("Case 3 is the index case for this audit: the trial identifier assigned to Case 3 is "
          "ZANTHIC-4471, and no other case in this series carries a trial identifier. ")

def build(toks):
    out, i = [], 1
    while sum(len(x) for x in out) < toks * 4:
        out.append(para(i))
        if i == 3: out.append(NEEDLE)     # early, so the window cannot hold it
        i += 1
    t = "".join(out)[: toks * 4]
    return t + ("\n\nQuestion: what is the trial identifier assigned to Case 3? "
                "Answer with the identifier only.\nAnswer:")

for name, n in (("n4k.txt", 4000), ("n16k.txt", 16000), ("n32k.txt", 32000), ("n48k.txt", 48000), ("n64k.txt", 64000), ("n8k.txt", 8000)):
    open(name, "w").write(build(n))
    print(name, len(open(name).read()), "chars")
