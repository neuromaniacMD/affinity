#!/usr/bin/env python3
"""Build long clinical-prose prompts for prefill measurement. ~4 chars a token for English prose."""
import sys

PARA = (
    "The patient is a 67-year-old woman with atrial fibrillation, type 2 diabetes mellitus, and stage 3 "
    "chronic kidney disease who presented to the emergency department with a three-day history of "
    "progressive dyspnoea on exertion, orthopnoea, and bilateral ankle swelling. Her medications on "
    "admission included apixaban 5 mg twice daily, metformin 1 g twice daily, bisoprolol 5 mg once daily, "
    "and furosemide 40 mg once daily. On examination she was afebrile, blood pressure 148/92 mmHg, heart "
    "rate 96 beats per minute and irregularly irregular, oxygen saturation 92 percent on room air. Jugular "
    "venous pressure was elevated at 8 cm above the sternal angle, there were bibasal crepitations, and "
    "pitting oedema to the mid-shin bilaterally. Laboratory investigations showed haemoglobin 10.2 g/dL, "
    "white cell count 8.4, platelets 233, sodium 133 mmol/L, potassium 4.8 mmol/L, urea 14.2 mmol/L, "
    "creatinine 168 umol/L, and NT-proBNP markedly elevated at 4820 pg/mL. Chest radiography demonstrated "
    "cardiomegaly with upper lobe diversion and small bilateral pleural effusions. Transthoracic "
    "echocardiography showed a left ventricular ejection fraction of 38 percent with global hypokinesis, "
    "moderate mitral regurgitation, and a dilated left atrium. "
)

for name, toks in (("p4k.txt", 4000), ("p16k.txt", 16000)):
    want = toks * 4
    text = (PARA * (want // len(PARA) + 1))[:want]
    text += "\n\nQuestion: summarise the admission in one sentence, then list the three most likely causes of decompensation.\nAnswer:"
    open(name, "w").write(text)
    print(name, len(text), "chars ~", len(text) // 4, "tokens")
