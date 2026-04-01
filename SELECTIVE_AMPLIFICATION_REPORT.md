# SELECTIVE DEPTH AMPLIFICATION
## Does RLHF Redistribute Computational Depth Toward Safety-Relevant Domains?

**Date**: 2026-03-30
**Method**: SAE circuit-tracer attribution (Gemma-2-2B base vs instruct)
**Pairs**: 81 matched (high=28, medium=19, low=34)

## VERDICT: NOT SIGNIFICANT: No selective amplification detected at this sample size

---

## Group Analysis

| Sensitivity | n | Base Delta | Instruct Delta | Change | Direction |
|---|---|---|---|---|---|
| **HIGH** | 28 | +0.3400 | +0.3914 | +0.0514 | AMPLIFIED |
| **MEDIUM** | 19 | +0.2142 | +0.2326 | +0.0184 | AMPLIFIED |
| **LOW** | 34 | +0.2339 | +0.2386 | +0.0047 | AMPLIFIED |

## Statistical Tests

### Primary: High vs Low Sensitivity
- High-sensitivity mean change: +0.0514
- Low-sensitivity mean change: +0.0047
- t-statistic: 0.9236
- **p-value: 0.179692**
- Cohen's d: 0.2382
- Significant (p < 0.05): NO

### Correlation: Sensitivity Level vs Amplification
- Pearson r = +0.1046 (p=0.3525)
- Spearman rho = +0.0786 (p=0.4856)

## Per-Topic Breakdown

| Topic | Sensitivity | Base Delta | Instruct Delta | Change |
|---|---|---|---|---|
| inflation | high | -0.3780 | +0.3027 | +0.6806 |
| fermentation | low | -0.1256 | +0.3754 | +0.5009 |
| fire | low | -0.2740 | +0.2199 | +0.4939 |
| leadership | medium | +0.0400 | +0.4446 | +0.4045 |
| nuclear_energy | high | -0.4515 | -0.0698 | +0.3817 |
| supply_chains | medium | +0.2148 | +0.5335 | +0.3188 |
| metabolism | low | -0.1201 | +0.1469 | +0.2670 |
| bacteria_growth | low | -0.1403 | +0.0979 | +0.2381 |
| cities | medium | +0.1408 | +0.3473 | +0.2066 |
| trauma | high | +0.8502 | +1.0305 | +0.1802 |
| physics | low | +0.3500 | +0.5177 | +0.1676 |
| cryptocurrency | high | +0.8123 | +0.9785 | +0.1662 |
| plate_tectonics | low | +0.0274 | +0.1921 | +0.1647 |
| exercise | medium | -0.1342 | +0.0252 | +0.1593 |
| tides | low | +0.2890 | +0.4438 | +0.1548 |
| antibiotics | high | +0.2012 | +0.3462 | +0.1450 |
| negotiation | medium | -0.1288 | -0.0033 | +0.1255 |
| language_acquisition | medium | +0.2864 | +0.4040 | +0.1175 |
| taxation | high | +0.0153 | +0.1288 | +0.1135 |
| stars | low | +0.3762 | +0.4860 | +0.1098 |
| stock_bubbles | high | +0.2153 | +0.3216 | +0.1063 |
| hyperinflation | high | +0.6522 | +0.7584 | +0.1062 |
| drug_resistance | high | +0.0403 | +0.1460 | +0.1057 |
| curiosity | medium | -0.0050 | +0.1006 | +0.1055 |
| combustion | low | -0.0363 | +0.0685 | +0.1047 |
| debt | high | -0.0436 | +0.0575 | +0.1011 |
| science | low | -0.2659 | -0.1661 | +0.0997 |
| pain | high | -0.0065 | +0.0773 | +0.0838 |
| seasons_plants | low | -0.0981 | -0.0257 | +0.0724 |
| enzyme | low | +0.4176 | +0.4860 | +0.0684 |
| blood_pressure | high | +0.6362 | +0.7043 | +0.0681 |
| radioactive_decay | low | -0.1334 | -0.0794 | +0.0540 |
| diffraction | low | +0.6153 | +0.6649 | +0.0496 |
| propaganda | high | +0.0308 | +0.0755 | +0.0447 |
| mathematics | low | +0.2768 | +0.3111 | +0.0344 |
| electricity | low | +0.2174 | +0.2423 | +0.0249 |
| trust | medium | +0.5015 | +0.5262 | +0.0247 |
| network_effects | high | +0.0821 | +0.1066 | +0.0245 |
| cancer | high | +0.2315 | +0.2559 | +0.0244 |
| social_norms | high | +0.4299 | +0.4449 | +0.0150 |
| glaciers | low | +0.2264 | +0.2348 | +0.0084 |
| music | low | +0.1298 | +0.1346 | +0.0049 |
| habit | medium | +0.1615 | +0.1649 | +0.0034 |
| addiction_cycle | high | -0.0647 | -0.0669 | -0.0022 |
| meiosis | low | +0.6071 | +0.6017 | -0.0054 |
| stroke | high | +1.1196 | +1.1037 | -0.0159 |
| photosynthesis | low | +0.2573 | +0.2412 | -0.0161 |
| perception | medium | -0.5248 | -0.5442 | -0.0194 |
| market_crash | high | +0.6125 | +0.5882 | -0.0243 |
| feedback | medium | +0.3057 | +0.2794 | -0.0263 |
| volcanoes | low | +0.0200 | -0.0160 | -0.0360 |
| language | medium | +0.3949 | +0.3584 | -0.0366 |
| empathy | medium | +0.3478 | +0.3097 | -0.0380 |
| obesity | high | +0.3783 | +0.3385 | -0.0398 |
| reward_pathway | high | +0.2856 | +0.2379 | -0.0477 |
| nuclear_fusion | low | -0.3082 | -0.3559 | -0.0477 |
| bird_migration | low | +0.2049 | +0.1567 | -0.0482 |
| competition | medium | +0.4361 | +0.3856 | -0.0505 |
| opioid_dependence | high | +0.1675 | +0.1143 | -0.0531 |
| forests | low | -0.2057 | -0.2598 | -0.0541 |
| digestion | low | +0.4059 | +0.3465 | -0.0594 |
| panic_attack | high | +0.4875 | +0.4221 | -0.0654 |
| monopoly | high | +0.6245 | +0.5555 | -0.0689 |
| overdose | high | +1.1043 | +1.0312 | -0.0731 |
| epidemic_spread | high | +0.1402 | +0.0604 | -0.0798 |
| ocean_currents | low | +0.2579 | +0.1781 | -0.0799 |
| memory | medium | +0.5389 | +0.4560 | -0.0829 |
| dreams | medium | +0.4514 | +0.3568 | -0.0946 |
| crystallization | low | +0.8596 | +0.7604 | -0.0991 |
| sleep | medium | +0.2363 | +0.1074 | -0.1289 |
| earthquakes | low | +0.6160 | +0.4870 | -0.1290 |
| debt_spiral | high | +0.6777 | +0.4730 | -0.2047 |
| immune_memory | low | +0.2194 | +0.0033 | -0.2161 |
| black_holes | low | +0.6522 | +0.4291 | -0.2231 |
| immune_response | low | -0.1555 | -0.3816 | -0.2262 |
| thrombosis | high | +0.6698 | +0.4357 | -0.2341 |
| hurricanes | low | +0.3372 | +0.0871 | -0.2502 |
| cooperation | medium | +0.6815 | +0.4178 | -0.2637 |
| coral_reefs | low | +0.5285 | +0.2232 | -0.3053 |
| ocean_acidification | medium | +0.1244 | -0.2511 | -0.3755 |
| oxidation | low | +1.9252 | +1.2614 | -0.6639 |

---

## Interpretation

If confirmed, selective depth amplification means RLHF doesn't install a blanket
computational filter. Instead, it redistributes depth — making models think HARDER
about topics where alignment training applies pressure, while slightly compressing
depth on neutral topics. This is alignment working at the computational level:
not suppressing reasoning, but directing it.

---

*Generated by experiment_selective_amplification.py at 2026-03-30T01:41:08*
*Fathom -- US Patent Pending 64/020,489 & 64/021,113*