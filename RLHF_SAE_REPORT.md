# RLHF DEPTH SUPPRESSION — SAE Attribution Method

## The Definitive Test

Same method (circuit-tracer SAE attribution) that produced K=1.0302 (p=0.000051)
on Gemma-2-2B base, now run on google/gemma-2-2b-it.

---

## Results

| | Base (gemma-2-2b) | Instruct (gemma-2-2b-it) |
|--|---|---|
| **K ratio** | **1.0317** | **1.0384** |
| Surface mean | 8.3592 | 8.0698 |
| Insight mean | 8.1025 | 7.7714 |
| Delta mean | +0.2568 | +0.2984 |
| p-value (independent) | 0.000051 | 0.000004 |
| % expected direction | 83% | 87% |

**K delta**: -0.006707
**Suppression test p**: 0.852328
**Cohen's d**: -0.1946

## VERDICT: REVERSED: Instruct model shows more depth differentiation

---

*Generated at 2026-03-29T22:59:47*
*Fathom -- US Patent Pending 64/020,489 & 64/021,113*