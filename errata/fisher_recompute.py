"""recompute the pooled p printed in README.md from the two per-model p values printed beside it.

correction notice, 2026-11-09. python standard library only.
run: python -I -B errata/fisher_recompute.py

README.md line numbers below are given twice: as read at commit d0079fa0 (before the
correction banner), and as of the notice (after the 7-line banner block was inserted).

three pooling rules are computed:
  fisher                 chi-square on -2*sum(ln p), 4 degrees of freedom (the README labels its pooled p "Fisher")
  stouffer, one-sided    z_i = inv_cdf(1 - p_i), pooled p = 1 - cdf(z)
  stouffer, two-sided    z_i = inv_cdf(1 - p_i/2) (each printed p read as two-sided), pooled p = 2*(1 - cdf(z))
"""
import math
from statistics import NormalDist

# per-model p values, quoted exactly as README.md prints them
#   "Gemma-2-2B-IT              26     [0, 7)    +0.535    0.663    0.013"   line 76 at d0079fa0 / line 83 as of the notice
#   "Llama-3.2-1B-Instruct      16     [0, 20)   +0.635    0.641    0.005"   line 77 at d0079fa0 / line 84 as of the notice
P_GEMMA = 0.013
P_LLAMA = 0.005

# pooled p printed in README.md
#   "cross-architecture     d = 0.584      Fisher p = 0.00018"   line 39 at d0079fa0 / line 46 as of the notice
#   "POOLED                                      +0.584             0.00018"   line 79 at d0079fa0 / line 86 as of the notice
P_PRINTED = 0.00018


def fisher(p1, p2):
    # chi-square with 4 degrees of freedom; closed-form survival function
    x = -2.0 * (math.log(p1) + math.log(p2))
    return x, math.exp(-x / 2.0) * (1.0 + x / 2.0)


def stouffer_one_sided(p1, p2):
    n = NormalDist()
    z = (n.inv_cdf(1.0 - p1) + n.inv_cdf(1.0 - p2)) / math.sqrt(2.0)
    return z, 1.0 - n.cdf(z)


def stouffer_two_sided(p1, p2):
    n = NormalDist()
    z = (n.inv_cdf(1.0 - p1 / 2.0) + n.inv_cdf(1.0 - p2 / 2.0)) / math.sqrt(2.0)
    return z, 2.0 * (1.0 - n.cdf(z))


def main():
    x, p_f = fisher(P_GEMMA, P_LLAMA)
    z1, p_s1 = stouffer_one_sided(P_GEMMA, P_LLAMA)
    z2, p_s2 = stouffer_two_sided(P_GEMMA, P_LLAMA)
    print("inputs (as printed in README.md): gemma p = %s, llama p = %s (2 models)" % (P_GEMMA, P_LLAMA))
    print("pooled p printed in README.md: %s (labelled Fisher)" % P_PRINTED)
    print("fisher:              x = %.6f (df = 4), p = %.6g (3 s.f.: %.3g)" % (x, p_f, p_f))
    print("stouffer one-sided:  z = %.6f, p = %.6g (3 s.f.: %.3g)" % (z1, p_s1, p_s1))
    print("stouffer two-sided:  z = %.6f, p = %.6g (3 s.f.: %.3g)" % (z2, p_s2, p_s2))
    tol = 0.05 * P_PRINTED
    for name, p in (("fisher", p_f), ("stouffer one-sided", p_s1), ("stouffer two-sided", p_s2)):
        same = abs(p - P_PRINTED) <= tol
        print("%s reproduces printed %s within 5%%: %s" % (name, P_PRINTED, "yes" if same else "no"))


if __name__ == "__main__":
    main()
