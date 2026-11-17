# Per-experiment release checklist

- [ ] Input TIFFs archived, readable and identical to those analyzed (checksums)
- [ ] Synthetic generator source and settings archived, including seed
- [ ] Ground-truth vector field and coordinate convention archived
- [ ] Exact 2WCF source commit and branch recorded
- [ ] Grouped (current) or flat (historical) parameter snapshot archived
- [ ] Python and relevant numerical-library versions recorded
- [ ] Complete per-frame validation NPZ, PKL/WXF or appropriately equivalent arrays archived
- [ ] Primary metrics re-evaluated by a matching evaluator using the matching ground truth
- [ ] EPE on a common field + retained-only EPE + retention + angular error reported
- [ ] Zero-flow and low-speed conditions handled without invalid relative/angular metrics
- [ ] Bounds, border exclusion and sample spacing stated
- [ ] At least one diagnostic figure and QC plot included
- [ ] Publication plots traceable to CSV/NPZ inputs and scripts
- [ ] For Rankine: finite-displacement/chord ground truth and center estimator checked
- [ ] Historical raw artifacts preserved; any corrections performed in new derived directories
