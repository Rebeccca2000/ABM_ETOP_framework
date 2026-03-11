# Response to Reviewers

Use the same comment IDs as `review/reviewer_response_tracker.csv`.

## Cover Note

We revised the manuscript against the current evidence pipeline rather than the earlier single-run workflow. The revised paper now uses the merged 30-seed summaries as the primary quantitative evidence, uses the deterministic fixed-policy sweep only as a mechanism-control experiment, and removes unsupported monetized benefit claims. The tracker, manuscript, appendix, change log, unresolved-issue log, and standalone build checklist were updated together. The paper package is now self-contained at the figure level, includes manuscript-local Elsevier support files in `paper/tex`, and compiles locally with `latexmk -pdf -bibtex main.tex`; the checklist records the remaining minor layout warnings and the final upload steps.

## Editor

- `ED-1`
  Comment: Major revision invited; provide a line-by-line response and outline every change.
  Response: We expanded the response package to atomic comment coverage, synchronized it with the manuscript and appendix edits, bundled the required Elsevier support files into the paper package, completed a successful local build, and logged the remaining evidence-limit items explicitly instead of softening unsupported claims.
  Manuscript changes: `paper/tex/main.tex`; `paper/tex/appendix.tex`; all review-package files.

- `BUILD-CHECK`
  Comment: Confirm the final local build path and standalone-package readiness.
  Response: We vendored `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` into `paper/tex`, ran `source /etc/profile.d/modules.sh && module load texlive/20220321 && cd paper/tex && latexmk -pdf -bibtex main.tex`, and confirmed successful local compilation. The updated checklist now distinguishes that verified local build from the final upload checks and notes the remaining minor layout warnings.
  Manuscript changes: `paper/tex/main.tex`; `paper/tex/appendix.tex`; `review/OVERLEAF_UPLOAD_CHECKLIST.md`.

## Reviewer 1

### High-Risk Issues

- `R1-HR1`
  Comment: Progressive allocation appears emergent although it may be imposed by bounds.
  Response: The current optimisation scripts use uniform `0.0-0.8` bounds for all 12 allocation variables. We now describe low-income-favoring allocations as observed outcomes of the revised experiments, not as consequences of unequal progressive bounds.
  Manuscript changes: Abstract; Methods; Results; Cross-policy synthesis; Conclusions; Appendix bounds table.

- `R1-HR2`
  Comment: The optimisation framing is unclear; is the study mono-objective or multi-objective?
  Response: The current workflow uses three separate single-objective Bayesian optimisation runs, followed by cross-policy comparison. We removed wording that implied a single joint multi-objective optimiser.
  Manuscript changes: Abstract; Introduction; Optimisation Methodology; Conclusions; Appendix methodology.

- `R1-HR3`
  Comment: How was stochasticity handled and were there enough repetitions?
  Response: The revised manuscript now reports 30-seed merged summaries with means and 95 percent confidence intervals for each sampled FPS value and adds a deterministic fixed-policy control experiment to interpret stochastic irregularity.
  Manuscript changes: Abstract; Methods; Results; Appendix robustness section.

- `R1-HR4`
  Comment: Why are the metrics not unified in scale, and what are their units?
  Response: We now state explicitly that mode-share equity is dimensionless, travel-time equity is reported in minutes, and total system travel time is reported in minutes. Cross-objective comparison is presented only through explicit normalization for decision support.
  Manuscript changes: Metric definitions in main text and appendix; cross-policy figure captions.

- `R1-HR5`
  Comment: Can an open-source implementation be provided for reproducibility?
  Response: We added a code-availability and reproducibility statement using `https://github.com/Rebeccca2000/ABM_ETOP_framework` and described which scripts and outputs underpin the revised manuscript.
  Manuscript changes: Methods section; review package.
  Note: Public visibility and archival details were not independently verified from the local environment.

### Remaining Weaknesses

- `R1-WX1`
  Comment: The ABM design and validation need to be clearer before drawing conclusions.
  Response: We rewrote the manuscript to present ABM-ETOP as a stylised subsidy-allocation scenario tool, not a calibrated city-scale forecast, regenerated Figure 1 as a cleaner local architecture schematic, and replaced the old validation language with clearer internal-check and reproducibility statements.
  Manuscript changes: Introduction; Framework; Limitations; Appendix calibration/validation.

- `R1-WX2`
  Comment: Why are private cars and private bikes excluded?
  Response: We now state that the current policy instrument is an operating subsidy delivered through the MaaS platform, so the present scope is shared modes, public transport, MaaS bundles, and walking rather than ownership, parking, or fuel policies.
  Manuscript changes: Introduction; Limitations.

- `R1-WX3`
  Comment: The simulation setup is unclear.
  Response: We now distinguish the 144 within-run time steps from optimisation iterations, explain the 12-dimensional space, document the 8 Latin-hypercube initial samples and the objective-specific BO updates, and report the current 100x100, 200-commuter setup.
  Manuscript changes: Methods; Appendix parameter and optimisation sections.

- `R1-WX4`
  Comment: References and sources are missing for many parameter values and assumptions.
  Response: We replaced the old mixed parameter/baseline tables with an audited implementation-parameter appendix, documented the commuter-attribute weights and recovered PBS job-resource requests from preserved scripts, and removed unsupported monetized-benefit claims from the active manuscript.
  Manuscript changes: Appendix parameter tables; main-text economic-appraisal section.
  Note: The revised appendix is explicit about which values are implementation settings rather than externally calibrated values.

- `R1-WX5`
  Comment: Algorithm 1 needs clearer definitions of `S`, `P`, `A`, and `E`, and pricing/BPR update timing.
  Response: We clarified the vector definitions in the appendix and explained that pricing and routing updates occur within the 144-step simulation horizon, while realized trip requests are endogenous rather than part of the exogenous environment.
  Manuscript changes: Main framework text; Appendix parameter specifications.

- `R1-WX6`
  Comment: The baseline modal-share table looks unrealistic and does not sum to 100 percent.
  Response: We removed the legacy baseline modal-share table from the manuscript rather than defending an incomplete export as a full mode split.
  Manuscript changes: Results section.

- `R1-WX7`
  Comment: Explain the allocation patterns and how trip counts vary with subsidies.
  Response: We now report the dominant allocation components at each revised optimum and use the trip-summary outputs to show that the total-system result is driven by moderate trip-time changes on a fairly stable trip-count range.
  Manuscript changes: Results section; Cross-policy synthesis.

- `R1-WX8`
  Comment: Do subsidy-use shares simply reflect group sizes, and why are some subsidies unused?
  Response: We now distinguish aggregate allocation shares from population shares and explain declining budget use as a saturation effect rather than an accounting anomaly.
  Manuscript changes: Mode-share results; Policy implications.

- `R1-WX9`
  Comment: Is the recommended range consistent with previous findings, and can equity improve by worsening outcomes for others?
  Response: We removed the old 54-72 percent claim and replaced it with seed-based FPS ranges and plateaus. We also added explicit wording that a better equity score is not treated as proof of a Pareto improvement, because disadvantaged users can improve, others can worsen, or both can move together. The synthesis section now ties that scenario-specific interpretation back to transport-equity scholarship that prioritises disadvantaged groups rather than equal nominal benefits for all travellers.
  Manuscript changes: Cross-policy synthesis; Policy implications; Conclusions.

### Overstatements

- `R1-O1`
  Comment: The title is too broad.
  Response: We narrowed the title to multimodal subsidy allocation policies.
  Manuscript changes: Title.

- `R1-O2`
  Comment: The manuscript claims validation through comparisons that are not actually presented.
  Response: We removed the unsupported benchmarking claim and describe Table 2 as a conceptual capability comparison.
  Manuscript changes: Literature review.

- `R1-O3`
  Comment: The paper equates horizontal equity with system efficiency.
  Response: We now describe total system travel time as a system-efficiency metric and avoid treating it as a complete horizontal-equity proxy.
  Manuscript changes: Metric definitions; Results; Conclusions; Appendix metrics section.

- `R1-O4`
  Comment: The implementation-gap claim is too strong for such a simplified scenario.
  Response: We now present ABM-ETOP as a stylised scenario-analysis tool and move empirical implementation to future work.
  Manuscript changes: Introduction; Policy implications; Methodological contribution; Limitations.

### Questions

- `R1-Q1`
  Comment: How does the framework transfer to city scale?
  Response: We now answer this directly in the implementation discussion: city-scale transfer would require replacing the stylised grid with an empirical network, local fare and demand data, calibrated behavioural parameters, and rerunning the seed-based and deterministic checks on that case. We keep the current manuscript's scope explicit by stating that this workflow has not yet been executed for a city-scale case.
  Manuscript changes: Policy implications; Limitations; Appendix calibration status.

- `R1-Q2`
  Comment: Is the approach applicable to real-world cases and how?
  Response: We now explain how the workflow could be ported to an empirical case through local data assembly, empirical calibration, repeated-seed stress testing, deterministic-control checking, and ex-post monitoring. The revised text makes clear that the current paper demonstrates the workflow path rather than a completed real-world deployment.
  Manuscript changes: Policy implications; Appendix calibration status.

- `R1-Q3`
  Comment: Is it a good recommendation to encourage car services in congested cities?
  Response: We added explicit caution that the observed car-service allocations are scenario-specific and should not be generalized to highly congested cities without local network and emissions testing.
  Manuscript changes: Policy implications.

- `R1-Q4`
  Comment: Can the paper give precise numbers for how many policies were explored, in what time, and with what computational power?
  Response: We now report the FPS set, dimensionality, initial samples, objective-specific BO updates, and the implied evaluation counts in the appendix. We also report the recoverable PBS job-resource requests from the preserved scripts and the measured walltimes that are actually present in preserved scheduler outputs: about 1 h 48 min for one-FPS deterministic jobs, about 1 h 50 min to 2 h 14 min for fixed-policy deterministic sweeps, and about 6 h 34 min to 9 h 16 min for representative total-system seed-array jobs on 10 CPUs. We still avoid inventing a full end-to-end runtime where preserved logs are incomplete.
  Manuscript changes: Methods; Appendix methodology.

- `R1-Q5`
  Comment: Are commuter-agent attributes internally consistent, for example age and disability?
  Response: We clarified the implemented commuter attributes, added the missing income attribute, documented the audited marginal weights for health, payment, disability, and technology access, and stated how age, disability, health, and tech access are used in the penalty rules. We also state explicitly that the current population is stylised and not jointly calibrated.
  Manuscript changes: Appendix agent description; parameter table; calibration text.

### Format and Formulation Issues

- `R1-F1`
  Response: Rewrote the flagged introduction sentence and citation phrasing.

- `R1-F2`
  Response: Removed the typo `reugular` during the introduction rewrite.

- `R1-F3`
  Response: Replaced the awkward sentence describing the framework in the introduction.

- `R1-F4`
  Response: Rewrote the literature-review sentence containing `particularly focus`.

- `R1-F5`
  Response: Corrected subject-verb agreement in the activity-based-models subsection.

- `R1-F6`
  Response: Rewrote the phrase `Despite of the advantages`.

- `R1-F7`
  Response: Standardized the term `activity-based models`.

- `R1-F8`
  Response: Removed the typo `chalenges`.

- `R1-F9`
  Response: Removed the punctuation issue in the rewritten paragraph.

- `R1-F10`
  Response: MATSim is now described as a framework.

- `R1-F11`
  Response: Corrected the grammar in the POLARIS sentence.

- `R1-F12`
  Response: Rewrote the repeated phrase about similar challenges.

- `R1-F13`
  Response: Rephrased the sentence to distinguish evaluation from optimisation.

- `R1-F14`
  Response: Standardized `activity-based` hyphenation.

- `R1-F15`
  Response: Clarified in the appendix that `E` contains exogenous environment variables and that realized demand is endogenous.

- `R1-F16`
  Response: We revised the last row of Table 2 directly so the first cell now reads `ABM-ETOP (this study)` and the row formatting is consistent with the rest of the table.

- `R1-F17`
  Response: Removed the typo `qeuation` during the metric rewrite.

- `R1-F18`
  Response: Removed the typo `simulaiton` during the simulation-process rewrite.

- `R1-F19`
  Response: Revised Figure 3's caption so it no longer misstates the shown grid as the optimisation grid.

- `R1-F20`
  Response: Added explicit explanation that the 12 dimensions come from 3 income groups crossed with 4 mode classes.

- `R1-F21`
  Response: Rewrote the opening sentence of the results section.

- `R1-F22`
  Response: Removed the legacy sentence containing `highlighted inequity`.

- `R1-F23`
  Response: Removed the `deomstrated` typo in the rewritten total-system section.

- `R1-F24`
  Response: Removed the unsupported net-emissions statement from the active manuscript.

- `R1-F25`
  Response: Replaced the implementation section and removed the flagged sentence.

- `R1-F26`
  Response: The retained figure blocks are now in consistent numerical order after the results rewrite.

- `R1-F27`
  Response: We regenerated Figure 1 as a manuscript-local schematic and updated the caption so the visual encoding and text now agree: darker cells and larger circles indicate higher illustrative allocation percentages.

- `R1-F28`
  Response: Revised Figure 4's caption and surrounding text to explain that the trace is illustrative and provider-specific.

- `R1-F29`
  Response: Figure 25 now points to the regenerated cross-policy output with improved readability.

- `R1-F30`
  Response: Added the missing income attribute to the commuter-agent description in the appendix.
