# Overleaf Upload and Build Checklist

This checklist distinguishes completed local compile verification from the final upload and PDF QA steps.

## Build-Readiness Note

- Local content review is complete for the manuscript, appendix, figure paths, captions, bibliography links, and review package.
- Local compilation was completed successfully on **March 11, 2026** with:
  - `source /etc/profile.d/modules.sh`
  - `module load texlive/20220321`
  - `cd /home/z5247491/MaaS-Simulation/paper/tex`
  - `latexmk -pdf -bibtex main.tex`
- The standalone paper package now bundles `cas-sc.cls`, `cas-common.sty`, and `cas-model2-names.bst` in `paper/tex`.
- The current `main.log` shows no missing references or bibliography failures. Remaining issues are minor layout warnings, mainly:
  - one title-block overfull box produced at `\maketitle`
  - a small overfull line in the compact Algorithm 1 table
  - a compact appendix-algorithm overfull line
  - several underfull boxes in narrow table cells

## Recommended Upload Structure

1. Upload the **contents** of `/home/z5247491/MaaS-Simulation/paper/` to the Overleaf project root.
2. Preserve this directory structure:
   - `tex/main.tex`
   - `tex/appendix.tex`
   - `tex/references.bib`
   - `tex/cas-sc.cls`
   - `tex/cas-common.sty`
   - `tex/cas-model2-names.bst`
   - `Images/`
   - `policy_figures/`
3. In Overleaf, set `tex/main.tex` as the main file.

## Compile Steps

4. Keep the relative paths unchanged. `tex/main.tex` expects:
   - figures in `../Images/`
   - policy figures in `../policy_figures/`
   - bibliography file `references.bib` in the same `tex/` folder
   - appendix file `appendix.tex` in the same `tex/` folder
5. Compile with `pdfLaTeX` + `BibTeX`, or use Overleaf `latexmk` with BibTeX enabled.
6. If compiling manually, use this sequence:
   - `pdflatex tex/main.tex`
   - `bibtex tex/main`
   - `pdflatex tex/main.tex`
   - `pdflatex tex/main.tex`

## Verification Steps

7. Confirm that all figures render without external-path errors. The manuscript should not reference `../../MaaS-Centralised/...` or `../../cross_policy_analysis_results/...`.
8. Confirm that the bibliography resolves under `cas-model2-names`.
9. Confirm that the appendix loads through `\input{appendix}` and that appendix pagination resets correctly.
10. Inspect Figure 1, Figures 7--29, Table 2, and the appendix tables in the compiled PDF for final layout quality.
11. Inspect the title block, Algorithm 1, the appendix service-update algorithm, and the narrow implementation tables if you want to eliminate the remaining minor box warnings before final submission.
12. If the journal or Overleaf template ships different Elsevier support files, re-run the PDF check to ensure there are no class-specific table or float regressions.

## Expected Outcome

- The paper package is structured to compile as a standalone project and has already compiled successfully in the local environment above.
- If compilation fails after upload, the next debugging target is the destination TeX environment or a changed template layer rather than missing manuscript figures or broken external paths.
