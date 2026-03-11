# Editorial Decision

The editor invited a major revision and flagged the resubmission as high risk. The revised manuscript is due by **April 14, 2026**.

> I have completed my evaluation of your manuscript. One of the reviewers suggested rejection while the other suggested major revision. Since one of the Reviewers recommended reconsideration of your manuscript following revision and modification, I invite you to resubmit your manuscript after addressing the comments below. But consider this as a 'high risk' resubmission. Please resubmit your revised manuscript by Apr 14, 2026.
>
> When revising your manuscript, please consider all issues mentioned in the reviewers' comments carefully: please outline every change made in response to their comments and provide suitable rebuttals for any comments not addressed. Please note that your revised submission may need to be re-reviewed.

# Review Summary

The article presents a framework to evaluate subsidy allocation policies regarding equity and efficiency. The framework consists of:
1. an agent-based model to represent commuters with various income ranges and service provider behaviours
2. Bayesian optimisation to explore scenarios and identify optimal policies
3. three evaluation metrics: mode share equity, travel time equity, and total system travel time

After a short literature review, the paper presents the framework components and applies the framework to explore various fixed pool subsidy budgets on a fictional simplified scenario. The review notes that many results are presented regarding mode share equity, travel time equity, total system travel time, and related optimal subsidy allocations. The review also notes that an optimal subsidy level is identified that improves all three metrics, and that a progressive subsidy structure in income groups is discussed. Appendices detail the agents' algorithms, parameters, optimisation methodology, calibration, and benefits calculation.

# Strengths

- The article is relevant to the journal.
- The article is clearly written and organised.
- The combination of ABM simulation and Bayesian optimization to identify the best policy is an interesting architecture.
- The results are well presented and clearly explained.
- The limitations are mentioned at the end of the paper.
- A lot of essential information is in Appendix.

# Weaknesses

- The work only explores progressive allocation by giving bounds to the allocation matrix and this is not clearly stated. The boundary constraints used for the allocation matrix are inputs to the model and force a progressive allocation, while the paper formulation makes it appear like an emerging result from the policy optimization. The reviewer specifically flagged the statement: “Our findings demonstrate that progressive strategies, specifically directing 55% to 65% of subsidies towards low-income populations, not only maximise equity but also, counter-intuitively, enhance system-wide efficiency.” The reviewer requests comparison with non-progressive subsidy distributions and asks for hypotheses and literature support for the claim that equity and efficiency are not inherently in conflict.
- The ABM design and implementation must be clearly described and validated before drawing conclusions. The reviewer asks whether the model was presented in previous work and whether an open-source implementation can be provided for reproducibility.
- The model only considers shared cars, shared bikes, and public transport. The reviewer asks why private cars and private bikes are not considered.
- The mode share equity metric uses weights and scaling to `[0,1]`, but the travel time equity metric does not. The reviewer asks whether the metrics can be unified in scale and asks for metric units.
- The simulation setup is not sufficiently clear. The reviewer asks why `Niter = 144`, what the 10 initial allocation matrices are, how the fictional environment was defined, whether other environments or income distributions were tested, and how many repetitions were done given the stochasticity in trip generation.
- References and sources are missing for many parameter values and assumptions, including value of time, congestion parameters, sensitivity coefficients, network parameters, benefits assumptions, and Australian data used for monetised benefits.
- The description of Algorithm 1 is incomplete. The reviewer asks for clearer definitions of the vectors `S`, `P`, `A`, and `E`, and asks at what time scale prices and BPR-based routing updates occur.
- The optimisation framing is unclear. The reviewer asks why multi-objective optimisation was not used and whether the current optimisation is mono-objective or multi-objective.
- Table 6 baseline modal share distributions by income group appear unrealistic and do not sum to 100%. The reviewer asks for an explanation.
- The reviewer asks for explanation of why optimal subsidy allocation pushes low-income groups toward shared bikes, shared cars, and public transport while pushing high-income groups away from cars, and asks how trip counts vary with subsidies.
- The reviewer asks whether the reported subsidy use by income group simply reflects group sizes.
- The reviewer states that subsidy utilisation and utilisation rates are not clearly explained and asks why some subsidies are not used.
- The reviewer asks whether the recommended 54-72% range is consistent with previous findings and whether deliberately uneven intervention has precedent in prior research.
- The reviewer asks for more discussion of the fact that equity can improve either by improving disadvantaged users or by worsening outcomes for others.

# Overstatements

- The title is too broad because the article evaluates subsidy allocation policies only. The reviewer suggested rephrasing it to: “ABM-ETOP: An Agent-Based Framework for Evaluating Multimodal Subsidy Allocation Policies for Equity and Efficiency”.
- The paper claims ABM-ETOP overcomes limitations of four-step models, activity-based models, and other ABMs “as validated through comparisons”, but the article does not actually present those comparisons.
- The paper appears to equate horizontal equity with total system travel time or system-wide efficiency without sufficient justification.
- The paper claims to address the implementation gap in transport policy, but the reviewer argues that the scenario is too simplified to support that claim.

# Questions

- How does the framework transfer to city scale?
- Is the approach applicable to real-world cases and how?
- Is it a good recommendation to encourage car services in cities with severe congestion?
- Can the paper give precise numbers for how many policies were explored, in what time, and with what computational power?
- Are commuter-agent attributes internally consistent, for example age and disability?

# Format and Formulation Issues

- Page 1: “(Martens, 2016) points out […]”
- Page 1: “[…] but reugular planning tools […]”
- Page 1: “The framework employs the heterogeneous advantages of the original designed flexible agent-based model […]”
- Page 2: “[…] particularly focus on FPS and PBS” should be “focused”.
- Page 2: “Activity-based models is a powerful […] tool.” should be revised.
- Page 2: “Despite of the advantages, there […]” should be “Despite the advantages”.
- Page 2: “[…] the activity models […]” should be “activity-based models”.
- Page 2: “The chalenges […]”
- Page 3: remove the extra space before the comma in “computational intensiveness ,”.
- Page 3: “MATSim is an open-source activity-based model” should say framework.
- Page 3: “The lack of detailed provider response modelling mean […]” should be “means”.
- Page 3: “[…] They also face similar computational and data challenges similar to activity-based models, […]”
- Page 3: “[…] typically lack the specific optimization capabilities required for evaluating complex subsidy structures (Zhao and Li, 2016). […]” should be rephrased because evaluation and optimization are different.
- Page 3: “[…] activity based models […]” should be “activity-based models”.
- Page 4: The definition of environment parameters `E` should clarify what is environment versus emerging demand.
- Page 6: see first cell of the last row of Table 2.
- Page 6: “In the qeuation […]”
- Page 7: “[…] current simulaiton […]”
- Figure 3: the spatial grid is described as 85x85 but only a 55x55 grid is shown.
- Page 9: “[…] 12-dimensional space of subsidy percentages across income groups and transport modes.” needs explanation.
- Page 9: “[…] we focused on examining […]” should start with “We”.
- Page 10: “[…] highlighted inequity across transport modes. […]” should likely be “highlighting”.
- Page 12: “Figure 20 deomstrated […]”
- Page 15: “The programme results in a net increase of 61,564 tonnes of CO2 annually.” was flagged.
- Page 15: “This ensures not only technical capability but also broad community acceptance Quay (2010), translating evidence-based findings into [...]”
- Figure ordering is incorrect for Figures 15/16, 19/20, and 26/27/28.
- Figure 1: “Circle size & cell color = allocation percentage” but the cells are all green.
- Figure 4: BikeShare1 shows a peak around timestep 60 but a very small price impact.
- Figure 25: the FPS text at the bottom right is hard to read.
- Appendix A1.1.1: the agents should have an income attribute that is missing.

# Revision Context

The current codebase now includes:
- seed-based optimization and merged summaries with means, standard deviations, and 95% confidence intervals
- deterministic workflows with fixed commuters, trips, and background traffic across FPS values
- fixed-policy seed evaluation
- Katana / PBS scripts for full sweeps and per-seed evaluation
- richer visualisations, sensitivity analysis, and cross-policy comparison outputs

These additions should be used as the evidence base for the revision rather than the older single-run framing.
