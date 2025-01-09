---
title: "SELF-IMPROVEMENT IN LANGUAGE MODELS: THE SHARPENING MECHANISM"
subject: self-improvement
license: CC-BY-4.0
date: 2024-12-31
authors:
  - name: Ziyuan Nan
    email: nanziyuan21s@ict.ac.cn
    affiliation: ICT CAS
---

## Motivation

A dominant hypothesis for why improvement without external feedback might be possible
is that models contain "hidden knowledge". But how to extract it?

The starting point is the widely observed phenomenon that langauge models are often
better at verifying whether responses are correct than they at generating correct responses.
So self-improvement can be viewed as any attempt to narrow this gap, i.e., use the model as
its own verifier to improve generation and *sharpen* the model toward high-quality responses.

Formmaly, consider a base model $\pi_{base}: \mathcal{X} \rightarrow \triangle(\mathcal{Y})$
representing a conditional distribution that maps a prompt $x \in \mathcal{X}$ to a
distribution over responses. $\pi_{base}$'s jet feature is that it is a good verifier, as
measured by some *self-reward* function $r_{self}(y |x; \pi_{base})$ measuring model
certainty, e.g. regularized sequence likelihood, llm-as-judges, model confidence.

**Sharpening** is any process that tilts $\pi_{base}$ toward responses that are more
certain. A sharpened model 
$\hat{\pi} \approx  \mathop{\arg\max}\limits_{y \in \mathcal{Y}} r_{self}(y|x; \pi_{base})$.

Sharpening may be implemented at inference-time, e.g. decoding strategies, or self-training.
Though it is unclear a-prior whether there are self-rewards related to task performance.
But many previous work suggest that models do have hidden knowledge, but it is 
computationally challenging to extract. Self-improvement algorithms leverage these
verifications to improve the quaility of generations.

