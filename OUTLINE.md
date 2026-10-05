# Working title
Context as a Cellular Automaton: A Survey of Local-Rule Memory and Context Management for LLM Agents

# Thesis
Most long-term memory / context-management systems for LLM agents can be read as
discrete-time local-rule dynamics on a graph of memory units: a cell state, a
neighborhood, a transition rule, an update schedule, a budgeted readout, and
(rarely) a training signal. Making this reading explicit (i) unifies atomic,
graph and profile memory; (ii) exposes that the transition rule is usually a
hand-set threshold on embeddings or a full LLM call; (iii) points to small
non-generative decision models as a missing middle for the rule; (iv) turns
per-hop evidence annotations into trajectory supervision.

# Sections
1 Introduction
2 Background (CA/NCA/GNCA; spreading activation & attractor memory; LLM agent memory & benchmarks)
3 Unifying formalism: Context-CA = (V,E,S,f,schedule,readout,budget); worked mappings (ProGraph Stage 2 = threshold automaton; HippoRAG PPR = linear diffusion; A-Mem = developmental edge growth; KV eviction = token-grid CA; ...) + Table 1
4 Transition functions: thresholds/cosine; linear diffusion; learned message passing; LLM-as-rule; small non-generative decision models (rerankers, NLI verifiers, typed decision models) + Table 2 cost per cell-step
5 State design: profile/residual decomposition; continuous activation & NCA residual update; decay/forgetting/admission/eviction; contradiction as state transition
6 Schedule, budget, readout: sync/async; hop budget; fan-out & lateral inhibition; token-budgeted readout; write-path excitation gating and the O(T^2) rewrite problem
7 Training signals & evaluation: per-hop evidence as trajectory supervision; frozen-LLM readout; same-FLOPs one-shot baseline; judge pitfalls
8 Relation to latent multi-agent communication (cells-as-agents spectrum; channel width; stability; EC critique)
9 Open problems & experimental agenda
10 Conclusion
