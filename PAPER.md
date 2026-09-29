# Reward Doesn't Prove Communication: A Multi-Level Verification Study of Emergent Communication in Multi-Agent RL

*A short technical report accompanying [github.com/ara-5/multi-agent-system](https://github.com/ara-5/multi-agent-system). All results are reproducible from the scripts and raw per-seed data committed in that repository; see the Reproducibility appendix.*

## Abstract

A rising training-reward curve is routinely treated as evidence that multi-agent reinforcement learning (MARL) agents have learned to communicate. We show, in a minimal three-symbol Lewis signaling game (PettingZoo's `simple_speaker_listener`), that this inference fails at three separate levels, each catching a different confound the previous one misses. First, reward improves even when a speaker's message carries essentially no information about the task-relevant variable it is supposed to encode — a shared-network architecture reaches decent reward while message&harr;target mutual information (MI) stays at correlational-noise levels (~1%), because nothing in that architecture forces target information to route through the message at all. Second, once MI is high, a widely-used causal-intervention audit (does forcing a different message change the receiver's action?) is itself confounded: without a placebo control, 33–62% of the "causal" signal in every condition we tested is just generic policy brittleness to any input perturbation, not message-specific reliance. Third, after placebo-correcting the causal metric, we find a failure mode absent from prior taxonomies: architectures whose message is causally load-bearing (the receiver's action really does depend on it, above placebo, p<0.05) while carrying almost no information about the target (~1% MI) — a channel that is *used* without being *informative*. We further show that (a) different informativeness-maximizing objectives produce protocols with measurably different robustness profiles under channel noise, independent of their causal-signal strength, and (b) the movement policy generalizes gracefully (retaining ~90% of its advantage over a random baseline) to spatial configurations 2&times; wider than the training distribution. Finally, we run the identical real/causal-swap/placebo audit design against two independent LLM agents applying an explicit symbolic protocol, and find a qualitatively different result (0% placebo-induced errors vs. 33–62% for the RL policies) — suggesting the placebo confound is not a fixed property of causal-intervention audits but depends on whether the audited system has explicit, inspectable structure to lean on. All experiments use bootstrap confidence intervals over 4–6 random seeds per condition; we report every negative and inconclusive result alongside the positive ones.

## 1. Introduction

Emergent communication research asks whether agents trained jointly, with no predefined language, develop a communication protocol that supports task performance. The standard evidence offered for "yes" is a reward curve that goes up when agents are allowed to communicate, sometimes accompanied by a correlational statistic (mutual information, or a confusion matrix between messages and meanings) computed after training. Lowe et al. (2019) argue this standard of evidence is inadequate — reward and even correlational metrics can be satisfied by shortcuts that have nothing to do with the message channel actually being used meaningfully.

This report takes that warning seriously and asks a narrower, more mechanical question: in a small, fully ground-truth-labeled task where we can *check* every claim directly, at how many distinct levels does "the reward/metric went up" fail to imply "real communication happened," and what does it take to catch each one? We find three such levels, not one, and each requires building a different kind of check:

1. **Reward vs. correlational informativeness.** A policy can improve at the task without its message carrying information about the variable it is meant to encode, if the network architecture gives the receiver another path to the same information.
2. **Correlational informativeness vs. causal reliance.** A message can be highly informative about the target (high MI) while the receiver's action does not actually depend on it, or — the inverse trap this report identifies as also live in practice — a causal-intervention audit built to catch that can itself be confounded by an unrelated failure mode (generic brittleness), without a placebo control.
3. **Causal reliance vs. task-relevant meaning.** Even a causally load-bearing channel — the receiver's action genuinely changes when the message changes — is not guaranteed to be carrying information about the actual task variable it is supposed to communicate.

We build architectural interventions, statistical auxiliary losses, and audit tools to make each of these three gaps visible and measurable, run everything across multiple random seeds with bootstrap confidence intervals (Colas et al., 2018), and extend the same verification logic to two further questions that matter for any communication protocol claimed to be real: does it survive a noisy channel, and does it generalize outside the training distribution? We close with a small pilot pointing the identical audit methodology at LLM agents instead of a trained RL policy, motivated by the parallel, higher-stakes question of whether multi-agent LLM systems might communicate in ways that evade an overseer (Motwani et al., 2024).

## 2. Related Work

Our central methodological claim — that reward and even correlational MI can overstate real communication — restates the thesis of **Lowe et al., "On the Pitfalls of Measuring Emergent Communication" (AAMAS 2019, [arXiv:1903.05168](https://arxiv.org/abs/1903.05168))**. We do not claim to be first to notice this; our contribution is a concrete, multi-seed, ground-truth-labeled demonstration of *how many separate levels* the problem recurs at, including one (Section 5.2's placebo-controlled causal audit) that we believe has not been explicitly tested for in prior emergent-communication work.

**Objectives that push message informativeness.** Eccles et al. ("Biases for Emergent Communication in Multi-agent RL," NeurIPS 2019, [arXiv:1912.05676](https://arxiv.org/abs/1912.05676)) propose positive-signalling and positive-listening auxiliary losses; we implement both directly from their equations (their Eq. 3 and Eq. 8) as one of two independently-sourced objectives compared here. Rita et al. ("Emergent Communication: Generalization and Overfitting in Lewis Games," NeurIPS 2022, [arXiv:2209.15342](https://arxiv.org/abs/2209.15342)) decompose the Lewis-game objective into speaker-unambiguity and listener-optimality terms conceptually close to our second objective. Krause et al. (2010) and Hu et al. ("IMSAT," ICML 2017, [arXiv:1702.08720](https://arxiv.org/abs/1702.08720)) are the source of the RIM/IMSAT label-free mutual-information lower bound we repurpose as that second objective — an unsupervised-clustering technique never designed for RL or communication. Wang et al. ("Information Bottleneck Approach," ICML 2020, [arXiv:1911.06992](https://arxiv.org/abs/1911.06992)) use an information bottleneck in the opposite direction (compression, not informativeness) — useful context that "information bottleneck" does not uniquely identify one technique in this space.

**Causal auditing of communication.** Jaques et al. ("Social Influence as Intrinsic Motivation," ICML 2019, [arXiv:1810.08647](https://arxiv.org/abs/1810.08647)) introduce the counterfactual-intervention framework our causal audit builds on. Eccles et al. (2019, Eq. 4–8) adapt this into their positive-listening loss for a multi-timestep recurrent setting; we adapt the same equations to our single-step feedforward setting and, distinctly from prior work, add a placebo control (Section 5.2) that neither paper's causal metric includes.

**Statistical rigor for RL comparisons.** Colas et al. ("How Many Random Seeds?," [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)) and Henderson et al. ("Deep RL that Matters," AAAI 2018, [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)) motivate our use of bootstrap confidence intervals over single-seed anecdotes throughout.

**LLM-agent collusion detection.** Motwani et al. ("Secret Collusion among AI Agents," NeurIPS 2024; "Hidden in Plain Text," [arXiv:2410.03768](https://arxiv.org/abs/2410.03768)) and "Detecting Multi-Agent Collusion Through Multi-Agent Interpretability" ([arXiv:2604.01151](https://arxiv.org/abs/2604.01151)) study whether LLM agents can communicate in ways that evade an overseer. We do not contribute to that literature directly — our LLM pilot (Section 6.7) is a cooperative, non-adversarial task — but we position our RL-policy findings as a fully-controllable, ground-truth-labeled testbed for the *verification methodology* that problem needs, and take one small step toward checking whether that methodology transfers to LLM agents at all.

**What we believe is not already covered.** None of the papers above run a channel-informativeness loss and a causal-reliance audit on the *same* trained models to check whether one can be inflated without the other (Section 6.1–6.2), and none report a placebo-controlled causal-intervention audit for emergent communication (Section 6.2), which is where we find our most novel result (Section 6.2(b)).

## 3. Environment and Task

We use PettingZoo's MPE2 `simple_speaker_listener`: a Speaker agent observes a one-hot (or, in the stock environment, an RGB-color-encoded) target landmark identity and can only emit a discrete message; a Listener agent observes its own velocity, all landmarks' relative positions, and the received message, and must move to the target landmark. Reward is a shared, dense, distance-based signal. This is a fully cooperative Lewis signaling game with full ground-truth access to the target, which is what makes it possible to check every metric against the truth before trusting it anywhere else.

We additionally built `ScaledScenario` (a modified scenario replacing the stock environment's 3-message-only, RGB-color-encoded target representation with a true N-dimensional one-hot) to test vocabulary sizes beyond 3 and to parameterize the environment's reset position range for the generalization experiment (Section 6.6).

## 4. Architectures and Training Objectives

**Architectures.** (i) `shared_trunk`: a single network (SB3's default `ActorCriticPolicy`) over the concatenated joint observation, with message and movement logits read off different columns of the same final layer. (ii) `bottleneck`: two independent sub-networks with no shared layer — message logits computed only from the speaker's observation slice, movement logits only from the listener's — implemented by overriding SB3's `ActorCriticPolicy._build` to replace its default cross-latent linear layer with `nn.Identity()`, since that layer would otherwise silently recreate a shortcut between the two halves. The value function remains centralized (a shared critic with decentralized actors, standard practice).

**Objectives**, applied on top of either architecture: (i) `none` — standard PPO only; (ii) `entropy` — a generic entropy bonus (`ent_coef=0.01`) on the full joint action distribution; (iii) `mi` — an auxiliary loss maximizing the RIM/IMSAT label-free lower bound on message&harr;target mutual information, computed only from the speaker's message logits: I(x;y) &ge; H(mean<sub>i</sub> p(y|x<sub>i</sub>)) &minus; mean<sub>i</sub> H(p(y|x<sub>i</sub>)); (iv) `eccles` — Eccles et al.'s positive-signalling and positive-listening losses (Eq. 3, Eq. 8), implemented from their published equations.

All models train with Stable-Baselines3 PPO. Section 6 results use 6 seeds/condition at 120k timesteps (a reduced budget from the 300k used for single-condition demonstrations elsewhere in the repository, to fit a same-day compute window on a 12-core CPU machine — see Limitations).

## 5. Verification Methodology

**5.1 Correlational mutual information.** After training, we run held-out episodes, record (true target, sent message) pairs, and compute MI from the resulting confusion matrix, reported as a percentage of the theoretical maximum (log&#8322; of the vocabulary size).

**5.2 Causal intervention with a placebo control.** For each held-out step, we recompute the listener's movement logits with the received-message slice of its observation replaced by every alternative message value, holding everything else fixed (`movement_logits_with_message_override`), and report the fraction of steps where this changes the greedy action (`argmax_flip_rate`). Critically, we run the *identical* intervention procedure on an unrelated slice of the listener's own observation (its own velocity) as a placebo control, and report the **placebo-corrected signal**: message flip-rate minus placebo flip-rate. Without this control, a high raw flip-rate is consistent with two very different explanations — real message-specific reliance, or a policy that is simply brittle to any sufficiently large input perturbation — and our own results (Section 6.2) show this is not a hypothetical concern. We additionally report two refinements: an **action-distance signal** (the mean L1 distance between the real-message and each alternative's movement distribution, message-condition minus placebo-condition, run through the identical comparison as the binary flip rate — a continuous effect size that can disagree with the binary metric when an effect is driven by a few large flips rather than a uniformly larger distributional shift) and a **target-conditioned flip-rate breakdown** (the same flip rate computed separately per true target, to check whether a condition's placebo-corrected signal is broadly shared across targets or concentrated in one).

**5.3 Noise robustness.** We corrupt the speaker's transmitted message independently at each environment step with probability *p* (replacing it with a uniformly random different message before the environment advances — exactly what a lossy channel does to a transmitted symbol), for *p* &isin; {0, 5, 10, 20, 30}%. We report episode reward and `decision_flip_rate`: of the steps a corruption event occurred on, the fraction where it actually changed the listener's greedy action (reusing 5.2's override hook to compare "message as intended" vs. "message as corrupted").

**5.4 Spatial generalization.** The message protocol here is scale-invariant by construction (the speaker's observation is a landmark-identity one-hot, not a position), so only the listener's movement policy can generalize or fail. We widen the environment's reset position range up to 2&times; the training distribution and report reward *relative to a random-policy baseline evaluated at the same scale*, since raw distance-based reward is mechanically larger in magnitude at wider scales for any policy, trained or not.

**5.5 LLM-agent pilot.** We run the identical real/causal-swap/placebo design (Section 5.2's logic, applied at the level of full agent behavior rather than logits) against two independent Claude Haiku subagents (Speaker, Listener) playing an explicit three-symbol codebook game, to test whether the same audit methodology, and its need for a placebo control, transfers to a qualitatively different kind of agent.

## 6. Experiments and Results

### 6.1 Correlational MI: the architecture is doing essentially all the work

| Architecture | Objective | MI (% of max) | 95% CI |
| --- | --- | --- | --- |
| bottleneck | none | 45.4% | [19.0, 72.3] |
| bottleneck | entropy | 38.8% | [19.0, 58.6] |
| bottleneck | mi | **99.7%** | [99.7, 99.7] |
| bottleneck | eccles | 86.0% | [72.3, 99.7] |
| shared_trunk | none | **1.8%** | [0.9, 2.7] |
| shared_trunk | mi | **1.1%** | [0.4, 2.1] |
| shared_trunk | eccles | **1.0%** | [0.5, 1.6] |

We predicted a channel-only auxiliary loss might inflate correlational MI even without the architectural bottleneck, by satisfying the loss on the speaker's side alone while the listener keeps reading the target shortcut directly. That is not what happened: on `shared_trunk`, both auxiliary losses leave MI at ~1%, statistically indistinguishable from random and a 98.6-point gap from the same loss on `bottleneck` (p<0.0001, n=6 vs. n=6). The architecture is not only necessary for the listener to causally depend on the message — it is necessary for the loss to move the correlational metric at all, likely because gradient from the much larger policy-gradient and value losses dominates the tiny message-column readout before a small-coefficient (0.1–0.2) auxiliary loss can shape it. The `shared_trunk/none` row — a plain shared network with no communication-specific loss at all — completes the direct four-way comparison (shared network / bottleneck / bottleneck+entropy / bottleneck+MI) most literally: it sits at MI indistinguishable from `shared_trunk`'s other two conditions, confirming the architecture alone determines the correlational-MI ceiling regardless of which objective, if any, sits on top of it.

### 6.2 Causal audit: a metric measuring "causal reliance" needs its own placebo control

A first pass using the raw argmax-flip-rate as evidence of causal reliance is dangerously easy to over-read: forcing an alternative value into the placebo slice (the listener's own velocity) produces a comparably high flip rate in every condition (33–62%). The trustworthy quantity is the placebo-corrected message-specific signal:

| Architecture | Objective | Signal (pp) | 95% CI | Distinguishable from placebo? |
| --- | --- | --- | --- | --- |
| bottleneck | none | 6.7 | [-2.2, 19.3] | No |
| bottleneck | entropy | 8.7 | [0.4, 17.7] | Barely |
| bottleneck | mi | 8.8 | [3.2, 16.2] | Yes |
| bottleneck | eccles | 29.9 | [9.9, 47.3] | Yes |
| shared_trunk | none | 16.3 | [-2.2, 37.1] | No |
| shared_trunk | mi | 20.3 | [12.7, 27.8] | Yes |
| shared_trunk | eccles | 43.9 | [26.1, 63.7] | Yes |

Three results follow. **(a)** Eccles' loss produces the strongest placebo-corrected causal signal of any bottleneck condition (29.9pp), consistent with its positive-listening term directly optimizing this exact quantity, unlike the RIM/IMSAT loss, which only optimizes the speaker's side — a mechanistically-explicable result that also validates the placebo-corrected metric is measuring something real.

**(b) `shared_trunk` shows a real, statistically significant causal signal — larger than most bottleneck conditions — despite ~1% MI.** This is a third failure mode beyond what motivated this ablation: not "looks informative but isn't causal" (which the data above rejects as the dominant story) but **causally load-bearing without being informative**. The listener's action measurably depends on the message-slice override more than on a placebo perturbation, even though that message carries essentially no information about the actual target — most likely reflecting the shared trunk's fundamental entanglement (its intervention reruns the *entire* joint network, not an isolated sub-network, so "causal effect of the message" and "generic sensitivity of an entangled network to one of its inputs" are harder to cleanly separate; see Limitations) rather than anything resembling a communication protocol.

**(c) That failure mode is specifically tied to an auxiliary loss acting on the channel, not to the `shared_trunk` architecture alone**: `shared_trunk/none`'s signal (16.3pp) is *not* distinguishable from placebo, unlike `shared_trunk/mi` and `shared_trunk/eccles`. The point estimate sits between the two significant conditions and 6 seeds cannot rule out a real effect here either, but as measured, this is consistent with (b)'s mechanism — an auxiliary loss's gradient reaching the message-adjacent columns enough to entangle them further with the movement columns, without reaching far enough to make the message itself informative (6.1).

**Two refinements corroborate and complicate (a)–(c).** The action-distance signal (Section 5.2) agrees closely with the binary flip-rate signal wherever the latter is large — `bottleneck/eccles` (0.540, 95% CI [0.252, 0.881]), `shared_trunk/eccles` (0.620, [0.397, 0.929]), `shared_trunk/mi` (0.169, [0.110, 0.238]) are all clearly above placebo on both metrics — but `bottleneck/mi`'s two metrics disagree: its flip-rate signal is significant (8.8pp) while its action-distance signal is not (0.075, 95% CI [-0.024, 0.213], includes 0). We read this as a power difference, not a contradiction: the L1 metric averages over every alternative message per step, diluting an effect driven by only one or two close-call flips, while the flip-rate metric only needs one alternative to flip. Target-conditioned heterogeneity (the flip-rate's max-min range across the three true targets) is modest for every condition (mean 3.4–12.8pp of a 0–100pp scale), indicating no condition's signal is concentrated in a single target.

### 6.3 Vocabulary scaling

| num_landmarks | Objective | MI (% of max) | p (vs. other objective) |
| --- | --- | --- | --- |
| 3 | entropy | 59.0% | p<0.0001 |
| 3 | mi | 99.7% | — |
| 5 | entropy | 72.5% | p<0.0001 |
| 5 | mi | 99.6% | — |
| 8 | entropy | 77.3% | p<0.0001 |
| 8 | mi | 97.5% | — |

The MI loss's correlational advantage is real and significant at every vocabulary size tested, but shrinks as vocabulary grows (40.7-point gap at 3 symbols &rarr; 20.2 points at 8) — a fix that looked complete at 3 symbols visibly loses ground as the task scales.

### 6.4 Passive in-training diagnostic

A free, zero-extra-rollout MI estimate logged during training correlates strongly with the expensive post-hoc ground truth *within* a fixed vocabulary size (r=0.999/0.901/0.865 at 3/5/8 landmarks), but much more weakly across the full, architecturally-diverse 36-run sweep (r=0.116), and its correlation with the causal signal is consistently near zero or negative. It is a good, free early-warning proxy for whether the architecture lets the loss move MI at all — not for whether that information is causally used, or (per 6.2b) whether a near-zero-MI channel might still be causally relied upon for unrelated reasons.

### 6.5 Noise robustness

Corrupting the transmitted message at rate *p* leaves reward almost unchanged for every condition (e.g. `bottleneck/mi`: &minus;19.01 at 0% &rarr; &minus;18.91 at 30%) — this task's dense, distance-based reward, integrated over 25 steps, absorbs a fraction of corrupted steps rather than compounding them into failure. The informative metric is `decision_flip_rate`:

| Condition | Decision flip-rate @ 30% noise (95% CI) |
| --- | --- |
| bottleneck / none | 50.3% [40.9, 58.8] |
| bottleneck / entropy | 48.6% [39.0, 59.6] |
| bottleneck / mi | 46.1% [38.9, 54.3] |
| bottleneck / eccles | 62.9% [52.2, 74.3] |
| shared_trunk / mi | 54.2% [47.9, 61.1] |
| shared_trunk / eccles | 58.8% [47.1, 73.7] |

Only one pairwise comparison reaches significance: `bottleneck/eccles` is reliably more decision-fragile under corruption than `bottleneck/mi` (diff=16.76pp, 95% CI=[2.97, 30.13], p=0.015) — despite `eccles` scoring the *highest* placebo-corrected causal signal in 6.2 and `mi` scoring the highest raw MI. The two objectives produce protocols with different failure profiles rather than one dominating: `eccles`'s protocol is more strongly, causally relied-upon when the channel is clean, but more brittle to per-instance corruption; `mi`'s protocol is comparatively more redundant under noise.

### 6.6 Spatial generalization

| Scale | mi: rel. improvement over random | entropy: rel. improvement over random |
| --- | --- | --- |
| 1.0&times; (training) | 60.5% [59.2, 61.4] | 58.8% [58.3, 59.4] |
| 1.5&times; | 59.7% [58.9, 60.1] | 58.8% [57.4, 59.7] |
| 2.0&times; | 52.4% [51.3, 53.0] | 52.1% [50.6, 53.0] |

Both conditions retain roughly 90% of their relative advantage at 2&times; the training range — graceful degradation, not collapse. The two objectives are statistically indistinguishable at 1.5&times;/2.0&times; scale (p=0.19, p=0.76), consistent with the message itself carrying no positional information to generalize in the first place; the movement policy's spatial generalization is essentially independent of which communication objective shaped the channel.

### 6.7 LLM-agent pilot: does the same audit methodology transfer?

Running the identical real/causal-swap/placebo design against two independent Claude Haiku subagents applying an explicit codebook (n=3 trials per condition) produced a clean 9/9 result: every causal-swap trial followed the swapped code rather than the (inaccessible) true target, and 0/9 placebo trials (an irrelevant sentence appended to the message) changed the answer — a sharper separation than any RL condition above, whose placebo flip rates ran 33–62%. The likely explanation is not that LLMs communicate "better" in some deep sense, but a substrate difference: an opaque MLP shaped by optimization pressure can be pushed across a decision boundary by any sufficiently large perturbation, while an LLM applying an explicitly stated symbolic rule can trivially distinguish "the code" from an unrelated sentence. **A causal-intervention audit's need for a placebo control is not a fixed property of the method — it depends on whether the audited system has explicit, inspectable structure to lean on.** This is a small, explicitly-scoped pilot (n=3/condition, one model, a cooperative, non-adversarial task) — see `llm_agent_audit_pilot.md` for full transcripts and limitations before citing it further.

## 7. Discussion: The Throughline

Every result in this report is an instance of the same question asked at a different level: *does this signal mean what it looks like it means, or could it be a correlate or artifact instead?* Reward can rise without the message being informative (6.1). A correlational metric can be high without the receiver's action causally depending on it — and the tool built to check *that* can itself be confounded without its own control (6.2). A channel can be causally load-bearing without being informative about the task (6.2b) — a genuine third category, not reducible to either "worked" or "didn't." Two objectives can produce indistinguishable causal-reliance numbers while differing significantly in noise robustness (6.5), and in a spatial-scale generalization test where the tested quantity (movement policy) is fully separable from the communication objective's effect, the two objectives turn out indistinguishable (6.6) — a useful negative result, since it rules out the objective as an explanation for movement-generalization differences we might otherwise have been tempted to over-interpret. And the placebo confound that dominates the RL results essentially vanishes for LLM agents applying an explicit protocol (6.7), suggesting the confound is a property of *opaque* function approximators specifically, not of causal-intervention audits in general. None of these findings required a new algorithm; each required building a small, specific check for a way the previous check could be wrong, and running it enough times (multiple seeds, bootstrap CIs) to trust the answer.

## 8. Limitations

- **Seed budget.** 6 seeds and a 120k-step training budget (vs. 300k used for single-condition demonstrations elsewhere in the repository) were chosen to fit a same-day compute window on a single 12-core CPU machine, not because 6 is an adequate seed count for the statistical-power standard Colas et al. (2018) argue for. Every p-value and CI in Section 6 should be read as suggestive, not conclusive, until re-run at a larger budget.
- **Causal-audit comparability across architectures.** `bottleneck`'s intervention isolates a genuinely separate listener sub-network; `shared_trunk`'s reruns the whole entangled joint network, so its causal-signal number answers a subtly different question than `bottleneck`'s. Section 6.2(b)'s finding should be read as "a real, placebo-controlled effect exists," not as a directly architecture-comparable effect size.
- **Placebo choice.** The listener's own velocity (zeroed or negated) is one reasonable placebo control, not a validated calibration; a magnitude-matched or margin-aware version (reporting logit-gap changes rather than only whether argmax flips) is a natural refinement.
- **Noise-robustness reward insensitivity is task-specific.** This task's dense, repeated-correction reward structure absorbs corrupted steps; a sparser or single-shot-outcome task would likely show reward degradation directly, which is why we report `decision_flip_rate` as the primary metric rather than reward.
- **Spatial-generalization scope.** Section 6.6 covers only `bottleneck/{entropy,mi}`, not `eccles` or `shared_trunk`, and runs on the one-hot-observation `ScaledScenario` variant rather than the main sweep's stock environment, since `--position-scale` required that variant's parameterized reset logic.
- **The LLM pilot is a pilot.** n=3 per condition, one small model, one cooperative (not adversarial) task. It does not test whether an LLM agent can be induced to *hide* meaning from an overseer, which is the actual steganography-detection setting motivating Section 2's LLM-collusion citations.

## 9. Conclusion and Future Work

A three-symbol Lewis game is small enough to fully instrument and label, and that turns out to matter: at every level we checked — reward, correlational MI, causal intervention, noise robustness, spatial generalization, and a cross-substrate LLM comparison — the naive metric could be misread, and only checking it against ground truth (or against a deliberately irrelevant control) revealed how. The most novel single finding is Section 6.2(b)'s third failure mode: a channel can be causally relied upon by a receiver without carrying real task information, distinct from both "no channel" and "clean protocol." Natural next steps, each already scoped with reasoning for why it was deferred rather than attempted here (see the repository's README "Next steps" section): a larger seed budget for tighter confidence intervals; extending the causal audit to a differentiable Gumbel-softmax channel and an off-policy algorithm (MADDPG/MATD3) to test whether the architecture-vs-loss finding is specific to PPO; and, most directly following from Section 6.7, an adversarial three-role extension of the LLM pilot where a Speaker is incentivized to hide information from an Overseer while still conveying it to a Listener — the actual steganography-detection setting this report's relevance claim points toward but does not yet test. One further extension was considered and deliberately not built: a held-out-target-identity generalization test (train with one landmark index never used as the goal, test on it), the literal counterpart to Rita et al.'s train/test-split framing. Because the speaker's observation is a one-hot over a flat categorical index read by a linear first layer, an index that is always 0 in training receives exactly zero gradient on its associated weight column — the test would mostly measure an untrained column's arbitrary output, not learned generalization, making it low-signal by construction rather than a genuine open question the way Section 6.6's spatial-scale test was. It becomes meaningful only alongside a redesigned target space with compositional or continuous structure (e.g., attribute-based targets), which would also be the natural setting for topographic-similarity-style compositionality metrics, scoped and skipped here for the identical reason.

## References

- Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018). How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)
- Eccles, T., Bachrach, Y., Lever, G., Lazaridou, A., & Graepel, T. (2019). Biases for Emergent Communication in Multi-agent Reinforcement Learning. NeurIPS 2019. [arXiv:1912.05676](https://arxiv.org/abs/1912.05676)
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep Reinforcement Learning that Matters. AAAI 2018. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Hu, W., Miyato, T., Tokui, S., Matsumoto, E., & Sugiyama, M. (2017). Learning Discrete Representations via Information Maximizing Self-Augmented Training. ICML 2017. [arXiv:1702.08720](https://arxiv.org/abs/1702.08720)
- Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P., Strouse, D. J., Leibo, J. Z., & de Freitas, N. (2019). Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning. ICML 2019. [arXiv:1810.08647](https://arxiv.org/abs/1810.08647)
- Krause, A., Perona, P., & Gomes, R. (2010). Discriminative Clustering by Regularized Information Maximization. NeurIPS 2010.
- Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J., & Dauphin, Y. (2019). On the Pitfalls of Measuring Emergent Communication. AAMAS 2019. [arXiv:1903.05168](https://arxiv.org/abs/1903.05168)
- Motwani, S. R., et al. (2024). Secret Collusion among AI Agents: Multi-Agent Deception via Steganography. NeurIPS 2024; Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs. [arXiv:2410.03768](https://arxiv.org/abs/2410.03768)
- Detecting Multi-Agent Collusion Through Multi-Agent Interpretability. [arXiv:2604.01151](https://arxiv.org/abs/2604.01151)
- Rita, M., Michel, P., Chaabouni, R., Pietquin, O., Dupoux, E., & Strub, F. (2022). Emergent Communication: Generalization and Overfitting in Lewis Games. NeurIPS 2022. [arXiv:2209.15342](https://arxiv.org/abs/2209.15342)
- Wang, R., He, X., Yu, R., Qiu, W., An, B., & Rabinovich, Z. (2020). Learning Efficient Multi-agent Communication: An Information Bottleneck Approach. ICML 2020. [arXiv:1911.06992](https://arxiv.org/abs/1911.06992)

## Reproducibility Appendix

All experiments are reproducible from the committed repository (`uv sync --extra dev --frozen && uv run pytest tests/`). Raw per-seed results and orchestration scripts:

| Section | Script(s) | Raw data |
| --- | --- | --- |
| 6.1–6.2 | `tools/run_seed_sweep.py`, `tools/aggregate_seed_sweep.py`, `tools/placebo_audit_sweep.py`, `tools/aggregate_placebo_audit.py` | `results/seed_sweep/` |
| 6.3 | `tools/run_seed_sweep.py --num-landmarks {3,5,8}` | `results/vocab_sweep_n*/` |
| 6.4 | `comm/mi_diagnostic_callback.py` | `results/seed_sweep/*_mi_diagnostic.csv` |
| 6.5 | `comm/noise_robustness.py`, `tools/run_noise_sweep.py`, `tools/aggregate_noise_sweep.py` | `results/noise_sweep/` |
| 6.6 | `comm/evaluate.py --position-scale`, `tools/run_generalization_sweep.py`, `tools/aggregate_generalization_sweep.py` | `results/generalization_sweep/` |
| 6.7 | — (live subagent pilot, not scripted) | `llm_agent_audit_pilot.md` |

An interactive, browser-executable version of the core comm results (live policy playback, confusion matrix, and a placebo-corrected causal-audit visualization) is available at the repository's [live demo](https://claude.ai/code/artifact/43ff9a43-2bfe-4ba7-9bbb-c12a43275fff).
