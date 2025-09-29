  MoE Routing Metrics Explained

  For your 16 experts, top-2 routing setup:

  1. entropy and entropy_normalized

  What it measures: How evenly traffic is distributed across experts.

  Formula:
  H = -Σ(p_i * log(p_i))  # Shannon entropy
  entropy_normalized = H / log(num_experts)  # Normalized to [0, 1]

  Ideal ranges for 16 experts:
  - entropy: 2.0 - 2.5 (out of max 2.77 = log(16))
  - entropy_normalized: 0.7 - 0.9
    - 1.0 = perfectly uniform (all experts used equally)
    - 0.0 = collapsed (all traffic to 1 expert)
    - 0.7-0.9 = healthy specialization with good coverage

  Interpretation:
  - Too high (>0.95): Experts aren't specializing, model may underperform
  - Too low (<0.5): Expert collapse, dead experts, wasted capacity
  - Your target: 0.75-0.85 (balanced specialization)

  ---
  2. kl_to_uniform

  What it measures: How far the routing distribution deviates from uniform.

  Formula:
  KL = Σ(p_i * log(p_i / (1/num_experts)))

  Ideal ranges:
  - 0.0 = perfectly uniform
  - 0.1 - 0.5 = healthy specialization
  - >1.0 = significant imbalance (not necessarily bad!)
  - >2.0 = severe imbalance (likely expert collapse)

  Interpretation:
  - KL divergence and entropy are related: lower entropy → higher KL
  - For top-2 routing with 16 experts, expect KL ≈ 0.3-0.8
  - Rising KL over training = experts specializing (normal early on)
  - Stable high KL (>1.5) = check for dead experts

  ---
  3. load_std

  What it measures: Standard deviation of expert load percentages.

  Example:
  # If expert loads are: [10%, 8%, 9%, 5%, 12%, ...]
  load_std = std([10, 8, 9, 5, 12, ...])

  Ideal ranges (for 16 experts):
  - Perfectly uniform: Each expert gets 6.25% (100/16)
  - load_std: 1.5 - 3.5 percentage points
    - <1.0: Too uniform, no specialization
    - 1.5-3.5: Healthy variation
    - >5.0: High imbalance, check for dead experts

  Interpretation:
  - Lower std = more balanced load
  - With top-2 routing, some imbalance is expected (not all experts equally good)

  ---
  4. load_cv (Coefficient of Variation)

  What it measures: Normalized measure of load imbalance (std/mean).

  Formula:
  load_cv = load_std / load_mean
  # For uniform: mean = 6.25%, so cv = load_std / 6.25

  Ideal ranges:
  - 0.0 = perfectly uniform
  - 0.2 - 0.5 = healthy imbalance
  - >0.8 = severe imbalance
  - >1.5 = likely expert collapse

  Interpretation:
  - Most important metric for load balancing
  - CV is scale-invariant (works for any number of experts)
  - Target: CV < 0.6 for stable training
  - If CV rises above 1.0, increase aux loss weight (currently 0.01 → try 0.02)

  ---
  5. max_load

  What it measures: Highest traffic percentage to any single expert.

  Ideal ranges (for 16 experts):
  - Uniform baseline: 6.25% (100/16)
  - Healthy: 8-15%
  - Concerning: 20-30%
  - Critical: >40% (one expert is dominant)

  Interpretation:
  - With top-2 routing, expect most popular expert to get ~2-3x average
  - max_load = 12-15% is ideal (some specialization, not dominant)
  - If any expert gets >30%, it's becoming a "superexpert" (may need intervention)

  ---
  Expected Evolution During Training

  Early training (steps 0-1000):

  entropy_normalized: 0.85-0.95  (exploration phase)
  kl_to_uniform: 0.1-0.3         (still uniform)
  load_cv: 0.2-0.4               (low variance)
  max_load: 7-10%                (balanced)

  Mid training (steps 1000-5000):

  entropy_normalized: 0.70-0.85  (specialization emerging)
  kl_to_uniform: 0.3-0.8         (diverging from uniform)
  load_cv: 0.3-0.6               (healthy variance)
  max_load: 10-18%               (some experts favored)

  Late training (steps 5000+):

  entropy_normalized: 0.65-0.80  (specialized)
  kl_to_uniform: 0.5-1.2         (distinct preferences)
  load_cv: 0.4-0.7               (stabilized imbalance)
  max_load: 12-20%               (clear favorites)

  ---
  Red Flags to Watch For

  🚨 Expert Collapse:

  entropy_normalized < 0.5
  kl_to_uniform > 2.0
  dead_experts > 4 (out of 16)
  max_load > 35%
  Fix: Increase aux loss weight, increase routing temperature

  🟡 Weak Specialization:

  entropy_normalized > 0.95 (after 2k steps)
  kl_to_uniform < 0.15
  load_cv < 0.2
  Fix: Decrease aux loss weight, check if model is learning

  ✅ Healthy Routing (your goal):

  entropy_normalized: 0.70-0.85
  kl_to_uniform: 0.3-0.8
  load_cv: 0.3-0.6
  dead_experts: 0-2
  drop_rate: <3%





    I still recommend having aux_loss_weight configurable (which I added), so you can experiment:
  # Conservative (current): good for stability
  --aux_loss_weight 0.01

  # Balanced (new default): faster specialization
  --aux_loss_weight 0.005

  # Aggressive (if needed): strong specialization
  --aux_loss_weight 0.003
