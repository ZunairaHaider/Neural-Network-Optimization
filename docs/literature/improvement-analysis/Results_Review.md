# Results Review of the Cited Patents

---

## Scope and How to Read


**Source labels**

- **[patent]** = stated in the patent body.
- **[assoc. research]** = from the inventors’ associated publication. Patents often report results only in figures or qualitatively.

**Improvement references**

- **Patent §2.x** = Part I of the *Improvement & Reference Package*.
- **Pruning §2.x** = Part II of the *Improvement & Reference Package*.

---


## US 10,614,354 — FPGA CNN Accelerator  
**Inventors / assignee:** Aydonat, Altera/Intel

### Result

- **[patent]** The patent describes a PE-array CNN engine with on-chip tiling and double-buffering to hide DRAM latency.
- **[assoc. research]** The matching Arria-10 OpenCL accelerator sustained approximately **1.38 TFLOP/s on AlexNet**, which was a then-record for FPGA CNN inference.

### Relation — Benefit

This patent provides a deployment substrate.

Our networks are very small and sparse, with about **116–291 links**, so a dense 1-TFLOP/s accelerator is over-provisioned for our current models. The transferable result is not the absolute throughput number, but the **tiling + double-buffering pattern** and the lesson about **which sparsity patterns actually accelerate**.

### Feeds improvement

- **Patent §2.6:** deploy the sparse network efficiently.
- **Pruning §2.3:** use FPGA targets to guide structured-pruning granularity and the `compare.py` benchmark.

---

## US 11,158,096 — Photonic Inverse Design via Straight-Through Estimators  
**Inventors / assignee:** Schubert, Google X

### Result

- **[patent]** The patent describes a binarized “paintbrush” design optimized under fabrication constraints.
- It reports that a tanh-based straight-through estimator, or **STE**, gives superior convergence.
- It also reports that binarized designs reduce memory and computation.
- Device performance is shown in figures.

### Relation — Benefit

The useful result is that one can back-propagate through discrete binary on/off choices using an STE and still converge. This technique is relevant for differentiating otherwise-discrete decisions, such as whether a connection should be added or removed.

### Feeds improvement

- **Patent §2.1:** relax the discrete “add this connection?” decision into a differentiable score, complementing the Lagrangian estimate.
- **Patent §2.4:** ML-surrogate scoring.

---


## US 2019/0130278 — ProGAN, Progressive Growing of GANs  
**Inventors / assignee:** Karras, NVIDIA

### Result

- **[patent]** The patent grows the GAN by adding layers with a `1−α` fade-in mechanism.
- It reaches **1024 × 1024** output resolution and stops at a target accuracy.
- The training remains stable as layers increase.
- **[assoc. research]** The associated ProGAN work reports **1024² CelebA-HQ** image generation and a record **CIFAR-10 inception score of 8.80**, with reduced training time.

### Relation — Compare / Benefit

Progressive growth is conceptually similar to our additive topology-optimization phase.

The concrete transferable result is that **scaling new structure in gradually**, using an `α` fade-in, stabilizes network growth.

Our method grows by adding a single node or link scored in closed form on about **1% data**. The philosophy is similar, but our method is designed to be much cheaper because it uses Lagrangian scoring rather than repeated full training.

### Feeds improvement

- **Patent §2.1:** full `ADD_SKIP` enumeration.
- Add an `α` fade-in ramp for newly added nodes or links to smooth the Gauss–Newton refit.

---

## US 2019/0171936 — Progressive Modification of Neural Networks  
**Inventors / assignee:** Karras, NVIDIA

### Result

- **[patent]** The patent describes a resolution ladder, for example `16 → 32 → ... → 512`.
- The topology evolves toward a predetermined complex target.
- It uses image-quality metrics such as **PSNR**, **SSIM**, and perceptual distance.
- It stops at a target accuracy.
- It applies to U-Net topologies.

### Relation — Benefit

This confirms that growing the topology during training can work for encoder–decoder and U-Net structures.

Its “stop at target accuracy” logic mirrors our validation-based stopping criterion in **Claim 8** of US 12,124,779 B2.

### Feeds improvement

- **Patent §2.1:** additive enumeration.
- **Patent §2.2:** growing the compression and decompression blocks toward a target.

---

## US 2018/0322388 — End-to-End Learned Communications  
**Inventors / assignee:** O’Shea, Virginia Tech

### Result

- **[patent]** The patent trains an encoder and decoder jointly over RF channels to optimize low BER or another selected objective.
- The system is deployable for radio communication.
- **[assoc. research]** Learned autoencoder constellations match or beat conventional modulation on AWGN channels, with over-the-air demonstrations.

### Relation — Directly Related

The encoder → channel → decoder pipeline mirrors our target structure:

```text
compression → modelling → decompression
```

This is directly connected to **Claim 10** of US 12,124,779 B2. It validates training the compressor jointly with the task end-to-end rather than only training it as a separate reconstruction module.

### Feeds improvement

- **Patent §2.2:** implement Claim 10 as one end-to-end objective:

```text
X → C → modelling → D → Y
```

---

## US 2020/0167659 — Architecture-Variation NAS  
**Inventors / assignee:** Moon, ETRI

### Result

- **[patent]** The patent describes variable-length-string candidate architectures generated by unsupervised architecture variation.
- It selects an architecture that meets a target before supervised fine-tuning.
- It claims improved efficiency through fewer hyperparameters and improved accuracy.
- No benchmark numbers are given in the patent body.

### Relation — Compare

This is a NAS competitor that trains or evaluates many candidate strings.

Our Lagrangian method scores candidates in closed form without retraining, which gives our method an efficiency advantage on small data.

Moon’s idea of selecting candidates that meet a target before full training is a useful pre-screening idea.

### Feeds improvement

- **Patent §2.5:** candidate pre-screening.
- **Patent §2.1:** candidate generation.

---

## US 2020/0364545 — Computational-Efficiency Neural Networks  
**Inventors / assignee:** Shattil, Genghiscomm

### Result

- **[patent]** The patent describes base + expanded-matrix weight updates.
- It uses piecewise-linear activations.
- It uses bit-shift arithmetic, for example divide-by-2 operations.
- It describes weight matrices reducing to the identity.
- It claims lower memory and faster processing “without significantly reducing accuracy,” but the result is qualitative.

### Relation — Benefit

The identity reduction, sparse-matrix reduction, and bit-shift ideas are compatible with executing our already-sparse networks cheaply.

### Feeds improvement

- **Patent §2.6:** efficient sparse inference and deployment.

---

## US 9,336,483 — Dynamically Updated NN for Content Distribution  
**Inventors / assignee:** Abeysooriya, Pearson

### Result

- **[patent]** The patent describes a lifecycle that generates, trains, evaluates, and atomically promotes replacement candidate networks in production.
- This is a systems result.
- No accuracy benchmark is reported.

### Relation — Contrast / Benefit

The patent’s “train many candidates and promote the best” strategy is exactly the kind of cost that our Lagrangian approach tries to avoid.

However, it is a useful template for online refresh:

```text
monitor → candidate → promote / rollback
```

### Feeds improvement

- **Patent §2.7:** online adaptive update via an `--online` flag.

---


## US 5,636,326 — Optimal Brain Surgeon  
**Inventors / assignee:** Hassibi & Stork, Ricoh  
**Status:** Strongest direct comparison

### Result

**[patent, concrete]** On the MONK’s problems, OBS pruned a trained network to:

| Problem | Prior method of Thrun et al. | OBS result | Reduction |
|---|---:|---:|---:|
| MONK problem 1 | 58 weights | 14 weights | 76% |
| MONK problem 2 | 39 weights | 15 weights | 62% |
| MONK problem 3 | 39 weights | 4 weights | 90% |

The patent reports equal performance while using full-Hessian saliency plus a closed-form correction to surviving weights that minimizes retraining.

### Relation — Directly Related

OBS is a sibling of our subtractive phase because both are second-order parsimony methods.

The difference is:

| OBS | Our subtractive phase |
|---|---|
| Starts from a trained dense network. | Interleaves additive and subtractive topology optimization. |
| Prunes by inverting the full Hessian. | Removes nodes/links through a Lagrangian and zero-memory Gauss–Newton estimate. |
| Uses closed-form correction after deletion. | Avoids full retraining and full-Hessian inversion. |
| Implemented directly as `obs.py` in the codebase. | Implemented in the patent topology-search pipeline. |

### Feeds improvement

- **Pruning §2.1:** authoritative specification for `obs.py`.
- Use OBS saliency and closed-form post-deletion correction.
- Justifies more frequent Hessian updates and rank-1 inverse maintenance.

---

## US 6,038,389 — Krylov Reduced-Order Modelling  
**Inventors / assignee:** Rahon & Masmoudi, IFP

### Result

- **[patent]** The patent reduces an `n`-grid-cell reservoir-flow FEM system to a Krylov subspace of dimension `d ≪ n`.
- The dimension `d` is described as being “of the order of a few.”
- This gives accurate, bias-free solutions at far lower cost for long simulations.

### Relation — Compare

This is the reduced-model prior art contrasted in the background and FIG. 1 of US 12,124,779 B2.

Rahon reduces the physical system in a **linear Krylov subspace** and recurses in real space. By contrast, US 12,124,779 B2 compresses non-linearly with a learned block and recurses in the compressed space, which should be more stable and can filter noise.

A further connection is that Masmoudi is also an ADAGOS inventor.

### Feeds improvement

- **Patent §2.2:** model inside the reduced or compressed representation, then reconstruct — Claim 10.
- **Patent §2.3:** physical-domain support.

---


## US 2015/0106311 — NIDA / DANNA Dynamic Neuromorphic Architecture  
**Inventors / assignee:** Birdwell & Dean, University of Tennessee

### Result

- **[patent]** The patent describes an evolutionary-optimization procedure that builds the network by implanting useful substructures and extracting unsuccessful ones.
- The system is hardware-mappable through DANNA.
- The method is stated to be supported by experiment, with results shown through worked examples and figures.

### Relation — Compare / Benefit

This is another automatic structure-construction method.

The contrast is:

| NIDA / DANNA | Our method |
|---|---|
| Population-based evolutionary search. | Single-candidate add/remove with a closed-form Lagrangian score. |
| Requires many evaluations. | Designed to reduce candidate-evaluation cost. |
| Extracts unsuccessful substructures. | Can inspire structural pruning and redundancy pruning. |

Its “extract unsuccessful substructures” idea is especially useful as an evolutionary structural-pruning concept.

### Feeds improvement

- **Pruning §2.2:** substructure extraction for `struct_pruning.py` and `redundancy_pruning.py`.
- **Patent §2.6:** DANNA neuromorphic hardware as a deployment route.

---

## US 2021/0390396 — Generative Models for Design  
**Inventors / assignee:** Fan, Stanford  
**Status:** Quantitative

### Result

- **[patent]** A GAN generator produces free-form metasurface designs.
- The patent reports deflection-efficiency distributions of generated designs against a brute-force topology-optimization baseline, using a 60-percentile efficiency threshold.
- **[assoc. research]** Generated metagratings reach the high-efficiency tail of brute-force optimization after light refinement, while covering the design space about an order of magnitude faster.

### Relation — Benefit

This patent provides an alternative ML-for-design paradigm:

```text
generate many designs
```

rather than simply building one surrogate.

Where many candidate designs are needed, for example when exploring excitations or geometries for a digital twin, a Fan-style generator could propose candidates that our fast surrogate then evaluates.

### Feeds improvement

This is complementary to the plan rather than a single-section change.

A possible extension is:

```text
generator + surrogate loop
```

This loop would sit on top of the deployed parsimonious model and use it as a fast evaluator.

It also informs:

- **Patent §2.4:** cheap surrogate scoring.

---

# At-a-Glance: Result → Relevance → Improvement

| Patent | Result source | Relation | Feeds improvement |
|---|---|---|---|
| US 10,614,354 — Aydonat | ≈1.38 TFLOP/s on AlexNet [assoc.]; tiling/double-buffering [patent] | Benefit — deploy | Patent §2.6; Pruning §2.3 |
| US 11,158,096 — Schubert | STE back-prop through binary choices converges [patent] | Benefit | Patent §2.1 / §2.4 |
| US 2019/0130278 — ProGAN | 1024²; inception 8.80 [assoc.]; α fade-in [patent] | Compare / benefit | Patent §2.1 |
| US 2019/0171936 — Karras | Grow to target topology; PSNR/SSIM stop [patent] | Benefit | Patent §2.1 / §2.2 |
| US 2018/0322388 — O’Shea | Learned constellations match/beat QAM; OTA [assoc.] | Directly related | Patent §2.2, Claim 10 |
| US 2020/0167659 — Moon | Efficient NAS; fewer hyperparameters [patent] | Compare | Patent §2.5 / §2.1 |
| US 2020/0364545 — Shattil | Lower memory/compute, accuracy kept [patent] | Benefit | Patent §2.6 |
| US 9,336,483 — Abeysooriya | Candidate generate/evaluate/promote [patent] | Contrast / benefit | Patent §2.7 |
| US 5,636,326 — Stork OBS | 76% / 62% / 90% weight reduction on MONK [patent] | Compare — sibling | Pruning §2.1 = `obs.py` |
| US 6,038,389 — Rahon | Reduce FEM to Krylov dimension `d ≪ n` [patent] | Compare | Patent §2.2 / §2.3 |
| US 2015/0106311 — Birdwell | Implant/extract substructures [patent] | Compare / benefit | Pruning §2.2; Patent §2.6 |
| US 2021/0390396 — Fan | GAN ≈ brute-force efficiency, about one order of magnitude faster [assoc.] | Benefit | Generator + surrogate loop |

---

# Synthesis

Three patents report crisp, comparable numbers:

1. **Stork OBS**  
   Reports **76% / 62% / 90%** weight reduction on MONK’s problems.

2. **ProGAN**  
   Reports **1024²** output and **CIFAR-10 inception score 8.80**.

3. **Fan generative design**  
   Reports approximately brute-force-level efficiency and about **one order-of-magnitude faster** coverage of the design space.

Among these, **Stork OBS** is the one that truly lies on our parsimony axis and directly feeds the pruning team, because it corresponds to `obs.py`.

Most of this patent set consists of enablers that feed concrete improvement directions:

| Improvement direction | Supporting patents |
|---|---|
| Claim 10 compression–modelling–decompression | O’Shea; Rahon |
| Additive growth | ProGAN; Karras |
| Pre-screening / search efficiency | Moon |
| Deployment | Aydonat; Shattil; Birdwell |
| Online update | Abeysooriya |
| Discretization | Schubert STE |

Fan is complementary rather than a single-section change. It suggests a generator + surrogate loop that uses the deployed parsimonious model as a fast evaluator.

