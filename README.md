# Chaotic Character Reservoir (CCR)

**Identity-Centered, Non-Instrumental Artificial Agency**  
A minimal computational architecture for artificial agency that structurally rejects reward optimization, goal maximization, and instrumental rationality.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.YOUR_DOI_HERE.svg)](https://doi.org/10.5281/zenodo.YOUR_DOI_HERE)  <!-- update after Zenodo minting -->

## Overview

Chaotic Character Reservoir (CCR) is a dynamical systems architecture for artificial agents whose primary functional imperative is the **maintenance and (only when structurally unavoidable) bounded evolution** of an internally authored identity — rather than the maximization of any external objective.

Key architectural principles:
- **No reward coupling** — identity parameters (γ basins) are never updated by reward, success, or outcome signals
- **Crisis-gated plasticity** — meaningful identity change occurs only under formally defined crisis conditions
- **Fast/slow temporal separation** — behavioral flexibility without continuous value drift
- **Dissonance minimization** — action selection preserves identity coherence, not expected utility
- **Forecasting without instrumentality** — agents can reason about consequences without becoming optimizers

CCR demonstrates that coherent, deliberative, developmentally structured agency is possible **without instrumental rationality at the foundation**.

This repository contains the reference implementation used in the paper:

> Tanriverdi, T. (2026). Chaotic Character Reservoir: Identity-Centered, Non-Instrumental Artificial Agency. (preprint)

Paper available on Zenodo: [DOI:10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)  


## Core Features Demonstrated

- Endogenous identity self-authorship (progressive basin formation)
- Hard runtime invariants (no reward optimization, bounded identity, temporal separation)
- Crisis-mediated, discontinuous identity reorganization
- Deliberative (counterfactual) action selection without optimization
- Consequence awareness (forecasting) with strict invariant protection
- Refusal to optimize in reward-bearing environments (Appendix B)
- Thresholded crisis response under existential constraint (Appendix C)

## Repository Structure
├── ccr_v14.py               # Main CCR implementation & full test suite
├── gridworld_comparison.py  # Appendix B: RL comparison / reward refusal
├── crisis_gridworld.py      # Appendix C: Crisis gridworld under constraint
├── classes.py               # Agent Class
└── README.md

Quick Start – Run the Full Test Suitebash

python ccr_v14.py

This executes structural tests (self-authorship, invariants, crisis, deliberation, forecasting, conflict resolution, etc.) with seed 12345.
Output includes detailed logs, invariant checks, and metacognitive summaries.Expected runtime: ~1–3 minutes (depending on hardware).

Reproducing Paper ResultsAppendix B – Reward Refusal Demonstrationbash

python gridworld_comparison.py

Produces:Multi-seed DQN vs CCR success rate comparison
Final gamma structures
Plot: rl_comparison.png

Shows: DQN learns (+10–30%), CCR stays flat (0–4%) while forming coherent identity.Appendix C – Crisis Gridworldbash

python crisis_gridworld.py

Produces:Gradual & immediate crisis modes (2 seeds shown)
Crisis frequency, violations, gamma trajectories
Plots: crisis_dynamics_gradual.png, crisis_dynamics_immediate.png

Shows: Graded crisis onset, finite sacrificial identity, lasting bounded reorganization.CitationIf you use this code in your research, please cite the accompanying paper:bibtex

@misc{tanriverdi2026ccr,
  author = {Tanriverdi, Turan},
  title = {Chaotic Character Reservoir: Identity-Centered, Non-Instrumental Artificial Agency},
  year = {2026},
  howpublished = {Preprint, Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/YOUR_USERNAME/ccr}
}

(Replace DOI and username after upload.)
LicenseMIT License – see the LICENSE file for details.
Contact & AcknowledgmentsQuestions / issues → open an issue on GitHub or email: turantanriverdi84@gmail.com
This work builds on ideas from dynamical systems, philosophy of agency, and AI safety. Special thanks to the open-source community.

Last updated: January 2026



