🚀 V14-OrbitalClass: Mars Habitat Digital Twin

A high-fidelity, AI-powered Mars habitat simulation system integrating quantum error correction, reinforcement learning, biological life-support modelling, and real-time mission control dashboards.

Table of Contents

Overview

Architecture

Project Structure

Module Reference

Configuration

Installation

Usage

Testing

Export & Deployment

Monitoring & Observability

Improvement Roadmap

Overview

V14-OrbitalClass is a modular, production-grade digital twin of a crewed Mars habitat. It simulates the full operational stack of a Mars mission, including:

Physics — thermal regulation, battery energy management, atmospheric gas dynamics

Biology & ISRU — crop, fish, and chicken biomass growth; in-situ resource utilisation (MOXIE O₂, Sabatier reactor, water extraction)

Quantum Error Correction — surface code stabiliser tracking with MWPM, Neural BP, and Cascade-CNN decoders

Reinforcement Learning — a DreamerV3-style world-model agent trained via simultaneous curriculum over distance, noise, and re-tiling

Real-Time Dashboard — multi-tab Dash/Streamlit mission control with live Tanner graph, noise heatmaps, and crew metrics

Voice Co-Pilot — speech-command interface for hands-free habitat control

Hardware Export — QAT-optimised model export to ExecuTorch, CoreML, XNNPack, and WASM/WebGPU

Architecture

main.py (MarsHabitatManager + MarsHabitatAI) │ ├── config.py — Hydra/OmegaConf typed configuration ├── codes.py — Surface code, stabiliser measurement & retiling ├── physics.py — Battery state-of-charge & thermal physics ├── biology.py — Logistic biomass models + ISRU subsystems ├── encoders.py — Mamba-2 temporal encoder + Multi-Scale GNN ├── decoders.py — MWPM, Neural BP, Cascade-CNN, Syndrome Router ├── agents.py — DreamerV3 world-model agent + CurriculumScheduler ├── environment.py — Gymnasium-compatible environment + SystemState ├── export.py — QAT, torch.export v2, ExecuTorch, WASM pipelines ├── logging_metrics.py — W&B tracker + Prometheus metrics + HDF5 logging ├── voice.py — Speech recognition & text-to-speech co-pilot ├── dashboard.py — Dash multi-tab mission control UI └── tests/ └── test_equivariance.py 

Project Structure

v14-orbitalclass/ ├── config.yaml ├── requirements.txt ├── Dockerfile ├── main.py ├── config.py ├── codes.py ├── physics.py ├── biology.py ├── encoders.py ├── decoders.py ├── agents.py ├── environment.py ├── export.py ├── logging_metrics.py ├── voice.py ├── dashboard.py ├── models/ # Saved agent checkpoints (auto-created) ├── logs/ # HDF5 telemetry logs (auto-created) └── tests/ └── test_equivariance.py 

Module Reference

config.py — Hydra/OmegaConf Configuration

Provides a fully typed MarsConfig dataclass loaded from config.yaml using OmegaConf/Hydra. Exposes structured sub-configs for:

Sub-configKey fieldsTARGET_TEMPvalue, min, max, leak_rateTARGET_O2 / TARGET_CO2value, thresholds, leak_rateBATTERYcapacity_kwh, DoD, charge/discharge efficiencyISRUMOXIE, Sabatier, water extractor ratesCREWsize, calorie needs, radiation limit, skill decayBIOLOGYcarrying capacity K and growth rate r per speciesEVENTSdust storm probability/season, disease, meteor, radiation flareENERGYsolar output, dust reduction, degradation rateDECODERMWPM/NeuralBP/CascadeCNN flags, e2e gradient trainingGNNhidden dimensions, layers, hierarchical poolingTEMPORALMamba-2 model size, history lengthTRAININGtimesteps, batch size, LR, curriculum flags, W&B settings 

codes.py — Surface Code & Stabiliser Tracking

Implements the SurfaceCode class for quantum error simulation:

Distance-d surface code with X/Z check matrices

Syndrome measurement with configurable noise

record_syndrome(syndrome, noise) — appends to rolling syndrome_history and noise_history buffers

dynamic_retile() — remaps stabiliser ordering and applies the inverse permutation to syndrome_history so temporal models see a consistent qubit layout (P0-2)

stabilizer_permutation — stores the active permutation for downstream equivariance verification

physics.py — Battery & Thermal Physics

Battery — tracks energy in kWh; enforces depth-of-discharge limits; models charge/discharge efficiency separately; exposes soc() (state-of-charge)

Physics — thermal update using external temperature; atmospheric gas leak modelling for O₂/CO₂ using configurable leak rates and equipment-status multipliers

biology.py — Biology & ISRU Systems

BiologySystem — logistic growth models for crops, fish, and chickens parameterised by carrying capacity (K) and intrinsic growth rate (r). Light attenuation during dust storms is passed in as a multiplier. Includes gate-level leakage tracker stub (P1-Noise).

ISRUSystem — wraps MOXIE O₂ production, Sabatier CH₄/H₂O reactor, and water extractor. Each subsystem's throughput is scaled by its configured efficiency and the action vector.

encoders.py — Temporal Encoder (Mamba-2 + Multi-Scale GNN)

Two complementary encoders extract features from syndrome history:

MultiScaleGNN

Built on torch_geometric MessagePassing (replacing manual scatter_add)

Hierarchical pooling over check-node hyperedges

return_node_embeddings=True mode exposes per-node embeddings for permutation equivariance testing (P0-1)

Graceful fallback when torch_geometric is unavailable

MambaTemporalEncoder (Mamba-2 / Hyena)

Processes syndrome history of shape (T, n_stabilisers) as a sequence

Uses Mamba-2 SSM blocks when mamba-ssm is installed; falls back to a standard GRU otherwise

Outputs a fixed-size embedding suitable for the decoder suite or RL agent

decoders.py — Decoder Suite

A multi-decoder pipeline with a learned router:

DecoderDescriptionMWPMMinimum-Weight Perfect Matching via pymatching — deterministic baselineNeuralBPDifferentiable belief-propagation with learned edge weights; trained end-to-end via syndrome_matching_loss (P0-3 / P1-Arch)CascadeCNNConvolutional residual network over the 2D syndrome gridSyndromeRouterSmall MLP that selects the best decoder per round based on syndrome statistics (P1-Arch) 

The e2e_grad_decoder flag and e2e_decoder_loss_coeff parameter enable joint training of the decoder with the RL agent, fixing the gradient-leak bug from earlier versions (P0-3).

agents.py — DreamerV3-Style World Model Agent

DreamerAgent — replaces the legacy DQN with a world-model agent. Learns a latent dynamics model, then plans entirely in imagination. Exposes predict(obs) and update(obs, action, reward, next_obs, done).

CurriculumScheduler — simultaneously advances three curriculum axes (surface-code distance, physical noise level, retiling frequency). Tracks episode success rate and gates advancement on a configurable threshold. Supports imitation pretraining for the first imitation_pretrain_steps before switching to RL.

environment.py — Gymnasium Environment

MarsHabitatEnv wraps MarsHabitatAI as a standard gymnasium.Env

Observation space: 23-dimensional continuous vector (temperature, O₂, CO₂, battery SOC, water, nutrients, crew health/mood/radiation, equipment status, biomass, etc.)

Action space: 16-dimensional continuous [0, 1] vector controlling heater, gas generators, water/nutrient dosing, maintenance, harvesting, repairs, and ISRU

SystemState is a typed dataclass with a to_obs() helper

export.py — QAT + torch.export + ExecuTorch Pipeline

Multi-target hardware export for edge deployment:

TargetMethodxnnpacktorch.export v2 + ExecuTorch XNNPack delegatecoremlcoremltools conversion from exported programwasmWASM + WebGPU stub for browser deploymentaot_inductorAOT Inductor for server-side compiled inference 

Quantisation-Aware Training (QAT) is applied before export — not just Post-Training Quantisation — preserving accuracy on edge hardware (P1-HW).

logging_metrics.py — W&B + Prometheus Logging

WandBTracker — wraps wandb with an enabled flag; no-ops cleanly when W&B is disabled

Logger — per-habitat structured logger that writes to an in-memory deque for the dashboard, flushes to HDF5 every 100 steps, and pushes Prometheus gauge metrics

Prometheus metrics are served on a configurable port (default 9090) for Grafana integration

voice.py — Voice Co-Pilot

Uses SpeechRecognition (Google backend) for keyword detection

VoiceCopilot runs a background listener thread and pushes (command, value) tuples to a queue.Queue

MarsHabitatAI._process_voice_commands() drains the queue each step

Supported commands include o2_gen and repair; extensible to the full action space

Text-to-speech alerts via pyttsx3

dashboard.py — Mission Control Dashboard

A multi-tab Dash application (MissionControl) served in a daemon thread:

TabContent📊 OverviewTemperature, O₂, crew health time-series🌿 BiologyCrop, fish, chicken biomass trends⚙️ EngineeringBattery SOC, solar efficiency👩‍🚀 CrewHealth, mood, radiation dose🕸️ Tanner GraphLive syndrome-history heatmap🔥 Noise MapPer-qubit noise estimation heatmap 

Refreshes every 3 seconds via dcc.Interval. Can be run standalone:

streamlit run dashboard.py 

main.py — Entrypoint

MarsHabitatAI — single habitat instance composing all subsystems:

Calls step() every simulation tick, which sequences sensor reading → control application → biology update → energy generation → event handling → crew update → equipment degradation → voice command processing → syndrome update → logging

MarsHabitatManager — orchestrates multiple habitats in parallel:

Instantiates habitats, starts the dashboard, and runs the simulation loop

Feeds observations to DreamerAgent, collects rewards, drives curriculum advancement

Saves all agent checkpoints on exit

CLI flags:

--habitats N Number of parallel habitats (default: 1) --hours N Simulation duration in hours (default: 72) --no-rl Disable RL; use zero-action baseline --port N Dashboard port (default: 8050) --metrics-port N Prometheus port (default: 9090) --wandb Enable Weights & Biases logging --export TARGET Export agent: xnnpack | coreml | wasm | none 

Configuration

All runtime parameters live in config.yaml and are loaded via OmegaConf, making the system compatible with Hydra for sweep-based hyperparameter search.

Key defaults:

TARGET_TEMP: value: 22.0 # °C BATTERY: capacity_kwh: 1000.0 dod_max: 0.8 CREW: size: 6 DECODER: use_mwpm: true use_neural_bp: true e2e_grad_decoder: true e2e_decoder_loss_coeff: 0.05 TEMPORAL: model: mamba2 history_len: 32 TRAINING: total_timesteps: 2000000 curriculum_distance: true curriculum_noise: true curriculum_retile: true 

Installation

Requirements

Python 3.11+

CUDA-capable GPU recommended (Mamba-2 requires CUDA; CPU fallback to GRU is available)

Steps

git clone <repo-url> cd v14-orbitalclass pip install -r requirements.txt 

For Mamba-2 SSM support (requires CUDA):

pip install mamba-ssm>=2.0.0 

For ExecuTorch export (platform-specific):

pip install executorch # see https://pytorch.org/executorch 

Docker

docker build -t mars-habitat-v14 . docker run -p 8050:8050 -p 9090:9090 mars-habitat-v14 

The container exposes port 8050 (dashboard) and 9090 (Prometheus metrics) and runs a 72-hour, single-habitat simulation by default.

Usage

Basic single-habitat, 72-hour simulation:

python main.py --habitats 1 --hours 72 

Multi-habitat run with W&B logging:

python main.py --habitats 3 --hours 200 --wandb 

No-RL baseline (zero-action policy):

python main.py --habitats 1 --hours 10 --no-rl 

Run and export agent to CoreML:

python main.py --habitats 1 --hours 10 --export coreml 

Standalone dashboard:

streamlit run dashboard.py 

Testing

The test suite lives in tests/test_equivariance.py and covers the three P0-priority correctness fixes:

TestWhat it validatestest_node_embedding_equivarianceGNN node outputs satisfy f(π·x)[i] = f(x)[π⁻¹(i)] for any data-qubit permutationtest_global_embedding_invarianceGlobal pooled embedding is permutation-invarianttest_stabilizer_retile_history_consistencyAfter dynamic_retile(), syndrome_history is correctly remapped via the inverse permutationtest_e2e_decoder_gradient_flowsGradients propagate from syndrome_matching_loss back through NeuralBP to both LLR inputs and learned edge weights 

Run all tests:

pytest tests/ -v 

Export & Deployment

The export.py module supports four hardware targets via export_to_executorch(model, example_inputs, output_path, target=...):

TargetUse casexnnpackARM/x86 edge devices (Raspberry Pi, embedded Linux)coremlApple Silicon (iPhone, iPad, Mac)wasmBrowser via WebAssembly + WebGPUaot_inductorHigh-throughput server inference 

All exports apply QAT before conversion to maintain accuracy comparable to the full-precision model.

Monitoring & Observability

ToolDefault portPurposeDash dashboard8050Live habitat visualisationPrometheus9090Gauge metrics for GrafanaW&BcloudExperiment tracking, loss curvesHDF5 logslogs/Persistent per-habitat telemetry 

Improvement Roadmap

PriorityItemStatusP0-1GNN permutation equivariance (return_node_embeddings, torch.roll test)✅P0-2Stabiliser indexing consistency after retiling✅P0-3Decoder gradient leak fix (syndrome_matching_loss + e2e_grad_decoder)✅P0-4StimBackend retiling stub with analytical fallback✅P1-ArchNeural BP with differentiable learned weights✅P1-ArchMamba-2 / Hyena temporal block with graceful fallback✅P1-ArchMulti-scale GNN with hierarchical pooling✅P1-ArchLearned Syndrome Router✅P1-NoiseGate-level leakage tracker stub✅P1-TrainDreamerV3-style world model replacing DQN✅P1-TrainSimultaneous curriculum on distance + noise + retile✅P1-HWQAT (not just PTQ)✅P1-HWtorch.export v2 + ExecuTorch + AOT Inductor✅P1-HWWASM + WebGPU export stub✅P2Visualisation dashboard with Tanner graph + noise heatmap✅QualitySplit monolith into modules✅QualityHydra/OmegaConf config management✅QualityType hints + pyright/mypy✅Qualitytorch_geometric MessagePassing replacing manual scatter_add✅QualityW&B logging + Prometheus metrics✅
