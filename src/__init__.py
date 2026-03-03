"""
src — BGA-AISNP core library
=============================
Modules
-------
preprocessing   : data loading & genotype encoding
model_registry  : classifier definitions, param grids, tuning
training        : unified train loop (train_all)
evaluation      : metrics, plots, Excel export
generative_model: Bayesian generative BGA classifier
tabpfn_model    : TabPFN wrapper (optional)

Legacy (kept for backward compatibility):
  data_utils    : original dev-A preprocessing (use preprocessing instead)
  models        : original dev-A XGBoost factory  (use model_registry instead)
"""
