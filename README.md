# Zephyr

A PyTorch framework for my personal research into deep learning-based weather forecasting, re-implementing various SOTA models on global weather data into a common architecture.


## Architecture

### Models

**Stormer** - ViT model.
**GraphCast** - GNN model.

### Training Infrastructure

Built on Accelerate for distributed training with:
- Automatic mixed precision (FP16)
- Gradient clipping and accumulation
- Early stopping and learning rate scheduling
- Latitude-weighted loss functions for physically-realistic error metrics
- Per-variable RMSE and MAE tracking
- Weights & Biases integration for experiment management

## Project Structure

```
zephyr/
├── data/           # Dataset classes, normalization, variable definitions
├── models/         # Model architectures
│   ├── graphs/     # Graph construction (icosahedral, grid)
│   ├── graphcast/  # GraphCast encoder-processor-decoder
│   └── stormer/    # Stormer transformer-graph hybrid
├── training/       # Training loop, losses, metrics
├── analysis/       # Influence analysis and interpretability tools
└── utils/          # Logging, paths, helpers
```

## TODO

- **[CURRENT]** Use graph influence algorithms to track how geospatial structures propagate through autoregressive predictions
- Correlate influence patterns with forecast skill degradation
- Compare influence structures between architectures (GraphCast vs Stormer)
- Implement other models.


## Citations

This project builds on and references the following works:

**GraphCast:**
```
@article{lam2023graphcast,
  title={GraphCast: Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

**WeatherBench:**
```
@article{rasp2020weatherbench,
  title={WeatherBench: A benchmark dataset for data-driven weather forecasting},
  author={Rasp, Stephan and Dueben, Peter D and Scher, Sebastian and Weyn, Jonathan A and Mouatadid, Soukayna and Thuerey, Nils},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={12},
  number={11},
  year={2020}
}
```

**Stormer:**
```
@inproceedings{nguyen2024stormer,
  title={Scaling transformer neural networks for skillful and reliable medium-range weather forecasting},
  author={Nguyen, Tung and Shah, Rohan and Bansal, Hritik and Arcomano, Troy and Maulik, Romit and Kotamarthi, Veerabhadra and Foster, Ian and Madireddy, Sandeep and Grover, Aditya},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## License

This project is research code. Please cite appropriately if you use or build upon this work.
