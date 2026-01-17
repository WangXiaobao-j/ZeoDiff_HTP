# ZeoDiff – Feature Extraction and Diffusion Prediction for Zeolites

ZeoDiff is an easy-to-use desktop application for predicting diffusion properties of zeolites. The software automatically extracts structural features from CIF crystal files and uses trained artificial neural networks (ANNs) to estimate molecular diffusion coefficients.

## Features

The software provides two main functions:

1. **Single Structure Prediction**  
   Predict the diffusion coefficient for a single zeolite structure from a CIF file. Results are displayed instantly in the application interface.

2. **Batch Processing**  
   Predict diffusion coefficients for multiple zeolite structures in a single run. Results are automatically compiled and saved to an Excel file for further analysis.

## Requirements

```
python         3.7+
matplotlib     3.5.1
numpy          1.21.2
pandas         1.3.4
scikit-learn   0.22.1
torch          1.7.1
tqdm           4.42.1
mendeleev      0.9.0
ase            3.22.0
pymatgen       2022.0.0
scipy          1.7.3
openpyxl       3.0.9
joblib         1.1.0
```


### Running the Software

```bash
python zeo_diff_predict.py
```

## Usage

### Single File Mode
1. Select "Single File" mode
2. Browse and choose your CIF file
3. Click "Run Prediction"
4. View results on screen

### Batch Mode
1. Select "Batch Mode"
2. Choose folder containing CIF files
3. Specify output Excel file location
4. Click "Run Batch Processing"
5. Results saved to `output/predictions.xlsx`

## File Structure

```
├── zeo_diff_predict.py              # Main GUI application
├── zeo_feature_extract.py           # Feature extraction module
├── config.py                        # Configuration settings
├── models/
│   ├── ann_final_model.pth          # Trained ANN model
│   └── scaler.pkl                   # Data scaler
└── zeo++_geometric_descriptors.xlsx # Features computed from Zeo++
```

## Output

The descriptors used for diffusion prediction include:

| Feature | Source | Description |
|---------|--------|-------------|
| FDSi | IZA/POCD | Framework silicon density (T/1000Å³) |
| PLD | Zeo++ | Pore limiting diameter (Å) |
| PLD/LCD | Zeo++ | Ratio of PLD to LCD |
| Vacc | Zeo++ | Accessible void fraction (%) |
| ASA | Zeo++ | Accessible surface area (m²/cm³) |
| Tort | Computed | Tortuosity |
| AvgA | Computed | Average cross-sectional area (Å²) |
| StdA | Computed | Standard deviation of cross-sectional area (Å²) |
| MaxA | Computed | Maximum cross-sectional area (Å²) |
| **Predicted_Ds (m2/s)** | **ANN** | **Diffusion coefficient** |

## Configuration

Modify `config.py` to customize:
- Model paths
- Feature extraction parameters
- Input/output directories

## Citation

"Harnessing confinement effect and interpretable machine learning to predict and guide alkane diffusion in zeolite catalysts"

## Contact

For questions or issues, contact: WXB047721@126.com
