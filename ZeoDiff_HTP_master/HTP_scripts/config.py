# =============================================================================
# Configuration File
# Authors: Xiaobao Wang, Ji Qi
# =============================================================================

import os


class PathConfig:
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    ANN_DIR = os.path.join(BASE_DIR, "models")
    ANN_MODEL_FILE = "ann_final_model.pth"
    SCALER_FILE = "scaler.pkl"
    ANN_MODEL_PATH = os.path.join(ANN_DIR, ANN_MODEL_FILE)
    SCALER_PATH = os.path.join(ANN_DIR, SCALER_FILE)
    
    BUILTIN_FEATURES_FILE = "zeo++_geometric_descriptors.xlsx"
    BUILTIN_FEATURES_PATH = os.path.join(BASE_DIR, BUILTIN_FEATURES_FILE)
    
    FEATURE_EXTRACT_SCRIPT = os.path.join(BASE_DIR, "zeo_feature_extract.py")
    
    @classmethod
    def validate_paths(cls):
        required_paths = {
            'ANN Model': cls.ANN_MODEL_PATH,
            'Scaler File': cls.SCALER_PATH,
            'Built-in Features Database': cls.BUILTIN_FEATURES_PATH,
            'Feature Extraction Script': cls.FEATURE_EXTRACT_SCRIPT,
        }
        
        missing = []
        for name, path in required_paths.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        return missing


# =============================================================================
# Model Configuration
# =============================================================================

class ModelConfig:
    
    HIDDEN_SIZE_1 = 128
    HIDDEN_SIZE_2 = 64
    HIDDEN_SIZE_3 = 32
    OUTPUT_SIZE = 1
    
    FEATURE_NAMES = [
        'FDSi',
        'PLD',
        'PLD/LCD',
        'Vacc',
        'Tort',
        'AvgA',
        'StdA',
        'MaxA',
        'ASA',
    ]
    
    INPUT_SIZE = len(FEATURE_NAMES)


# =============================================================================
# GUI Configuration
# =============================================================================

class GUIConfig:
    
    WINDOW_TITLE = "High-throughput Zeolite Feature Extraction and Diffusion Coefficient Prediction Platform"
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 1000
    MIN_WIDTH = 1200
    MIN_HEIGHT = 800
    
    COLORS = {
        'primary': '#2C3E50',
        'secondary': '#34495E',
        'accent': '#3498DB',
        'success': '#27AE60',
        'warning': '#F39C12',
        'danger': '#E74C3C',
        'light': '#ECF0F1',
        'dark': '#2C3E50',
        'white': '#FFFFFF'
    }


# =============================================================================
# Feature Extraction Configuration
# =============================================================================

class FeatureConfig:
    
    DEFAULT_INPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_cifs")
    DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "zeolites_features.xlsx")
    
    PREPROCESSING = {
        'auto_remove_symmetry': True,
        'symprec': 0.1,
    }
    
    ENERGY_GRID = {
        'spacing_A': 0.2,
        'cutoff_A': 14.0,
        'probe_type': 'ch4',
        'min_length': 28.0,
        'min_distance_A': 0.1,
        'ff_params': {
            'o': (0.3304, 0.442329411),
            'si': (0.2310, 0.18458107),
            'ch4': (0.3730, 1.230540467)
        }
    }
    
    DIFFUSION_CHANNEL = {
        'energy_min': -20,
        'energy_max': 300,
        'cluster_eps': 0.3,
        'min_samples': 10,
        'section_interval': 0.5,
        'min_points_threshold': 100,
        'min_sections': 20
    }
    
    @classmethod
    def get_batch_config(cls):
        return {
            'input_folder': cls.DEFAULT_INPUT_FOLDER,
            'output_file': cls.DEFAULT_OUTPUT_FILE,
            'preprocessing': cls.PREPROCESSING,
            'energy_grid': cls.ENERGY_GRID,
            'diffusion_channel': cls.DIFFUSION_CHANNEL
        }


# =============================================================================
# Utility Functions
# =============================================================================

def print_config_summary():
    print("=" * 80)
    print("Configuration Summary")
    print("=" * 80)
    
    print("\n[Path Configuration]")
    print(f"Base Directory: {PathConfig.BASE_DIR}")
    print(f"ANN Model: {PathConfig.ANN_MODEL_PATH}")
    print(f"Scaler: {PathConfig.SCALER_PATH}")
    print(f"Feature Extraction Script: {PathConfig.FEATURE_EXTRACT_SCRIPT}")
    
    print("\n[Model Configuration]")
    print(f"Input Features: {ModelConfig.INPUT_SIZE}")
    print(f"Hidden Layer Structure: {ModelConfig.HIDDEN_SIZE_1} -> {ModelConfig.HIDDEN_SIZE_2} -> {ModelConfig.HIDDEN_SIZE_3}")
    
    print("\n[Feature Extraction Configuration]")
    print(f"Default Input Folder: {FeatureConfig.DEFAULT_INPUT_FOLDER}")
    print(f"Default Output File: {FeatureConfig.DEFAULT_OUTPUT_FILE}")
    
    print("\n[Validation Results]")
    missing = PathConfig.validate_paths()
    if missing:
        print("⚠ Missing Files:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("✓ All required files found")
    
    print("=" * 80)


if __name__ == "__main__":
    print_config_summary()
