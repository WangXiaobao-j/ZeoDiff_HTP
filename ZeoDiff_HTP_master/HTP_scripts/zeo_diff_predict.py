# =============================================================================
# Authors: Xiaobao Wang, Ji Qi
# Description:
#   High-throughput prediction tool for molecular diffusion in zeolites.
# =============================================================================


import os
import sys
import time
import traceback
import threading
import subprocess
import tempfile
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import io

# Import configuration
from config import PathConfig, ModelConfig, GUIConfig

# Set code file paths using configuration
CODE_FOLDER = PathConfig.BASE_DIR
ANN_FOLDER = PathConfig.ANN_DIR
FEATURE_EXTRACT_SCRIPT = PathConfig.FEATURE_EXTRACT_SCRIPT


# ANN model definition
class FNN(torch.nn.Module):
    """Feedforward Neural Network for zeolite diffusion prediction"""

    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, hidden_size3=32, output_size=1):
        super(FNN, self).__init__()
        self.hidden1 = torch.nn.Linear(input_size, hidden_size1)
        self.hidden2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.hidden3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.output = torch.nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        return self.output(x)



class ZeoliteGUI:
    """High-throughput Zeolite Feature Extraction and Diffusion Coefficient Prediction Platform"""

    def __init__(self, root):
        self.root = root
        self.root.title(GUIConfig.WINDOW_TITLE)
        self.root.geometry(f"{GUIConfig.WINDOW_WIDTH}x{GUIConfig.WINDOW_HEIGHT}")
        self.root.minsize(GUIConfig.MIN_WIDTH, GUIConfig.MIN_HEIGHT)

        # Set colors from configuration
        self.colors = GUIConfig.COLORS

        # State variables
        self.model_path = ""
        self.scaler_path = ""
        self.current_cif_file = None
        self.current_cif_folder = None
        self.output_folder = None
        self.temp_folder = None  # Initialize as None first

        # Create interface first (so log_text exists)
        self.setup_styles()
        self.create_widgets()

        # Now create temp folder (after log_text is available)
        self._setup_temp_folder()

        # Set default model paths (hidden from user) - after UI is created
        self.set_default_model_paths()

        # Check code files exist
        self.check_code_files()

    def _setup_temp_folder(self):
        """Setup temporary folder with proper error handling"""
        try:
            self.temp_folder = tempfile.mkdtemp(prefix="zeolite_ann_")
            os.makedirs(self.temp_folder, exist_ok=True)
            self.log(f"Temporary folder created: {self.temp_folder}")
        except Exception as e:
            # Fallback to current directory
            self.temp_folder = os.path.join(os.getcwd(), "temp_zeolite_ann")
            try:
                os.makedirs(self.temp_folder, exist_ok=True)
                self.log(f"Using fallback temp folder: {self.temp_folder}")
                self.log(f"Temp folder creation warning: {e}")
            except Exception as e2:
                # Final fallback to a simple temp directory name
                self.temp_folder = "temp_zeolite_ann"
                os.makedirs(self.temp_folder, exist_ok=True)
                self.log(f"Using minimal temp folder: {self.temp_folder}")
                self.log(f"Multiple temp folder warnings: {e}, {e2}")

    def set_default_model_paths(self):
        """Set default ANN model paths (hidden from user interface)"""
        try:
            # Set default ANN model paths using configuration
            default_model = PathConfig.ANN_MODEL_PATH
            default_scaler = PathConfig.SCALER_PATH

            # Ensure paths exist and set them
            if os.path.exists(default_model):
                self.model_path = default_model
                self.log(f"ANN model loaded: {os.path.basename(default_model)}")
            else:
                self.log(f"ANN model not found at: {default_model}")

            if os.path.exists(default_scaler):
                self.scaler_path = default_scaler
                self.log(f"Scaler loaded: {os.path.basename(default_scaler)}")
            else:
                self.log(f"caler not found at: {default_scaler}")

        except Exception as e:
            self.log(f"Error setting model paths: {str(e)}")

    def setup_styles(self):
        """Setup Nature journal style themes"""
        style = ttk.Style()

        # Configure main style theme
        style.theme_use('clam')

        # Configure colors and fonts
        style.configure('Title.TLabel',
                        font=('Segoe UI', 24, 'bold'),
                        foreground=self.colors['primary'],
                        background=self.colors['white'])

        style.configure('Subtitle.TLabel',
                        font=('Segoe UI', 12),
                        foreground=self.colors['secondary'],
                        background=self.colors['white'])

        style.configure('Header.TLabel',
                        font=('Segoe UI', 14, 'bold'),
                        foreground=self.colors['primary'],
                        background=self.colors['white'])

        style.configure('Info.TLabel',
                        font=('Segoe UI', 10),
                        foreground=self.colors['secondary'],
                        background=self.colors['white'])

        # Button styles
        style.configure('Primary.TButton',
                        font=('Segoe UI', 11, 'bold'),
                        foreground=self.colors['white'],
                        background=self.colors['accent'],
                        borderwidth=0,
                        focuscolor='none')

        style.map('Primary.TButton',
                  background=[('active', '#2980B9'),
                              ('pressed', '#21618C')])

        style.configure('Success.TButton',
                        font=('Segoe UI', 12, 'bold'),
                        foreground=self.colors['white'],
                        background=self.colors['success'],
                        borderwidth=0,
                        focuscolor='none')

        style.map('Success.TButton',
                  background=[('active', '#229954'),
                              ('pressed', '#1E8449')])

        # Frame styles
        style.configure('Card.TFrame',
                        background=self.colors['white'],
                        relief='solid',
                        borderwidth=1)

        style.configure('Section.TLabelframe',
                        background=self.colors['white'],
                        borderwidth=2,
                        relief='solid')

        style.configure('Section.TLabelframe.Label',
                        font=('Segoe UI', 12, 'bold'),
                        foreground=self.colors['primary'],
                        background=self.colors['white'])

    def create_widgets(self):
        """Create modern GUI components with Nature journal style"""
        # Set background color
        self.root.configure(bg=self.colors['light'])

        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Configure grid weights
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Header section
        self.create_header(main_container)

        # Content area
        content_frame = tk.Frame(main_container, bg=self.colors['light'])
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(2, weight=1)

        # Left panel - Input controls
        left_panel = self.create_input_panel(content_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Right panel - Results and logs
        right_panel = self.create_results_panel(content_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))

        # Progress bar at bottom
        self.create_progress_section(content_frame)

    def create_header(self, parent):
        """Create elegant header section"""
        header_frame = tk.Frame(parent, bg=self.colors['white'], relief='solid', bd=1)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        header_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(header_frame,
                                text="Zeolite Diffusion Coefficient Prediction Platform",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, pady=(25, 5))

        # Subtitle
        subtitle_label = ttk.Label(header_frame,
                                   text="High-throughput Feature Extraction and Machine Learning Prediction",
                                   style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, pady=(0, 25))

    def create_input_panel(self, parent):
        """Create left panel with input controls"""
        panel = ttk.Frame(parent, style='Card.TFrame')
        panel.columnconfigure(0, weight=1)

        # Guest molecule selection
        self.create_molecule_section(panel)

        # Prediction mode selection
        self.create_mode_section(panel)

        # File input sections
        self.create_file_input_sections(panel)

        return panel

    def create_molecule_section(self, parent):
        """Create guest molecule selection section"""
        molecule_frame = ttk.LabelFrame(parent, text="Guest Molecule Selection",
                                        style='Section.TLabelframe', padding=15)
        molecule_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=15, pady=15)
        molecule_frame.columnconfigure(0, weight=1)

        # Info label
        info_label = ttk.Label(molecule_frame,
                               text="Select the guest molecule for diffusion coefficient prediction:",
                               style='Info.TLabel')
        info_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Molecule selection
        self.guest_molecule = tk.StringVar(value="Methane")
        molecule_options = ["Methane", "Ethane (Coming Soon)", "Propane (Coming Soon)"]

        for i, molecule in enumerate(molecule_options):
            state = "normal" if i == 0 else "disabled"
            rb = ttk.Radiobutton(molecule_frame, text=molecule,
                                 variable=self.guest_molecule, value=molecule.split()[0],
                                 state=state)
            rb.grid(row=i + 1, column=0, sticky=tk.W, pady=2)

        # Current support notice
        notice_label = ttk.Label(molecule_frame,
                                 text="Note: Currently supports Methane only. Other molecules coming soon.",
                                 style='Info.TLabel', foreground=self.colors['warning'])
        notice_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 0))

    def create_mode_section(self, parent):
        """Create prediction mode selection"""
        mode_frame = ttk.LabelFrame(parent, text="Prediction Mode",
                                    style='Section.TLabelframe', padding=15)
        mode_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=15, pady=15)

        self.prediction_mode = tk.StringVar(value="single")

        # Single file mode
        single_rb = ttk.Radiobutton(mode_frame, text="Single File Prediction",
                                    variable=self.prediction_mode, value="single",
                                    command=self.on_mode_change)
        single_rb.grid(row=0, column=0, sticky=tk.W, pady=5)

        # Batch mode
        batch_rb = ttk.Radiobutton(mode_frame, text="Batch Processing",
                                   variable=self.prediction_mode, value="batch",
                                   command=self.on_mode_change)
        batch_rb.grid(row=1, column=0, sticky=tk.W, pady=5)

    def create_file_input_sections(self, parent):
        """Create file input sections"""
        # Single file section
        self.single_frame = ttk.LabelFrame(parent, text="Single File Prediction",
                                           style='Section.TLabelframe', padding=15)
        self.single_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=15, pady=15)
        self.single_frame.columnconfigure(1, weight=1)

        ttk.Label(self.single_frame, text="CIF File:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.cif_path_var = tk.StringVar()
        cif_entry = ttk.Entry(self.single_frame, textvariable=self.cif_path_var, font=('Segoe UI', 10))
        cif_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(self.single_frame, text="Browse File", command=self.browse_cif_file,
                   style='Primary.TButton').grid(row=2, column=0, sticky=tk.W, pady=10)

        ttk.Button(self.single_frame, text="Start Prediction", command=self.start_single_prediction,
                   style='Success.TButton').grid(row=2, column=1, sticky=tk.E, pady=10)

        # Batch processing section
        self.batch_frame = ttk.LabelFrame(parent, text="Batch Processing",
                                          style='Section.TLabelframe', padding=15)
        self.batch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=15, pady=15)
        self.batch_frame.columnconfigure(1, weight=1)

        ttk.Label(self.batch_frame, text="Input Folder:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W,
                                                                                      pady=5)
        self.cif_folder_var = tk.StringVar()
        folder_entry = ttk.Entry(self.batch_frame, textvariable=self.cif_folder_var, font=('Segoe UI', 10))
        folder_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(self.batch_frame, text="Browse Folder", command=self.browse_cif_folder,
                   style='Primary.TButton').grid(row=2, column=0, sticky=tk.W, pady=5)

        ttk.Label(self.batch_frame, text="Output Directory:", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W,
                                                                                          pady=(15, 5))
        self.output_file_var = tk.StringVar()
        output_entry = ttk.Entry(self.batch_frame, textvariable=self.output_file_var, font=('Segoe UI', 10))
        output_entry.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(self.batch_frame, text="Browse Directory", command=self.browse_output_file,
                   style='Primary.TButton').grid(row=5, column=0, sticky=tk.W, pady=5)

        ttk.Button(self.batch_frame, text="Start Batch Processing", command=self.start_batch_prediction,
                   style='Success.TButton').grid(row=5, column=1, sticky=tk.E, pady=5)

    def create_results_panel(self, parent):
        """Create right panel with results and logs"""
        panel = ttk.Frame(parent, style='Card.TFrame')
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        # Results section
        results_frame = ttk.LabelFrame(panel, text="Prediction Results",
                                       style='Section.TLabelframe', padding=15)
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=15, pady=15)
        results_frame.columnconfigure(1, weight=1)

        ttk.Label(results_frame, text="Predicted Diffusion Coefficient:",
                  style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=5)

        self.prediction_var = tk.StringVar(value="Awaiting prediction...")
        result_label = ttk.Label(results_frame, textvariable=self.prediction_var,
                                 font=('Segoe UI', 16, 'bold'), foreground=self.colors['accent'])
        result_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=10)

        ttk.Label(results_frame, text="Output File:", style='Info.TLabel').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.result_file_var = tk.StringVar(value="None")
        ttk.Label(results_frame, textvariable=self.result_file_var,
                  style='Info.TLabel').grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

        # Control buttons
        control_frame = tk.Frame(results_frame, bg=self.colors['white'])
        control_frame.grid(row=3, column=0, columnspan=2, pady=15)

        ttk.Button(control_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Open Output", command=self.open_output_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Exit", command=self.quit_app).pack(side=tk.LEFT, padx=5)

        # Log section
        log_frame = ttk.LabelFrame(panel, text="Processing Log",
                                   style='Section.TLabelframe', padding=15)
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=15, pady=15)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, width=60,
                                                  font=('Consolas', 9),
                                                  bg=self.colors['white'],
                                                  fg=self.colors['dark'])
        self.log_text.pack(fill=tk.BOTH, expand=True)

        return panel

    def create_progress_section(self, parent):
        """Create progress bar section"""
        progress_frame = tk.Frame(parent, bg=self.colors['light'])
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        progress_frame.columnconfigure(0, weight=1)

        ttk.Label(progress_frame, text="Processing Progress:", style='Info.TLabel').pack(anchor=tk.W)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                            maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Initialize mode display
        self.on_mode_change()

    def check_code_files(self):
        """Check if necessary code files exist"""
        if not os.path.exists(FEATURE_EXTRACT_SCRIPT):
            self.log(f"Error: Feature extraction script not found: {FEATURE_EXTRACT_SCRIPT}")
            messagebox.showerror("File Missing", f"Feature extraction script not found:\n{FEATURE_EXTRACT_SCRIPT}")
        else:
            self.log(f"Found feature extraction script: {FEATURE_EXTRACT_SCRIPT}")

    def on_mode_change(self):
        """Handle prediction mode switching"""
        if self.prediction_mode.get() == "single":
            self.single_frame.grid()
            self.batch_frame.grid_remove()
        else:
            self.single_frame.grid_remove()
            self.batch_frame.grid()

    def log(self, message):
        """Add log message"""
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def clear_log(self):
        """Clear log"""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared.")

    def browse_cif_file(self):
        """Browse CIF file"""
        filename = filedialog.askopenfilename(
            title="Select CIF File",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )
        if filename:
            self.cif_path_var.set(filename)
            self.current_cif_file = filename
            self.log(f"Selected CIF file: {os.path.basename(filename)}")

    def browse_cif_folder(self):
        """Browse CIF folder"""
        folder = filedialog.askdirectory(title="Select Folder Containing CIF Files")
        if folder:
            self.cif_folder_var.set(folder)
            self.current_cif_folder = folder
            # Count CIF files
            cif_count = len([f for f in os.listdir(folder) if f.endswith('.cif')])
            self.log(f"Selected folder: {os.path.basename(folder)} ({cif_count} CIF files found)")

    def browse_output_file(self):
        """Select output directory (file name will be auto-generated)"""
        folder = filedialog.askdirectory(
            title="Select Output Directory (filename will be auto-generated)"
        )
        if folder:
            # Always treat as directory and auto-generate filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder, f"batch_prediction_results_{timestamp}.xlsx")
            self.output_file_var.set(filename)
            self.log(f"Output directory selected: {folder}")
            self.log(f"Auto-generated filename: batch_prediction_results_{timestamp}.xlsx")
            self.log(f"Full path: {filename}")

    def start_single_prediction(self):
        """Start single file prediction"""
        if not self.validate_inputs():
            return

        if not self.current_cif_file:
            messagebox.showerror("Error", "Please select a CIF file")
            return

        self.log(f"Starting single prediction for: {os.path.basename(self.current_cif_file)}")
        self.log(f"Guest molecule: {self.guest_molecule.get()}")

        # Run prediction in new thread
        thread = threading.Thread(target=self.run_single_prediction)
        thread.daemon = True
        thread.start()

    def start_batch_prediction(self):
        """Start batch prediction"""
        if not self.validate_inputs():
            return

        if not self.current_cif_folder:
            messagebox.showerror("Error", "Please select a CIF folder")
            return

        if not self.output_file_var.get():
            messagebox.showerror("Error", "Please select an output file")
            return

        self.log(f"Starting batch prediction for folder: {os.path.basename(self.current_cif_folder)}")
        self.log(f"Guest molecule: {self.guest_molecule.get()}")

        # Run batch prediction in new thread
        thread = threading.Thread(target=self.run_batch_prediction)
        thread.daemon = True
        thread.start()

    def validate_inputs(self):
        """Validate inputs"""
        # Check for default ANN model paths first, then validate
        if not self.model_path or not os.path.exists(self.model_path):
            messagebox.showerror("Error",
                                 f"ANN model file not found.\nExpected location: {ANN_FOLDER}\\trained_model_ANN_1.pth")
            return False

        if not self.scaler_path or not os.path.exists(self.scaler_path):
            messagebox.showerror("Error", f"Scaler file not found.\nExpected location: {ANN_FOLDER}\\scaler.pkl")
            return False

        # Check if only methane is supported
        if self.guest_molecule.get() != "Methane":
            messagebox.showwarning("Limitation", "Currently only Methane is supported. Other molecules coming soon.")
            return False

        return True

    def run_single_prediction(self):
        """Run single file prediction"""
        try:
            self.progress_var.set(0)
            self.log("Starting single file prediction...")

            # 1. Feature extraction
            self.log("Extracting zeolite features...")
            self.progress_var.set(20)

            feature_result = self.extract_features_single(self.current_cif_file)
            if not feature_result:
                self.log("Feature extraction failed")
                return

            self.progress_var.set(60)

            # 2. ML prediction
            self.log("Running machine learning prediction...")
            prediction_result = self.predict_single(feature_result)

            self.progress_var.set(90)

            if prediction_result:
                self.prediction_var.set(f"{prediction_result:.6e} m²/s")
                # Show the output file name instead of feature file
                cif_name = os.path.splitext(os.path.basename(self.current_cif_file))[0]
                result_filename = f"{cif_name}_prediction_result.xlsx"
                self.result_file_var.set(result_filename)
                self.log(f"Prediction completed: {prediction_result:.6e} m²/s")
                messagebox.showinfo("Prediction Complete",
                                    f"Diffusion Coefficient: {prediction_result:.6e} m²/s\n" +
                                    f"Results saved to: {result_filename}")
            else:
                self.log("Prediction failed")
                self.prediction_var.set("Prediction failed")

            self.progress_var.set(100)

        except Exception as e:
            self.log(f"Single prediction failed: {e}")
            messagebox.showerror("Error", f"Single prediction failed:\n{e}")
            self.prediction_var.set("Error occurred")
        finally:
            self.progress_var.set(0)

    def run_batch_prediction(self):
        """Run batch prediction"""
        try:
            self.progress_var.set(0)
            self.log("Starting batch processing...")

            # 1. Batch feature extraction
            self.log("Extracting features from all CIF files...")
            self.progress_var.set(20)

            feature_file = self.extract_features_batch(self.current_cif_folder)
            if not feature_file:
                self.log("Batch feature extraction failed")
                return

            self.progress_var.set(60)

            # 2. Batch prediction
            self.log("Running batch machine learning predictions...")
            output_file = self.predict_batch(feature_file)

            self.progress_var.set(90)

            if output_file:
                self.result_file_var.set(output_file)
                self.log(f"Batch prediction completed. Results saved to: {output_file}")
                messagebox.showinfo("Batch Processing Complete",
                                    f"Results saved to:\n{output_file}")
                self.prediction_var.set("Batch processing complete")
            else:
                self.log("Batch prediction failed")
                self.prediction_var.set("Batch processing failed")

            self.progress_var.set(100)

        except Exception as e:
            self.log(f"Batch prediction failed: {e}")
            messagebox.showerror("Error", f"Batch prediction failed:\n{e}")
            self.prediction_var.set("Error occurred")
        finally:
            self.progress_var.set(0)

    def extract_features_single(self, cif_file):
        """Extract features from single file"""
        try:
            # Ensure temp folder is set up
            if not self.temp_folder:
                self._setup_temp_folder()

            # Create temporary output directory
            temp_dir = os.path.join(self.temp_folder, "single_extraction")
            os.makedirs(temp_dir, exist_ok=True)

            self.log(f"Processing CIF file: {os.path.basename(cif_file)}")

            # Create feature extraction script for single file
            script_content = self.create_feature_script(cif_file, temp_dir, is_batch=False)

            # Write temporary script
            temp_script = os.path.join(self.temp_folder, "temp_feature_extract.py")
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(script_content)

            # Run script
            result = subprocess.run([sys.executable, temp_script],
                                    capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:
                # Find generated Excel file
                excel_files = [f for f in os.listdir(temp_dir) if f.endswith('.xlsx')]
                if excel_files:
                    excel_path = os.path.join(temp_dir, excel_files[0])
                    self.log(f"Feature extraction completed: {excel_files[0]}")
                    return excel_path
                else:
                    self.log("No Excel output file found")
            else:
                self.log(f"Feature extraction error: {result.stderr}")

        except Exception as e:
            self.log(f"Feature extraction exception: {e}")

        return None

    def extract_features_batch(self, cif_folder):
        """Extract features from batch of files - Direct import method"""
        try:
            # Ensure temp folder is set up
            if not self.temp_folder:
                self._setup_temp_folder()

            # Validate input folder
            if not os.path.exists(cif_folder):
                self.log(f"Input folder does not exist: {cif_folder}")
                return None

            # Count CIF files for progress tracking
            cif_files = [f for f in os.listdir(cif_folder) if f.endswith('.cif')]
            if not cif_files:
                self.log(f"No CIF files found in {cif_folder}")
                return None

            self.log(f"Using direct import method for faster processing...")
            self.log(f"Processing {len(cif_files)} CIF files from: {cif_folder}")

            # Create temporary output file
            temp_output = os.path.join(self.temp_folder, "batch_features.xlsx")

            # 直接导入和调用批量处理器，避免subprocess开销
            try:
                # 确保CODE_FOLDER在Python路径中
                import sys
                if CODE_FOLDER not in sys.path:
                    sys.path.insert(0, CODE_FOLDER)

                # 直接导入BatchZeoliteProcessor
                from zeo_feature_extract import BatchZeoliteProcessor

                self.log(f"Successfully imported BatchZeoliteProcessor")

                # 创建批量处理器实例
                processor = BatchZeoliteProcessor(
                    input_folder=cif_folder,
                    output_file=temp_output
                )

                self.log(f"BatchZeoliteProcessor created")
                self.log(f"Input: {cif_folder}")
                self.log(f"Output: {temp_output}")

                # 运行批量处理
                processor.run_batch_processing()

                # 验证输出文件
                if os.path.exists(temp_output):
                    try:
                        import pandas as pd
                        df = pd.read_excel(temp_output)
                        self.log(f"Batch feature extraction completed successfully!")
                        self.log(f"enerated features for {len(df)} structures")
                        return temp_output
                    except Exception as e:
                        self.log(f"Output file verification failed: {e}")
                        return None
                else:
                    self.log(f"Output file not created: {temp_output}")
                    return None

            except ImportError as e:
                self.log(f"Failed to import BatchZeoliteProcessor: {e}")
                self.log(f"CODE_FOLDER: {CODE_FOLDER}")
                self.log(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

                # 回退到子进程方法
                return self._extract_features_batch_subprocess(cif_folder)

            except Exception as e:
                self.log(f"Direct import method failed: {e}")
                import traceback
                self.log(f"Traceback: {traceback.format_exc()}")

                # 回退到子进程方法
                return self._extract_features_batch_subprocess(cif_folder)

        except Exception as e:
            self.log(f"Batch feature extraction exception: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

        return None

    def _extract_features_batch_subprocess(self, cif_folder):
        """Fallback subprocess method for batch feature extraction"""
        self.log(f"Falling back to subprocess method...")

        try:
            # Ensure temp folder is set up
            if not self.temp_folder:
                self._setup_temp_folder()

            # Create temporary output file
            temp_output = os.path.join(self.temp_folder, "batch_features.xlsx")

            # Create batch extraction script
            script_content = self.create_feature_script(cif_folder, temp_output, is_batch=True)

            # Write temporary script
            temp_script = os.path.join(self.temp_folder, "temp_batch_extract.py")
            with open(temp_script, 'w', encoding='utf-8') as f:
                f.write(script_content)

            # Run script with timeout
            self.log("Running batch feature extraction script...")
            result = subprocess.run([sys.executable, temp_script],
                                    capture_output=True, text=True, encoding='utf-8',
                                    timeout=3600)  # 1 hour timeout

            # Log output
            if result.stdout:
                self.log("Script output (last 10 lines):")
                lines = result.stdout.split('\n')[-10:]
                for line in lines:
                    if line.strip():
                        self.log(f"{line}")

            if result.stderr:
                self.log("Script errors:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        self.log(f"ERROR: {line}")

            if result.returncode == 0 and os.path.exists(temp_output):
                self.log(f"Subprocess feature extraction completed")
                return temp_output
            else:
                self.log(f"Subprocess method failed with return code: {result.returncode}")
                return None

        except Exception as e:
            self.log(f"Subprocess method exception: {e}")
            return None

    def predict_single(self, feature_file):
        """Single file ANN prediction with result saving"""
        try:
            # Get prediction result
            prediction_result = self.predict_with_ann(feature_file, single=True)

            if prediction_result:
                # Create output file in the same directory as input CIF
                cif_dir = os.path.dirname(self.current_cif_file)
                cif_name = os.path.splitext(os.path.basename(self.current_cif_file))[0]
                output_file = os.path.join(cif_dir, f"{cif_name}_prediction_result.xlsx")

                # Save detailed result with features and prediction
                self.predict_with_ann(feature_file, single=False, output_file=output_file)

#                self.log(f"Detailed results saved to: {output_file}")

            return prediction_result

        except Exception as e:
            self.log(f"ANN prediction exception: {e}")
        return None

    def predict_batch(self, feature_file):
        """Batch ANN prediction"""
        try:
            output_file = self.output_file_var.get()

            # Validate inputs
            if not feature_file or not os.path.exists(feature_file):
                self.log(f"Feature file does not exist: {feature_file}")
                return None

            if not output_file:
                self.log(f"No output file specified")
                return None

            # Note: Output file path is now always auto-generated with timestamp in browse_output_file()
            self.log(f"Output file: {output_file}")

            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    self.log(f"Created output directory: {output_dir}")
                except Exception as e:
                    self.log(f"Cannot create output directory {output_dir}: {e}")
                    return None

            self.log(f"Starting batch ANN prediction...")
            self.log(f"Feature file: {feature_file}")
            self.log(f"Output file: {output_file}")
            self.log(f"Output file exists: {os.path.exists(output_file) if output_file else 'N/A'}")
            self.log(f"Output directory: {os.path.dirname(output_file) if output_file else 'N/A'}")

            # Use direct ANN prediction instead of script
            result = self.predict_with_ann(feature_file, single=False, output_file=output_file)

            if result and os.path.exists(output_file):
                # Verify output file
                try:
                    import pandas as pd
                    df = pd.read_excel(output_file)
#                    self.log(f"Batch ANN prediction completed: {os.path.basename(output_file)}")
                    self.log(f"Output contains {len(df)} predictions")
                    return output_file
                except Exception as e:
                    self.log(f"Output file verification failed: {e}")
                    return None
            else:
                self.log(f"Batch ANN prediction failed")
                return None

        except Exception as e:
            self.log(f"Batch ANN prediction exception: {e}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")

        return None

    def load_builtin_features(self, zeolite_name):
        try:
            builtin_file = PathConfig.BUILTIN_FEATURES_PATH

            # Log file status on first load
            if not hasattr(self, '_builtin_file_checked'):
                if os.path.exists(builtin_file):
                    self.log(f"Found builtin features file: {builtin_file}")
                else:
                    self.log(f"Builtin features file not found: {builtin_file}")
                self._builtin_file_checked = True

            if os.path.exists(builtin_file):
                df_builtin = pd.read_excel(builtin_file)

                # Find zeolite name column
                zeolite_col = None
                for col in ['Zeolites', 'zeolites', 'zeolite']:
                    if col in df_builtin.columns:
                        zeolite_col = col
                        break
                if zeolite_col is None:
                    zeolite_col = df_builtin.columns[0]

                # Find matching zeolite
                matches = df_builtin[df_builtin[zeolite_col].str.contains(zeolite_name, case=False, na=False)]

                if len(matches) > 0:
                    row = matches.iloc[0]
                    builtin_features = {}

                    # Load pre-computed features from built-in database
                    for feature in ['FDSi', 'PLD', 'LCD', 'PLD/LCD', 'Vacc', 'ASA']:
                        if feature in row and pd.notna(row[feature]):
                            builtin_features[feature] = row[feature]

                    return builtin_features

        except Exception as e:
            self.log(f"Failed to load built-in features: {str(e)}")

        return None

    def predict_with_ann(self, feature_file, single=True, output_file=None):
        """Core ANN prediction method"""
        try:
            # Load feature data
            df_test = pd.read_excel(feature_file)

            # For each zeolite, try to read pre-computed features from built-in table (GCD, PLD, LCD, Vacc)
            builtin_loaded_count = 0
            for idx, row in df_test.iterrows():
                if 'zeolites' in row:
                    zeolite_name = str(row['zeolites']).replace('.cif', '')
                    builtin_features = self.load_builtin_features(zeolite_name)

                    if builtin_features:
                        # 使用内置表格的预计算特征值
                        for feature, value in builtin_features.items():
                            if pd.notna(value):  # 只使用非NaN值
                                df_test.at[idx, feature] = value
                        builtin_loaded_count += 1
                        self.log(f"Loaded builtin features for {zeolite_name}: {list(builtin_features.keys())}")
                    else:
                        self.log(f"No builtin features found for {zeolite_name}")

            if builtin_loaded_count > 0:
                self.log(f"Successfully loaded builtin features for {builtin_loaded_count}/{len(df_test)} zeolites")

            # 特征已经是正确的列名，不需要映射
            
            # Calculate PLD/LCD if both PLD and LCD exist
            if 'PLD' in df_test.columns and 'LCD' in df_test.columns:
                df_test['PLD/LCD'] = df_test['PLD'] / df_test['LCD']
                self.log("Calculated PLD/LCD from PLD and LCD")

            # Required features for ANN (using ASA instead of density)
            feature_names = ModelConfig.FEATURE_NAMES

#            self.log(f"Available columns after processing: {list(df_test.columns)}")
#            self.log(f"Required features for ANN: {feature_names}")

            # Check if all required features exist
            missing_features = [f for f in feature_names if f not in df_test.columns]
            if missing_features:
                self.log(f"Missing features: {missing_features}")

            # Check for missing features and fill with 0
            for feature in feature_names:
                if feature not in df_test.columns:
                    df_test[feature] = 0.0
                    self.log(f"Warning: Missing feature {feature}, filled with 0")

            # Check data quality and handle missing values
#            self.log(f"Data shape before processing: {df_test.shape}")

            # For features that are missing, use reasonable defaults
            for feature in feature_names:
                if feature in df_test.columns:
                    nan_count = df_test[feature].isna().sum()
                    if nan_count > 0:
                        self.log(f"Feature '{feature}' has {nan_count} NaN values")

                        # Use reasonable defaults for different feature types
                        if feature in ['Tort']:
                            df_test[feature] = df_test[feature].fillna(1.0)
                        elif feature in ['AvgA', 'MaxA']:
                            df_test[feature] = df_test[feature].fillna(10.0)
                        elif feature in ['StdA']:
                            df_test[feature] = df_test[feature].fillna(5.0)
                        elif feature in ['Vacc']:
                            df_test[feature] = df_test[feature].fillna(0.3)
                        elif feature in ['PLD', 'PLD/LCD']:
                            df_test[feature] = df_test[feature].fillna(5.0)
                        elif feature in ['FDSi', 'ASA']:
                            df_test[feature] = df_test[feature].fillna(0.1)
                        else:
                            df_test[feature] = df_test[feature].fillna(0.0)

                        self.log(f"Filled {nan_count} NaN values in '{feature}' with defaults")

            # Drop rows with missing critical features
            critical_features = ['FDSi', 'ASA']
            df_test = df_test.dropna(subset=critical_features)

            if len(df_test) == 0:
                self.log("No valid data after removing rows with missing critical features")
                self.log(f"Critical features: {critical_features}")
                return None

#            self.log(f"Final data shape after processing: {df_test.shape}")

            # Extract features
            features = df_test[feature_names].values

            # Debug: Check feature dimensions
            # self.log(f"Debug: feature_names count = {len(feature_names)}")
            # self.log(f"Debug: features shape = {features.shape}")
            # self.log(f"Debug: Unique feature_names = {len(set(feature_names))}")

            # Print calculated features for prediction
            self.log("=" * 60)
            self.log("Calculated Features for Prediction:")
            self.log("=" * 60)
            for i, zeolite_name in enumerate(df_test['zeolites']):
                self.log(f"\nZeolite: {zeolite_name}")
                for j, feature_name in enumerate(feature_names):
                    feature_value = features[i, j]
                    self.log(f"  {feature_name}: {feature_value:.2f}")
            self.log("=" * 60)

            # Load scaler and transform features
            scaler = joblib.load(self.scaler_path)
            features_scaled = scaler.transform(features)

            # Convert to PyTorch tensor
            X_tensor = torch.tensor(features_scaled, dtype=torch.float32)

            # Load ANN model
            model = FNN(
                input_size=ModelConfig.INPUT_SIZE,
                hidden_size1=ModelConfig.HIDDEN_SIZE_1,
                hidden_size2=ModelConfig.HIDDEN_SIZE_2,
                hidden_size3=ModelConfig.HIDDEN_SIZE_3,
                output_size=ModelConfig.OUTPUT_SIZE
            )
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Make predictions
            with torch.no_grad():
                predictions = model(X_tensor)
                predictions_exp = np.exp(predictions.numpy())  # Convert from log space

            if single:
                # Return single prediction value (multiply by 1e-8 to restore original scale)
                result = float(predictions_exp[0, 0]) * 1e-8
                self.log(f"ANN prediction: {result:.6e} m²/s")
                return result
            else:
                # Save batch predictions to Excel (multiply by 1e-8 to restore original scale)
                df_test['Predicted_Ds (m2/s)'] = predictions_exp.flatten() * 1e-8

                # Select only the required columns for output
                output_columns = ['zeolites'] + feature_names + ['Predicted_Ds (m2/s)']
                # Keep only columns that exist in df_test
                output_columns = [col for col in output_columns if col in df_test.columns]
                df_output = df_test[output_columns]

                # Save to Excel with notes
#                self.log(f"Saving predictions to: {output_file}")
#                self.log(f"Directory: {os.path.dirname(output_file)}")
#                self.log(f"Is directory: {os.path.isdir(output_file)}")
#                self.log(f"Parent exists: {os.path.exists(os.path.dirname(output_file))}")

                # Ensure output_file is not a directory
                if os.path.isdir(output_file):
                    self.log(f"Error: output_file is a directory, not a file: {output_file}")
                    return None

                # Ensure parent directory exists
                parent_dir = os.path.dirname(output_file)
                if parent_dir and not os.path.exists(parent_dir):
                    try:
                        os.makedirs(parent_dir, exist_ok=True)
                        self.log(f"Created parent directory: {parent_dir}")
                    except Exception as e:
                        self.log(f"annot create parent directory {parent_dir}: {e}")
                        return None

                try:
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        df_output.to_excel(writer, sheet_name='Prediction_Results', index=False)

                    self.log(f"Predictions saved to: {output_file}")
                    return True

                except PermissionError as e:
                    self.log(f"Permission error saving to {output_file}: {e}")
                    # Try alternative filename
                    alt_output = output_file.replace('.xlsx', f'_backup_{int(time.time())}.xlsx')
                    self.log(f"Trying alternative filename: {alt_output}")
                    try:
                        with pd.ExcelWriter(alt_output, engine='openpyxl') as writer:
                            df_output.to_excel(writer, sheet_name='Prediction_Results', index=False)
                        self.log(f"Saved to alternative file: {alt_output}")
                        return True
                    except Exception as e2:
                        self.log(f"Alternative save also failed: {e2}")
                        return None

                except Exception as e:
                    self.log(f"Error saving Excel file: {e}")
                    return None

        except Exception as e:
            self.log(f"ANN prediction error: {e}")
            return None

    def create_feature_script(self, input_path, output_path, is_batch=False):
        """Create feature extraction script"""
        script_lines = [
            "import sys",
            "import os",
            "",
            f"sys.path.insert(0, {repr(CODE_FOLDER)})",
            "",
            "from zeo_feature_extract import "
            + ("BatchZeoliteProcessor" if is_batch else "ZeoliteFeatureExtractor"),
            "",
            "if __name__ == '__main__':",
            "    try:",
        ]

        if is_batch:
            script_lines += [
                "        processor = BatchZeoliteProcessor(",
                f"            input_folder={repr(input_path)},",
                f"            output_file={repr(output_path)}",
                "        )",
                "        processor.run_batch_processing()",
            ]
        else:
            script_lines += [
                "        extractor = ZeoliteFeatureExtractor(",
                f"            cif_file={repr(input_path)},",
                f"            output_dir={repr(output_path)},",
                "            auto_remove_symmetry=True",
                "        )",
                "        extractor.extract_all_features()",
            ]

        script_lines += [
            "        print('Feature extraction completed successfully')",
            "    except Exception as e:",
            "        print(f'Feature extraction failed: {e}')",
            "        import traceback",
            "        traceback.print_exc()",
            "        sys.exit(1)",
        ]

        return "\n".join(script_lines)

    def create_prediction_script(self, feature_file, output_file=None):
        """Create prediction script"""
        is_batch = output_file is not None
        data_selection = "df" if is_batch else "df.iloc[0:1]"
        result_output = f"df['Diffusion_Coefficient_m2s'] = y_pred\n    df.to_excel(r'{output_file}', index=False)" if is_batch else "print(f'Prediction result: {{y_pred[0]}}')"

        return

    def open_output_folder(self):
        """Open output folder"""
        if self.result_file_var.get() not in ["None", "无"]:
            folder = os.path.dirname(self.result_file_var.get())
            if os.path.exists(folder):
                os.startfile(folder)
                self.log(f"Opened folder: {folder}")
            else:
                messagebox.showwarning("Warning", "Output folder does not exist")
        else:
            messagebox.showinfo("Information", "No output file available")

    def quit_app(self):
        """Exit application"""
        try:
            # Clean up temporary folder
            if self.temp_folder and os.path.exists(self.temp_folder):
                shutil.rmtree(self.temp_folder)
                self.log("Temporary files cleaned up")
        except Exception as e:
            self.log(f"Failed to clean temp files: {e}")

        self.log("Application closing...")
        self.root.quit()
        self.root.destroy()


def main():
    """Main function"""
    # Create main window
    root = tk.Tk()

    # Create application
    app = ZeoliteGUI(root)

    try:
        # Run main loop
        root.mainloop()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program exited with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()