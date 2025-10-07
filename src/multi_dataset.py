"""
Multi-dataset support for threat detection.
Supports UNSW-NB15, CICIDS2017, and NSL-KDD datasets.
"""
import argparse
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class DatasetAdapter:
    """Base class for dataset adapters."""
    
    def __init__(self, name: str):
        self.name = name
        self.raw_dir = RAW_DIR / name
        self.processed_dir = PROCESSED_DIR / name
        
    def load_raw(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load raw dataset. Returns (train_df, test_df) or (full_df, None)."""
        raise NotImplementedError
        
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and structure."""
        df = df.copy()
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        return df
        
    def extract_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and binary label (0=normal, 1=attack)."""
        raise NotImplementedError


class CICIDS2017Adapter(DatasetAdapter):
    """Adapter for CICIDS2017 dataset."""
    
    def __init__(self):
        super().__init__("cicids2017")
        
    def load_raw(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load CICIDS2017 dataset."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Look for CSV files
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.raw_dir}. "
                "Download CICIDS2017 dataset and place CSV files in this directory."
            )
        
        # Combine all CSV files
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, low_memory=False)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df, None
        
    def extract_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract label from CICIDS2017 dataset."""
        df = self.standardize(df)
        
        # CICIDS2017 typically has 'label' column with attack types or 'BENIGN'
        if 'label' not in df.columns:
            raise KeyError("'label' column not found in CICIDS2017 dataset")
        
        # Convert to binary: BENIGN=0, others=1
        y = (df['label'].str.upper() != 'BENIGN').astype(int)
        
        # Drop label and non-feature columns
        X = df.drop(columns=['label'])
        drop_cols = [c for c in ['timestamp', 'flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port']
                     if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
            
        return X, y


class NSLKDDAdapter(DatasetAdapter):
    """Adapter for NSL-KDD dataset."""
    
    def __init__(self):
        super().__init__("nsl_kdd")
        
    def load_raw(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load NSL-KDD dataset."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = self.raw_dir / "KDDTrain+.txt"
        test_file = self.raw_dir / "KDDTest+.txt"
        
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training file not found at {train_file}. "
                "Download NSL-KDD dataset and place KDDTrain+.txt and KDDTest+.txt in this directory."
            )
        
        # NSL-KDD column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        train_df = pd.read_csv(train_file, names=columns, header=None)
        test_df = pd.read_csv(test_file, names=columns, header=None) if test_file.exists() else None
        
        return train_df, test_df
        
    def extract_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract label from NSL-KDD dataset."""
        df = self.standardize(df)
        
        if 'attack_type' not in df.columns:
            raise KeyError("'attack_type' column not found in NSL-KDD dataset")
        
        # Convert to binary: normal=0, others=1
        y = (df['attack_type'].str.lower() != 'normal').astype(int)
        
        # Drop label and difficulty columns
        X = df.drop(columns=['attack_type'])
        if 'difficulty' in X.columns:
            X = X.drop(columns=['difficulty'])
            
        return X, y


def get_adapter(dataset_name: str) -> DatasetAdapter:
    """Get appropriate dataset adapter."""
    adapters = {
        'cicids2017': CICIDS2017Adapter,
        'nsl_kdd': NSLKDDAdapter,
    }
    
    if dataset_name not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(adapters.keys())}")
    
    return adapters[dataset_name]()


def prepare_dataset(
    dataset_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    output_name: str = None,
) -> None:
    """
    Prepare a dataset for training.
    
    Args:
        dataset_name: Name of dataset (cicids2017, nsl_kdd)
        test_size: Test set size if splitting is needed
        random_state: Random seed
        output_name: Output directory name (defaults to dataset_name)
    """
    adapter = get_adapter(dataset_name)
    
    # Load raw data
    print(f"Loading {dataset_name} dataset...")
    train_df, test_df = adapter.load_raw()
    
    # Standardize
    train_df = adapter.standardize(train_df)
    if test_df is not None:
        test_df = adapter.standardize(test_df)
    
    # Clean
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.drop_duplicates()
    
    if test_df is not None:
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.drop_duplicates()
    else:
        # Split data
        X, y = adapter.extract_label(train_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        train_df = X_train.copy()
        train_df['label'] = y_train.values
        test_df = X_test.copy()
        test_df['label'] = y_test.values
    
    # Save processed data
    output_dir = PROCESSED_DIR / (output_name or dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved processed train to {train_path}")
    print(f"Saved processed test to {test_path}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Print label distribution
    if 'label' in train_df.columns:
        train_dist = train_df['label'].value_counts()
        print(f"\nTrain label distribution:")
        print(f"  Normal (0): {train_dist.get(0, 0)}")
        print(f"  Attack (1): {train_dist.get(1, 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare additional intrusion detection datasets")
    parser.add_argument("--dataset", choices=["cicids2017", "nsl_kdd"], required=True,
                        help="Dataset to prepare")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test size if splitting is needed")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output-name", type=str,
                        help="Output directory name (defaults to dataset name)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        dataset_name=args.dataset,
        test_size=args.test_size,
        random_state=args.random_state,
        output_name=args.output_name,
    )
