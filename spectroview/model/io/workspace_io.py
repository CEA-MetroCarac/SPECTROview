# spectroview/model/workspace_io.py
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

class WorkspaceIO:
    """Core persistence utility for high-performance ZIP-based workspace storage."""
    
    @staticmethod
    def save_workspace(file_path: str, metadata: dict, arrays: dict = None, dataframes: dict = None):
        """Save workspace to a compressed ZIP archive in direct-streaming format (v2).
        
        Args:
            file_path: Absolute destination file path.
            metadata: Lightweight settings/GUI/peak dictionary.
            arrays: Dict of numpy arrays to store.
            dataframes: Dict of pandas DataFrames to store.
        """
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Open ZIP file with standard DEFLATE compression
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            # 1. Serialize lightweight metadata to JSON
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # 2. Serialize numpy arrays by streaming them directly to zip as individual .npy files
            # (let ZIP handle compression for maximum I/O performance and minimal RAM footprint)
            if arrays:
                for name, arr in arrays.items():
                    with zf.open(f'arrays/{name}.npy', 'w', force_zip64=True) as f:
                        np.save(f, arr, allow_pickle=False)
                
            # 3. Serialize DataFrames using highly optimized Parquet format streamed directly to zip
            if dataframes:
                for name, df in dataframes.items():
                    with zf.open(f'dataframes/{name}.parquet', 'w', force_zip64=True) as f:
                        df.to_parquet(f, index=False)

    @staticmethod
    def load_workspace(file_path: str) -> tuple:
        """Load workspace from a compressed ZIP archive (v2) or fallback to legacy raw JSON (v1).
        
        Sniffs magic bytes to determine if the file is a ZIP archive or a legacy raw JSON.
        
        Args:
            file_path: Absolute path of file to load.
            
        Returns:
            tuple: (metadata, arrays, dataframes, is_legacy)
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Sniff first 4 magic bytes
        with open(path, 'rb') as f:
            magic = f.read(4)
            
        # Standard ZIP magic number is 0x04034b50 ("PK\x03\x04")
        if magic != b'PK\x03\x04':
            # Signal loader to fall back to legacy JSON parser (v1)
            return None, None, None, True
            
        metadata = {}
        arrays = {}
        dataframes = {}
        
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            
            # 1. Load metadata JSON
            if 'metadata.json' in names:
                metadata = json.loads(zf.read('metadata.json').decode('utf-8'))
                
            # 2. Load arrays: support flat .npy stream
            for name in names:
                if name.startswith('arrays/') and name.endswith('.npy'):
                    arr_name = name[len('arrays/'):-len('.npy')]
                    with zf.open(name, 'r') as f:
                        arrays[arr_name] = np.load(f, allow_pickle=False)
                    
            # 3. Load DataFrames: support Parquet streams
            for name in names:
                if name.startswith('dataframes/') and name.endswith('.parquet'):
                    df_name = name[len('dataframes/'):-len('.parquet')]
                    with zf.open(name, 'r') as f:
                        dataframes[df_name] = pd.read_parquet(f)
                
        return metadata, arrays, dataframes, False
