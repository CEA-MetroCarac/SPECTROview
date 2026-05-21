# spectroview/model/workspace_io.py
import io
import json
import zipfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class WorkspaceIO:
    """Core persistence utility for high-performance ZIP-based workspace storage."""
    
    @staticmethod
    def save_workspace(file_path: str, metadata: dict, arrays: dict = None, dataframes: dict = None):
        """Save workspace to a compressed ZIP archive.
        
        Args:
            file_path: Absolute destination file path.
            metadata: Lightweight settings/GUI/peak dictionary.
            arrays: Dict of numpy arrays to store in arrays.npz.
            dataframes: Dict of pandas DataFrames to store in dataframes.pkl.
        """
        # Ensure parent directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Open ZIP file with standard DEFLATE compression
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            # 1. Serialize lightweight metadata to JSON
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            # 2. Serialize numpy arrays using fast uncompressed np.savez 
            # (let ZIP handle compression for maximum I/O performance)
            if arrays:
                array_buf = io.BytesIO()
                np.savez(array_buf, **arrays)
                zf.writestr('arrays.npz', array_buf.getvalue())
                
            # 3. Serialize DataFrames using highly optimized python pickle protocol
            if dataframes:
                df_buf = io.BytesIO()
                pickle.dump(dataframes, df_buf, protocol=pickle.HIGHEST_PROTOCOL)
                zf.writestr('dataframes.pkl', df_buf.getvalue())

    @staticmethod
    def load_workspace(file_path: str) -> tuple:
        """Load workspace from a compressed ZIP archive.
        
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
            # Signal loader to fall back to legacy JSON parser
            return None, None, None, True
            
        metadata = {}
        arrays = {}
        dataframes = {}
        
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            
            # 1. Load metadata JSON
            if 'metadata.json' in names:
                metadata = json.loads(zf.read('metadata.json').decode('utf-8'))
                
            # 2. Load arrays NPZ
            if 'arrays.npz' in names:
                array_data = io.BytesIO(zf.read('arrays.npz'))
                # Must load with allow_pickle=False for security and speed
                with np.load(array_data, allow_pickle=False) as npz:
                    arrays = {k: npz[k] for k in npz.files}
                    
            # 3. Load DataFrames pickle
            if 'dataframes.pkl' in names:
                df_data = io.BytesIO(zf.read('dataframes.pkl'))
                dataframes = pickle.load(df_data)
                
        return metadata, arrays, dataframes, False
