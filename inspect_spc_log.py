import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from spectroview.model.m_spc import SpcReader

def inspect_log(filepath):
    print(f"--- Inspecting {filepath} ---")
    if not os.path.exists(filepath):
        print("File not found.")
        return

    try:
        reader = SpcReader(filepath)
        flogoff = reader.header.get('flogoff')
        file_size = os.path.getsize(filepath)
        print(f"File Size: {file_size}")
        print(f"Log Offset (flogoff): {flogoff}")
        
        # Access the calculated offset (need to re-calculate or expose it)
        # We can just look at where the reader seeked if we trust it, but we want to debug.
        # Let's manually calculate using the reader's properties.
        
        has_txvals = (reader.header['ftflgs'] & 0x80) != 0
        is_16bit = (reader.header['ftflgs'] & 0x10) != 0
        real_bytes_per_point = 4 if (not is_16bit or (len(reader.subheaders) > 0 and reader.subheaders[0]['exp'] == -128)) else 2
        
        calc_offset = 512 + (reader.header['npts'] * 4 if has_txvals else 0) + \
                      reader.header['fnsub'] * (32 + reader.header['npts'] * real_bytes_per_point)
        
        print(f"Calculated Offset: {calc_offset}")
        
        # Read around calculated offset
        with open(filepath, 'rb') as f:
            f.seek(max(0, calc_offset - 100))
            data = f.read(200)
            print(f"Content around {calc_offset} (-100 to +100): {repr(data)}")
            
        # Check specifically for ACQ. TIME
        print(f"Log Content Start Bytes: {[ord(c) for c in reader.log_content[:20]]}")
        acq_index = reader.log_content.find("ACQ. TIME")
        if acq_index != -1:
            print(f"FOUND 'ACQ. TIME' at index {acq_index} in log_content")
            print(f"Context: {repr(reader.log_content[acq_index-20:acq_index+50])}")
        else:
            print("String 'ACQ. TIME' NOT FOUND in log_content")
            
        # Check if Key is in metadata
        target_key = "ACQ. TIME (S)"
        if target_key in reader.log_metadata:
            print(f"SUCCESS: Metadata contains '{target_key}': '{reader.log_metadata[target_key]}'")
        else:
            print(f"FAILURE: Metadata MISSING '{target_key}'")
            print("Keys found: ", list(reader.log_metadata.keys())[:5])
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_log("examples/spectroscopic_data/MoS2_Raman.spc")
    inspect_log("examples/spectroscopic_data/MoS2_PL.spc")
