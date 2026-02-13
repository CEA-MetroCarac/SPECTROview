import struct
import numpy as np
import datetime

class SpcReader:
    """
    Reader for Galactic SPC files (New Format, 1996+).
    Based on the SPC file format specification.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.f = None
        self.header = {}
        self.subheaders = []
        self.x_data = None
        self.y_data = None
        self.log_content = ""
        
        # Read the file upon initialization
        self.read()
        
    def read(self):
        with open(self.filepath, 'rb') as f:
            self.f = f
            self._read_header()
            
            # Helper to check flags
            def check_flag(flag_byte, bit_mask):
                return (flag_byte & bit_mask) != 0

            # Determine data offsets
            # Structure: 
            # 1. Header (512 bytes)
            # 2. X Data (if TXVALS/0x80 is set): npts * 4 bytes floats
            # 3. Directories (if TMULTI/0x02 is set?? No, usually handled via random access, but here we scan linearly)
            # 4. Subfiles: [Subheader (32 bytes) + Y Data (npts * 4/2 bytes)] * fnsub
            
            has_txvals = check_flag(self.header['ftflgs'], 0x80) # 0x80 = TXVALS
            is_16bit = check_flag(self.header['ftflgs'], 0x10)   # 0x10 = TSPREC

            # Move to end of header
            self.f.seek(512)

            # Read X-axis
            if has_txvals:
                # Explicit X array exists
                x_bytes = self.f.read(self.header['npts'] * 4)
                self.x_data = np.frombuffer(x_bytes, dtype='<f4').astype(np.float64)
            else:
                # Generate linear X
                self.x_data = np.linspace(
                    self.header['first'], 
                    self.header['last'], 
                    self.header['npts']
                )

            # Read Subfiles (Y data)
            self.y_data = [] 
            self.subheaders = []
            
            for i in range(self.header['fnsub']):
                subheader = self._read_subheader()
                self.subheaders.append(subheader)
                
                npts = self.header['npts'] 
                
                # Logic restructure:
                # 1. Check Subheader Exp to decide format
                exponent = subheader['exp']
                
                # Special case: If exp == -128 (0x80), data is float32
                if exponent == -128:
                     is_float_override = True
                else:
                     is_float_override = False
                     
                if is_float_override or not is_16bit:
                     # Read 32-bit floats
                     raw_data = self.f.read(npts * 4)
                     y_chunk = np.frombuffer(raw_data, dtype='<f4').astype(np.float64)
                else:
                    # 16-bit integers
                    raw_data = self.f.read(npts * 2)
                    y_chunk = np.frombuffer(raw_data, dtype='<i2').astype(np.float64)
                    
                    if exponent > 127: exponent -= 256 # Just in case, though we fixed reading
                    scale_factor = 2.0 ** (exponent - 16)
                    y_chunk = y_chunk * scale_factor
                    
                self.y_data.append(y_chunk)
            
            if self.header['fnsub'] > 1:
                self.y_data = np.vstack(self.y_data)
            else:
                self.y_data = self.y_data[0]
                
            # Log Block
            # ...
            # Recalculate offset logic...
            
            real_bytes_per_point = 4 if (not is_16bit or (len(self.subheaders) > 0 and self.subheaders[0]['exp'] == -128)) else 2
            
            calculated_data_end = 512 + (self.header['npts'] * 4 if has_txvals else 0) + \
                                  self.header['fnsub'] * (32 + self.header['npts'] * real_bytes_per_point)

            self.f.seek(calculated_data_end)
            try:
                self.log_content = self.f.read().decode('utf-8', errors='ignore')
                self._parse_log_metadata()
            except:
                pass

    def _parse_log_metadata(self):
        """Parse Key=Value pairs from log block."""
        self.log_metadata = {}
        # Filter strings to avoid binary garbage
        import re
        # Pattern: look for printable keys followed by = and value
        # Allow alphanumeric, spaces, and typical punctuation in unit labels like (cm-1)
        # Reject keys with control chars or extended ASCII if it looks like garbage
        key_pattern = re.compile(r'^[A-Za-z0-9\s\.\(\)_\-\%\/]+$')
        
        for line in self.log_content.splitlines():
            line = line.strip()
            if not line: continue
            
            if '=' in line:
                parts = line.split('=', 1)
                key = parts[0].strip()
                val = parts[1].strip()
                
                # Heuristic: Key must match safe pattern and have reasonable length
                if len(key) < 64 and len(key) > 1 and key_pattern.match(key):
                     self.log_metadata[key] = val

    def _read_header(self):
        # Read the fixed 512-byte header
        # Using struct unpack
        # Structure references: https://github.com/rohanisaac/spc/blob/master/spc/spc.py
        # and standard SPC format docs
        
        self.f.seek(0)
        
        # We only unlock specific fields we typically need
        # ftflgs (1B), fversn (1B), fexper(1B), fexp(1B), npts(4B), first(8B), last(8B), fnsub(4B)
        # 0x00: ftflgs (byte), fversn (byte)
        # 0x02: fexper (byte), fexp (byte)
        # 0x04: npts (int32)
        # 0x08: first (double)
        # 0x10: last (double)
        # 0x18: fnsub (int32)
        # ...
        # 0x20: flogoff (int32)
        
        header_fmt = '<BBBBiddi' # Little endian
        # size = 1+1+1+1+4+8+8+4 = 28 bytes
        
        chunk = self.f.read(28)
        vals = struct.unpack(header_fmt, chunk)
        
        self.header = {
            'ftflgs': vals[0],
            'fversn': vals[1],
            'fexper': vals[2], # Instrument technique code
            'fexp': vals[3],   # scaling exponent (for 16-bit integer files)
            'npts': vals[4],   # Number of points per spectrum
            'first': vals[5],  # X coordinate of first point
            'last': vals[6],   # X coordinate of last point
            'fnsub': vals[7]   # Number of subfiles (spectra)
        }
        
        # Read flogoff at 0x20 (byte 32)
        self.f.seek(32)
        self.header['flogoff'] = struct.unpack('<i', self.f.read(4))[0]
        
        # Read fxtype, fytype, fztype, fpost at 0x24 (byte 36)
        # 4 bytes = 4 chars/bytes
        self.f.seek(36)
        vals = struct.unpack('<BBBB', self.f.read(4))
        self.header['fxtype'] = vals[0]
        self.header['fytype'] = vals[1]
        self.header['fztype'] = vals[2]
        self.header['fpost'] = vals[3]
        
        # Read fcatxt (X axis label) at 0x10E (270)
        # Size is 30 bytes
        self.f.seek(270)
        self.header['fcatxt'] = self.f.read(30).split(b'\0')[0].decode('utf-8', errors='ignore').strip()
        
        # Read axes labels (null terminated strings)
        # ... legacy comments ...
        # flogtxt (0xA0, 19 bytes)
        # fmods (0xB3, 4 bytes)
        # fprocs (0xB7, 1 byte)
        # flevel (0xB8, 1 byte)
        # fsampin (0xB9, 2 bytes)
        # ffactor (0xBB, 4 bytes float)
        # fmethod (0xBF, 48 bytes)
        # fzinc (0xEF, 4 bytes float)
        # fwplanes (0xF3, 4 bytes int)
        # fwinc (0xF7, 4 bytes float)
        # fwtype (0xFB, 1 byte)
        # fresv (0xFC, 187 bytes)
        
        # Let's read the comment and axis labels if possible
        # fcmnt starts at 96 (0x60)
        self.f.seek(96)
        self.header['fcmnt'] = self.f.read(130).split(b'\0')[0].decode('utf-8', errors='ignore').strip()
        
        # fcatxt (X axis unit) at 226 (0xE2) ?? No, check offset
        # The offset 0x82 is 130 decimal. But wait, 0x60 is 96. 96+130 = 226.
        # So fcmnt ends at 226.
        # fcatxt starts at 226? No, 0x60 is 96. Size is 130 bytes?
        # Use simpler offsets:
        # fcatxt seems to be at 226?
        # Actually, let's just grab the date which is useful.
        # fdate is at 0x28 (40). It is a 4 byte int (MMDDYYHH with top 4 bits for Y2K)
        self.f.seek(40)
        date_int = struct.unpack('<i', self.f.read(4))[0]
        # Parse date if non-zero
        self.header['date'] = str(date_int) # Placeholder
        
    def _read_subheader(self):
        # Subheader is 32 bytes
        # 0x00: subflgs (byte)
        # 0x01: subexp (byte) - exponent for Y values (if 16-bit)
        # 0x02: subindx (word) - index number
        # 0x04: subtime (float) - time of scan
        # 0x08: subnext (float) - time to next scan
        # 0x0C: subnois (float) - noise
        # 0x10: subnpts (long) - number of points (usually same as header npts, can be 0 meaning 'use header npts')
        # 0x14: subscan (long) - number of scans
        # 0x18: subwlevel (float) - W axis value (e.g. Z level for tomograms or Cycle for kinetics)
        # 0x1C: subresv (4 bytes)
        
        sh_fmt = '<Bbhfffiff'
        chunk = self.f.read(32)
        if not chunk:
            return {}
        vals = struct.unpack(sh_fmt, chunk[:28])
        
        sh = {
            'flags': vals[0],
            'exp': vals[1],
            'index': vals[2],
            'time': vals[3],
            'next': vals[4],
            'noise': vals[5],
            'npts': vals[6],
            'scans': vals[7],
            'wlevel': vals[8]
        }
        return sh
