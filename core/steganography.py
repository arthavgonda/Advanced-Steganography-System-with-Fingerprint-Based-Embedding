from PIL import Image
import numpy as np
import os
import struct
import jpegio
import zlib
import soundfile as sf
import hashlib
import secrets
import subprocess
import tempfile
from reedsolo import RSCodec
import torch

MAGIC = b'STEG'
SPLIT_MAGIC = b'SPLT'

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def secure_cleanup_tensor(tensor):
    if tensor is not None and tensor.device.type in ['cuda', 'mps']:
        tensor.zero_()
        del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

def generate_fingerprint(base_seed):
    return hashlib.sha256(base_seed.encode() if isinstance(base_seed, str) else base_seed).digest()[:16]

def prng_positions(total_positions, needed_positions, fingerprint_bytes):
    rng_seed = int.from_bytes(hashlib.sha256(fingerprint_bytes).digest()[:8], 'big')
    rng = np.random.RandomState(rng_seed % (2**32))
    if needed_positions > total_positions:
        raise ValueError(f"Need {needed_positions} positions but only {total_positions} available")
    positions = rng.choice(total_positions, size=needed_positions, replace=False)
    return sorted(positions)

def add_ecc(data_bytes, ecc_symbols=32):
    rs = RSCodec(ecc_symbols)
    return rs.encode(data_bytes)

def remove_ecc(encoded_bytes, ecc_symbols=32):
    rs = RSCodec(ecc_symbols)
    return rs.decode(encoded_bytes)[0]

def convert_to_wav(input_path, output_path):
    try:
        result = subprocess.run([
            'ffmpeg', '-i', input_path, '-ar', '44100', '-ac', '2',
            '-y', output_path
        ], capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception:
        return False

def get_file_capacity(file_path, file_type='auto'):
    if file_type == 'auto':
        file_type = detect_file_type(file_path)
    
    if file_type in ['image', 'png', 'jpg', 'jpeg', 'bmp', 'webp']:
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            try:
                jpeg = jpegio.read(file_path)
                total_coeffs = 0
                for comp in jpeg.coef_arrays:
                    rows, cols = comp.shape
                    for i in range(rows):
                        for j in range(cols):
                            coef = int(comp[i, j])
                            if abs(coef) > 1:
                                total_coeffs += 1
                return total_coeffs // 8
            except Exception:
                pass
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            return w * h * 3 * 2 // 8
    elif file_type in ['audio', 'mp3', 'wav', 'flac']:
        temp_wav = None
        needs_cleanup = False
        
        if not file_path.lower().endswith('.wav'):
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            if convert_to_wav(file_path, temp_wav):
                file_path = temp_wav
                needs_cleanup = True
            else:
                if temp_wav and os.path.exists(temp_wav):
                    os.remove(temp_wav)
                size = os.path.getsize(file_path)
                estimated_samples = size * 8
                num_segments = estimated_samples // 4096
                usable_bins = 400
                return (num_segments * usable_bins) // 8
        
        try:
            data, rate = sf.read(file_path, dtype='float32')
            if len(data.shape) == 2:
                data = (data[:, 0] + data[:, 1]) / 2
            segment_size = 4096
            hop_size = 1024
            num_segments = (len(data) - segment_size) // hop_size
            usable_bins = 400
            capacity = (num_segments * usable_bins) // 8
            return capacity
        except Exception:
            size = os.path.getsize(file_path)
            estimated_samples = size * 8
            num_segments = estimated_samples // 4096
            usable_bins = 400
            return (num_segments * usable_bins) // 8
        finally:
            if needs_cleanup and temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
    elif file_type in ['video', 'mp4', 'avi', 'mkv']:
        size = os.path.getsize(file_path)
        return size // 32
    elif file_type in ['text', 'json', 'xml', 'html', 'csv']:
        size = os.path.getsize(file_path)
        return size // 4
    else:
        size = os.path.getsize(file_path)
        return size // 8

def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif', '.tiff']
    audio_exts = ['.mp3', '.wav', '.flac', '.ogg', '.aac', '.m4a']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    text_exts = ['.txt', '.json', '.xml', '.html', '.csv', '.log']
    
    if ext in image_exts:
        return 'image'
    elif ext in audio_exts:
        return 'audio'
    elif ext in video_exts:
        return 'video'
    elif ext in text_exts:
        return 'text'
    else:
        return 'binary'

def f5_embed_gpu(cover_path, payload_bytes, output_path, fingerprint_bytes):
    jpeg = jpegio.read(cover_path)
    
    compressed_payload = zlib.compress(payload_bytes, level=9)
    use_compressed = len(compressed_payload) < len(payload_bytes)
    
    if use_compressed:
        final_payload = b'\x01' + compressed_payload
        print(f"Compressed: {len(payload_bytes)} -> {len(compressed_payload)} bytes ({len(compressed_payload)/len(payload_bytes)*100:.1f}%)")
    else:
        final_payload = b'\x00' + payload_bytes
    
    ecc_payload = add_ecc(final_payload, ecc_symbols=32)
    payload_bits_np = np.unpackbits(np.frombuffer(ecc_payload, dtype=np.uint8))
    needed_bits = len(payload_bits_np)
    
    coef_arrays = jpeg.coef_arrays
    
    usable_coeffs = []
    for comp_idx in range(len(coef_arrays)):
        comp = coef_arrays[comp_idx]
        rows, cols = comp.shape
        for i in range(rows):
            for j in range(cols):
                coef = int(comp[i, j])
                if abs(coef) > 1:
                    usable_coeffs.append((comp_idx, i, j, coef))
    
    if needed_bits > len(usable_coeffs):
        raise ValueError(f"Data too large: need {needed_bits} bits, capacity {len(usable_coeffs)} bits")
    
    positions = prng_positions(len(usable_coeffs), needed_bits, fingerprint_bytes)
    
    payload_bits = torch.from_numpy(payload_bits_np).to(DEVICE, dtype=torch.int32)
    positions_tensor = torch.tensor(positions, device=DEVICE, dtype=torch.int64)
    
    try:
        for idx in range(len(payload_bits)):
            pos = positions_tensor[idx].item()
            bit = payload_bits[idx].item()
            
            comp_idx, i, j, coef = usable_coeffs[pos]
            comp = coef_arrays[comp_idx]
            
            target_bit = int(bit)
            current_bit = int(abs(coef) % 2)
            
            if current_bit != target_bit:
                if coef > 0:
                    if coef % 2 == 0:
                        new_val = coef + 1
                    else:
                        new_val = coef - 1
                        if new_val == 0:
                            new_val = coef + 1
                else:
                    if abs(coef) % 2 == 0:
                        new_val = coef - 1
                    else:
                        new_val = coef + 1
                        if new_val == 0:
                            new_val = coef - 1
                
                comp[i, j] = new_val
    finally:
        secure_cleanup_tensor(payload_bits)
        secure_cleanup_tensor(positions_tensor)
    
    jpeg.coef_arrays = coef_arrays
    jpegio.write(jpeg, output_path)
    
    return needed_bits

def f5_extract_gpu(stego_path, fingerprint_bytes):
    jpeg = jpegio.read(stego_path)
    coef_arrays = jpeg.coef_arrays
    
    usable_coeffs = []
    for comp_idx in range(len(coef_arrays)):
        comp = coef_arrays[comp_idx]
        rows, cols = comp.shape
        for i in range(rows):
            for j in range(cols):
                coef = int(comp[i, j])
                if abs(coef) > 1:
                    usable_coeffs.append(coef)
    
    max_bytes = len(usable_coeffs) // 8
    
    usable_coeffs_np = np.array(usable_coeffs, dtype=np.int32)
    usable_coeffs_tensor = torch.from_numpy(usable_coeffs_np).to(DEVICE)
    
    try:
        for try_size in range(100, max_bytes + 1, 100):
            positions = prng_positions(len(usable_coeffs), try_size * 8, fingerprint_bytes)
            positions_tensor = torch.tensor(positions, device=DEVICE, dtype=torch.int64)
            
            try:
                selected_coeffs = usable_coeffs_tensor[positions_tensor]
                bits = torch.abs(selected_coeffs) % 2
                bits_np = bits.cpu().numpy().astype(np.uint8)
                
                payload_bytes = np.packbits(bits_np).tobytes()
                
                decoded = remove_ecc(payload_bytes, ecc_symbols=32)
                
                if decoded[0:1] == b'\x01':
                    decompressed = zlib.decompress(decoded[1:])
                    return decompressed
                elif decoded[0:1] == b'\x00':
                    return decoded[1:]
            except:
                continue
            finally:
                secure_cleanup_tensor(selected_coeffs)
                secure_cleanup_tensor(bits)
                secure_cleanup_tensor(positions_tensor)
    finally:
        secure_cleanup_tensor(usable_coeffs_tensor)
    
    raise ValueError("Could not extract valid data")

def phase_coding_embed_gpu(cover_path, payload_bytes, output_path, fingerprint_bytes):
    data, rate = sf.read(cover_path, dtype='float32')
    
    if len(data.shape) == 2:
        is_stereo = True
        left = data[:, 0]
        right = data[:, 1]
        mid = (left + right) / np.sqrt(2)
        side = (left - right) / np.sqrt(2)
        working_channel = mid
    else:
        is_stereo = False
        working_channel = data
    
    compressed_payload = zlib.compress(payload_bytes, level=9)
    use_compressed = len(compressed_payload) < len(payload_bytes)
    
    if use_compressed:
        final_payload = b'\x01' + compressed_payload
        print(f"Compressed: {len(payload_bytes)} -> {len(compressed_payload)} bytes")
    else:
        final_payload = b'\x00' + payload_bytes
    
    ecc_payload = add_ecc(final_payload, ecc_symbols=32)
    payload_bits_np = np.unpackbits(np.frombuffer(ecc_payload, dtype=np.uint8))
    
    win_len = 4096
    hop_len = 1024
    num_frames = (len(working_channel) - win_len) // hop_len
    
    bin_start = 50
    bin_end = 450
    usable_bins_per_frame = bin_end - bin_start
    
    total_positions = num_frames * usable_bins_per_frame
    needed_bits = len(payload_bits_np)
    
    if needed_bits > total_positions:
        raise ValueError(f"Audio too short: need {needed_bits} positions, have {total_positions}")
    
    positions = prng_positions(total_positions, needed_bits, fingerprint_bytes)
    
    window = np.hanning(win_len)
    
    working_channel_tensor = torch.from_numpy(working_channel).to(DEVICE, dtype=torch.float32)
    window_tensor = torch.from_numpy(window).to(DEVICE, dtype=torch.float32)
    payload_bits = torch.from_numpy(payload_bits_np).to(DEVICE, dtype=torch.int32)
    
    phase_delta = 0.2
    
    try:
        bit_idx = 0
        for pos in positions:
            frame_idx = pos // usable_bins_per_frame
            bin_offset = pos % usable_bins_per_frame
            bin_idx = bin_start + bin_offset
            
            frame_start = frame_idx * hop_len
            frame_end = frame_start + win_len
            
            if frame_end > len(working_channel):
                break
            
            frame = working_channel_tensor[frame_start:frame_end] * window_tensor
            
            spectrum = torch.fft.rfft(frame)
            magnitude = torch.abs(spectrum)
            phase = torch.angle(spectrum)
            
            magnitude_cpu = magnitude.cpu().numpy()
            flux = np.sum(np.abs(np.diff(magnitude_cpu)))
            if flux > np.mean(magnitude_cpu) * 5:
                continue
            
            if bin_idx >= len(phase) - 1:
                continue
            
            bit_val = payload_bits[bit_idx].item()
            
            if bit_val == 1:
                phase_diff = phase[bin_idx] - phase[bin_idx - 1]
                phase_diff += phase_delta
                phase[bin_idx] = phase[bin_idx - 1] + phase_diff
            
            spectrum_new = magnitude * torch.exp(1j * phase)
            frame_new = torch.fft.irfft(spectrum_new, n=win_len)
            
            working_channel_tensor[frame_start:frame_end] += (frame_new - frame.real) * window_tensor
            
            secure_cleanup_tensor(frame)
            secure_cleanup_tensor(spectrum)
            secure_cleanup_tensor(magnitude)
            secure_cleanup_tensor(phase)
            secure_cleanup_tensor(spectrum_new)
            secure_cleanup_tensor(frame_new)
            
            bit_idx += 1
            if bit_idx >= needed_bits:
                break
        
        modified_channel = working_channel_tensor.cpu().numpy()
        
        if is_stereo:
            left_new = (mid + side) / np.sqrt(2)
            right_new = (mid - side) / np.sqrt(2)
            output_data = np.stack([left_new, right_new], axis=1)
        else:
            output_data = modified_channel
    finally:
        secure_cleanup_tensor(working_channel_tensor)
        secure_cleanup_tensor(window_tensor)
        secure_cleanup_tensor(payload_bits)
    
    if output_path.lower().endswith('.flac'):
        sf.write(output_path, output_data, rate, format='FLAC', subtype='PCM_24')
    else:
        sf.write(output_path, output_data, rate, subtype='PCM_24')
    
    return needed_bits

def phase_coding_extract_gpu(stego_path, fingerprint_bytes):
    data, rate = sf.read(stego_path, dtype='float32')
    
    if len(data.shape) == 2:
        left = data[:, 0]
        right = data[:, 1]
        mid = (left + right) / np.sqrt(2)
        working_channel = mid
    else:
        working_channel = data
    
    win_len = 4096
    hop_len = 1024
    num_frames = (len(working_channel) - win_len) // hop_len
    
    bin_start = 50
    bin_end = 450
    usable_bins_per_frame = bin_end - bin_start
    
    total_positions = num_frames * usable_bins_per_frame
    
    max_bytes = min(total_positions // 8, 1000000)
    
    working_channel_tensor = torch.from_numpy(working_channel).to(DEVICE, dtype=torch.float32)
    window = np.hanning(win_len)
    window_tensor = torch.from_numpy(window).to(DEVICE, dtype=torch.float32)
    
    try:
        for try_size in range(1000, max_bytes + 1, 1000):
            try:
                positions = prng_positions(total_positions, try_size * 8, fingerprint_bytes)
                
                bits = []
                
                for pos in positions:
                    frame_idx = pos // usable_bins_per_frame
                    bin_offset = pos % usable_bins_per_frame
                    bin_idx = bin_start + bin_offset
                    
                    frame_start = frame_idx * hop_len
                    frame_end = frame_start + win_len
                    
                    if frame_end > len(working_channel):
                        bits.append(0)
                        continue
                    
                    frame = working_channel_tensor[frame_start:frame_end] * window_tensor
                    spectrum = torch.fft.rfft(frame)
                    phase = torch.angle(spectrum)
                    
                    if bin_idx >= len(phase) - 1:
                        bits.append(0)
                        secure_cleanup_tensor(frame)
                        secure_cleanup_tensor(spectrum)
                        secure_cleanup_tensor(phase)
                        continue
                    
                    phase_diff = (phase[bin_idx] - phase[bin_idx - 1]).cpu().item()
                    
                    if phase_diff > 0.1:
                        bits.append(1)
                    else:
                        bits.append(0)
                    
                    secure_cleanup_tensor(frame)
                    secure_cleanup_tensor(spectrum)
                    secure_cleanup_tensor(phase)
                
                bits_arr = np.array(bits, dtype=np.uint8)
                payload_bytes = np.packbits(bits_arr).tobytes()
                
                decoded = remove_ecc(payload_bytes, ecc_symbols=32)
                
                if decoded[0:1] == b'\x01':
                    decompressed = zlib.decompress(decoded[1:])
                    return decompressed
                elif decoded[0:1] == b'\x00':
                    return decoded[1:]
            except:
                continue
    finally:
        secure_cleanup_tensor(working_channel_tensor)
        secure_cleanup_tensor(window_tensor)
    
    raise ValueError("Could not extract valid audio data")

def h264_embed(cover_path, payload_bytes, output_path, fingerprint_bytes):
    temp_dir = tempfile.mkdtemp()
    raw_h264 = os.path.join(temp_dir, 'stream.h264')
    
    try:
        result = subprocess.run([
            'ffmpeg', '-i', cover_path, '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb',
            '-f', 'h264', raw_h264
        ], capture_output=True, timeout=60)
        
        if result.returncode != 0 or not os.path.exists(raw_h264):
            raise ValueError("Failed to extract H.264 stream")
        
        with open(raw_h264, 'rb') as f:
            bitstream = bytearray(f.read())
        
        compressed_payload = zlib.compress(payload_bytes, level=9)
        use_compressed = len(compressed_payload) < len(payload_bytes)
        
        if use_compressed:
            final_payload = b'\x01' + compressed_payload
        else:
            final_payload = b'\x00' + payload_bytes
        
        ecc_payload = add_ecc(final_payload, ecc_symbols=32)
        payload_bits_np = np.unpackbits(np.frombuffer(ecc_payload, dtype=np.uint8))
        
        nal_starts = []
        for i in range(len(bitstream) - 3):
            if bitstream[i:i+3] == b'\x00\x00\x01' or (i > 0 and bitstream[i-1:i+3] == b'\x00\x00\x00\x01'):
                nal_starts.append(i)
        
        modifiable_positions = []
        for idx in range(len(nal_starts) - 1):
            nal_start = nal_starts[idx]
            nal_end = nal_starts[idx + 1]
            nal_data = bitstream[nal_start:nal_end]
            
            start_code_len = 3
            if nal_start > 0 and bitstream[nal_start-1] == 0:
                start_code_len = 4
            
            if len(nal_data) < start_code_len + 10:
                continue
            
            nal_type = nal_data[start_code_len] & 0x1F
            
            if nal_type in [1, 5]:
                for offset in range(start_code_len + 5, len(nal_data) - 5):
                    pos = nal_start + offset
                    if bitstream[pos] != 0 and bitstream[pos] != 0xFF:
                        modifiable_positions.append(pos)
        
        if len(payload_bits_np) > len(modifiable_positions):
            raise ValueError(f"Video too small: need {len(payload_bits_np)} bits, have {len(modifiable_positions)} positions")
        
        positions = prng_positions(len(modifiable_positions), len(payload_bits_np), fingerprint_bytes)
        
        bitstream_np = np.frombuffer(bitstream, dtype=np.uint8).copy()
        payload_bits = torch.from_numpy(payload_bits_np).to(DEVICE, dtype=torch.int32)
        positions_tensor = torch.tensor(positions, device=DEVICE, dtype=torch.int64)
        
        try:
            for idx in range(len(payload_bits)):
                pos = modifiable_positions[positions_tensor[idx].item()]
                bit = payload_bits[idx].item()
                bitstream_np[pos] = (bitstream_np[pos] & 0xFE) | int(bit)
        finally:
            secure_cleanup_tensor(payload_bits)
            secure_cleanup_tensor(positions_tensor)
        
        modified_h264 = os.path.join(temp_dir, 'modified.h264')
        with open(modified_h264, 'wb') as f:
            f.write(bitstream_np.tobytes())
        
        result = subprocess.run([
            'ffmpeg', '-i', modified_h264, '-i', cover_path, '-map', '0:v',
            '-map', '1:a?', '-c:v', 'copy', '-c:a', 'copy', '-y', output_path
        ], capture_output=True, timeout=60)
        
        if result.returncode != 0:
            raise ValueError("Failed to remux video")
        
        return len(payload_bits_np)
        
    finally:
        for f in [raw_h264, modified_h264]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def h264_extract_gpu(stego_path, fingerprint_bytes):
    temp_dir = tempfile.mkdtemp()
    raw_h264 = os.path.join(temp_dir, 'stream.h264')
    
    try:
        result = subprocess.run([
            'ffmpeg', '-i', stego_path, '-c:v', 'copy', '-bsf:v', 'h264_mp4toannexb',
            '-f', 'h264', raw_h264
        ], capture_output=True, timeout=60)
        
        if result.returncode != 0 or not os.path.exists(raw_h264):
            raise ValueError("Failed to extract H.264 stream")
        
        with open(raw_h264, 'rb') as f:
            bitstream = f.read()
        
        nal_starts = []
        for i in range(len(bitstream) - 3):
            if bitstream[i:i+3] == b'\x00\x00\x01' or (i > 0 and bitstream[i-1:i+3] == b'\x00\x00\x00\x01'):
                nal_starts.append(i)
        
        modifiable_positions = []
        for idx in range(len(nal_starts) - 1):
            nal_start = nal_starts[idx]
            nal_end = nal_starts[idx + 1]
            nal_data = bitstream[nal_start:nal_end]
            
            start_code_len = 3
            if nal_start > 0 and bitstream[nal_start-1] == 0:
                start_code_len = 4
            
            if len(nal_data) < start_code_len + 10:
                continue
            
            nal_type = nal_data[start_code_len] & 0x1F
            
            if nal_type in [1, 5]:
                for offset in range(start_code_len + 5, len(nal_data) - 5):
                    pos = nal_start + offset
                    if bitstream[pos] != 0 and bitstream[pos] != 0xFF:
                        modifiable_positions.append(pos)
        
        max_bytes = len(modifiable_positions) // 8
        
        bitstream_np = np.frombuffer(bitstream, dtype=np.uint8)
        bitstream_tensor = torch.from_numpy(bitstream_np).to(DEVICE, dtype=torch.int32)
        
        try:
            for try_size in range(1000, max_bytes + 1, 1000):
                try:
                    positions = prng_positions(len(modifiable_positions), try_size * 8, fingerprint_bytes)
                    
                    actual_positions = [modifiable_positions[p] for p in positions]
                    positions_tensor = torch.tensor(actual_positions, device=DEVICE, dtype=torch.int64)
                    
                    try:
                        selected_bytes = bitstream_tensor[positions_tensor]
                        bits = selected_bytes & 1
                        bits_np = bits.cpu().numpy().astype(np.uint8)
                        
                        payload_bytes = np.packbits(bits_np).tobytes()
                        
                        decoded = remove_ecc(payload_bytes, ecc_symbols=32)
                        
                        if decoded[0:1] == b'\x01':
                            decompressed = zlib.decompress(decoded[1:])
                            return decompressed
                        elif decoded[0:1] == b'\x00':
                            return decoded[1:]
                    finally:
                        secure_cleanup_tensor(selected_bytes)
                        secure_cleanup_tensor(bits)
                        secure_cleanup_tensor(positions_tensor)
                except:
                    continue
        finally:
            secure_cleanup_tensor(bitstream_tensor)
        
        raise ValueError("Could not extract valid video data")
        
    finally:
        if os.path.exists(raw_h264):
            os.remove(raw_h264)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

def embed_in_image_gpu(cover_path, payload_bytes, output_path, bit_position="lsb", fingerprint_bytes=None):
    if fingerprint_bytes is None:
        fingerprint_bytes = secrets.token_bytes(16)
    
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        return f5_embed_gpu(cover_path, payload_bytes, output_path, fingerprint_bytes)
    
    img = Image.open(cover_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)
    
    payload_bits_np = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    
    flat_pixels = pixels.flatten()
    
    if len(payload_bits_np) > len(flat_pixels) * 2:
        raise ValueError("Image too small for payload")
    
    positions = prng_positions(len(flat_pixels), len(payload_bits_np), fingerprint_bytes)
    
    bit_masks = {
        'lsb': 0xFE,
        'mid': 0xFB,
        'msb': 0x7F
    }
    bit_shifts = {
        'lsb': 0,
        'mid': 2,
        'msb': 7
    }
    
    mask = bit_masks.get(bit_position, 0xFE)
    shift = bit_shifts.get(bit_position, 0)
    
    flat_pixels_tensor = torch.from_numpy(flat_pixels).to(DEVICE, dtype=torch.int32)
    payload_bits = torch.from_numpy(payload_bits_np).to(DEVICE, dtype=torch.int32)
    positions_tensor = torch.tensor(positions, device=DEVICE, dtype=torch.int64)
    
    try:
        for idx in range(len(payload_bits)):
            pos = positions_tensor[idx].item()
            bit = payload_bits[idx].item()
            flat_pixels_tensor[pos] = (flat_pixels_tensor[pos] & mask) | (int(bit) << shift)
        
        stego_pixels_np = flat_pixels_tensor.cpu().numpy().astype(np.uint8).reshape(pixels.shape)
    finally:
        secure_cleanup_tensor(flat_pixels_tensor)
        secure_cleanup_tensor(payload_bits)
        secure_cleanup_tensor(positions_tensor)
    
    stego_img = Image.fromarray(stego_pixels_np, 'RGB')
    stego_img.save(output_path, quality=95)
    
    return len(payload_bits_np)

def extract_from_image_gpu(stego_path, bit_position="lsb", fingerprint_bytes=None):
    if fingerprint_bytes is None:
        raise ValueError("Fingerprint required for extraction")
    
    if stego_path.lower().endswith(('.jpg', '.jpeg')):
        return f5_extract_gpu(stego_path, fingerprint_bytes)
    
    img = Image.open(stego_path).convert('RGB')
    pixels = np.array(img, dtype=np.uint8)
    flat_pixels = pixels.flatten()
    
    bit_shifts = {
        'lsb': 0,
        'mid': 2,
        'msb': 7
    }
    shift = bit_shifts.get(bit_position, 0)
    
    max_bytes = len(flat_pixels) * 2 // 8
    
    flat_pixels_tensor = torch.from_numpy(flat_pixels).to(DEVICE, dtype=torch.int32)
    
    try:
        for try_size in range(100, max_bytes + 1, 100):
            positions = prng_positions(len(flat_pixels), try_size * 8, fingerprint_bytes)
            positions_tensor = torch.tensor(positions, device=DEVICE, dtype=torch.int64)
            
            try:
                selected_pixels = flat_pixels_tensor[positions_tensor]
                bits = (selected_pixels >> shift) & 1
                bits_np = bits.cpu().numpy().astype(np.uint8)
                
                payload = np.packbits(bits_np).tobytes()
                
                if payload[:4] == MAGIC:
                    return payload
            finally:
                secure_cleanup_tensor(selected_pixels)
                secure_cleanup_tensor(bits)
                secure_cleanup_tensor(positions_tensor)
    finally:
        secure_cleanup_tensor(flat_pixels_tensor)
    
    raise ValueError("No hidden data found")

def embed_in_audio(cover_path, payload_bytes, output_path, fingerprint_bytes=None):
    if fingerprint_bytes is None:
        fingerprint_bytes = secrets.token_bytes(16)
    
    needs_conversion = False
    temp_wav = None
    
    if not cover_path.lower().endswith('.wav'):
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        if convert_to_wav(cover_path, temp_wav):
            cover_path = temp_wav
            needs_conversion = True
        else:
            raise ValueError("Could not convert audio to WAV")
    
    if not output_path.lower().endswith(('.wav', '.flac')):
        output_path = os.path.splitext(output_path)[0] + '.flac'
        print(f"Output will be saved as FLAC: {output_path}")
    
    try:
        result = phase_coding_embed_gpu(cover_path, payload_bytes, output_path, fingerprint_bytes)
        return result
    except Exception as e:
        print(f"ERROR embedding: {e}")
        raise
    finally:
        if needs_conversion and temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

def extract_from_audio(stego_path, fingerprint_bytes=None):
    if fingerprint_bytes is None:
        raise ValueError("Fingerprint required for extraction")
    
    if not stego_path.lower().endswith(('.wav', '.flac')):
        raise ValueError("Extraction only works with WAV/FLAC files")
    
    return phase_coding_extract_gpu(stego_path, fingerprint_bytes)

def embed_in_video(cover_path, payload_bytes, output_path, fingerprint_bytes=None):
    if fingerprint_bytes is None:
        fingerprint_bytes = secrets.token_bytes(16)
    
    return h264_embed(cover_path, payload_bytes, output_path, fingerprint_bytes)

def extract_from_video(stego_path, fingerprint_bytes=None):
    if fingerprint_bytes is None:
        raise ValueError("Fingerprint required for extraction")
    
    return h264_extract_gpu(stego_path, fingerprint_bytes)

def embed_in_text(cover_path, payload_bytes, output_path):
    with open(cover_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    
    lines = text.split('\n')
    bit_idx = 0
    
    for i in range(len(lines)):
        if bit_idx >= len(payload_bits):
            break
        if payload_bits[bit_idx] == 1:
            lines[i] = lines[i] + ' '
        bit_idx += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def extract_from_text(stego_path):
    with open(stego_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    lines = text.split('\n')
    bits = []
    
    for line in lines[:14*8]:
        if line.endswith(' '):
            bits.append(1)
        else:
            bits.append(0)
    
    bits_arr = np.array(bits, dtype=np.uint8)
    header_bytes = np.packbits(bits_arr)
    
    if bytes(header_bytes[:4]) != MAGIC:
        raise ValueError("No hidden data found")
    
    file_size = int.from_bytes(bytes(header_bytes[4:12]), 'big')
    name_len = int.from_bytes(bytes(header_bytes[12:14]), 'big')
    total_len = 14 + name_len + file_size
    
    bits = []
    for i, line in enumerate(lines[:total_len*8]):
        if line.endswith(' '):
            bits.append(1)
        else:
            bits.append(0)
    
    bits_arr = np.array(bits, dtype=np.uint8)
    payload = np.packbits(bits_arr)
    
    return bytes(payload)

def split_file(file_path, num_parts):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    part_size = (file_size + num_parts - 1) // num_parts
    
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = min(start + part_size, file_size)
        part_data = data[start:end]
        parts.append(part_data)
    
    return parts

def create_split_payload(part_data, part_num, total_parts, filename, fingerprint):
    header = SPLIT_MAGIC
    header += fingerprint
    header += total_parts.to_bytes(2, 'big')
    header += part_num.to_bytes(2, 'big')
    header += len(part_data).to_bytes(8, 'big')
    header += len(filename).to_bytes(2, 'big')
    header += filename
    
    payload = header + part_data
    return payload

def parse_split_payload(payload):
    if payload[:4] != SPLIT_MAGIC:
        raise ValueError("Not a split file payload")
    
    offset = 4
    fingerprint = payload[offset:offset+16]
    offset += 16
    total_parts = int.from_bytes(payload[offset:offset+2], 'big')
    offset += 2
    part_num = int.from_bytes(payload[offset:offset+2], 'big')
    offset += 2
    part_size = int.from_bytes(payload[offset:offset+8], 'big')
    offset += 8
    name_len = int.from_bytes(payload[offset:offset+2], 'big')
    offset += 2
    filename = payload[offset:offset+name_len].decode('utf-8')
    offset += name_len
    part_data = payload[offset:offset+part_size]
    
    return {
        'fingerprint': fingerprint,
        'total_parts': total_parts,
        'part_num': part_num,
        'filename': filename,
        'data': part_data
    }

def embed_universal(cover_path, secret_path, output_path, bit_position="lsb", fingerprint_bytes=None):
    if fingerprint_bytes is None:
        fingerprint_bytes = secrets.token_bytes(16)
    
    with open(secret_path, 'rb') as f:
        secret_bytes = f.read()
    
    filename = os.path.basename(secret_path).encode('utf-8')
    header = MAGIC
    header += len(secret_bytes).to_bytes(8, 'big')
    header += len(filename).to_bytes(2, 'big')
    header += filename
    payload = header + secret_bytes
    
    file_type = detect_file_type(cover_path)
    
    if file_type == 'image':
        embed_in_image_gpu(cover_path, payload, output_path, bit_position, fingerprint_bytes)
        method = "F5 Algorithm" if output_path.lower().endswith(('.jpg', '.jpeg')) else "Fingerprint LSB"
        print(f"Embedded {len(secret_bytes)} bytes using {method}: {output_path}")
    elif file_type == 'audio':
        embed_in_audio(cover_path, payload, output_path, fingerprint_bytes)
        print(f"Embedded {len(secret_bytes)} bytes using Advanced Phase Coding: {output_path}")
    elif file_type == 'video':
        embed_in_video(cover_path, payload, output_path, fingerprint_bytes)
        print(f"Embedded {len(secret_bytes)} bytes in H.264 video: {output_path}")
    elif file_type == 'text':
        embed_in_text(cover_path, payload, output_path)
        print(f"Embedded {len(secret_bytes)} bytes in text: {output_path}")
    else:
        embed_in_audio(cover_path, payload, output_path, fingerprint_bytes)
        print(f"Embedded {len(secret_bytes)} bytes in binary file: {output_path}")

def extract_universal(stego_path, output_folder='.', bit_position="lsb", fingerprint_bytes=None):
    file_type = detect_file_type(stego_path)
    
    if file_type == 'image':
        payload = extract_from_image_gpu(stego_path, bit_position, fingerprint_bytes)
    elif file_type == 'audio':
        payload = extract_from_audio(stego_path, fingerprint_bytes)
    elif file_type == 'video':
        payload = extract_from_video(stego_path, fingerprint_bytes)
    elif file_type == 'text':
        payload = extract_from_text(stego_path)
    else:
        payload = extract_from_audio(stego_path, fingerprint_bytes)
    
    offset = 4
    file_size = int.from_bytes(payload[offset:offset+8], 'big')
    offset += 8
    name_len = int.from_bytes(payload[offset:offset+2], 'big')
    offset += 2
    filename = payload[offset:offset+name_len].decode('utf-8')
    offset += name_len
    file_data = payload[offset:offset+file_size]
    
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, filename)
    with open(out_path, 'wb') as f:
        f.write(file_data)
    
    print(f"Extracted {len(file_data)} bytes to: {out_path}")
    return out_path