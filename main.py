import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'io_utils'))

import argparse
import math
import glob
import secrets
import hashlib
from PIL import Image

from steganography import (embed_universal, extract_universal, get_file_capacity, detect_file_type,
                                 split_file, create_split_payload, parse_split_payload, embed_in_image_gpu,
                                 embed_in_audio, generate_fingerprint, MAGIC)
from file_handler import normalize_path
from crypto import BitcoinStyleCrypto

def generate_random_filename(extension):
    random_hex = secrets.token_hex(16)
    return f"{random_hex}{extension}"

def verify_file_created(filepath, min_size=1024):
    if not os.path.exists(filepath):
        return False, f"File not created: {filepath}"
    
    size = os.path.getsize(filepath)
    if size < min_size:
        return False, f"File too small ({size} bytes): {filepath}"
    
    return True, "OK"

def cmd_keygen(args):
    crypto = BitcoinStyleCrypto()
    private_key, public_key = crypto.generate_keypair()
    
    private_path = normalize_path(args.private) if args.private else 'keys/my_private.pem'
    public_path = normalize_path(args.public) if args.public else 'keys/my_public.pem'
    
    password = None
    if args.password:
        password = input("Enter password to protect private key (or press Enter for none): ").strip()
        password = password if password else None
    
    crypto.save_private_key(private_key, private_path, password)
    crypto.save_public_key(public_key, public_path)
    
    fingerprint = crypto.get_key_fingerprint(public_key)
    
    print("\nKey pair generated")
    print(f"Private: {private_path}")
    print(f"Public:  {public_path}")
    print(f"Fingerprint: {fingerprint}")

def cmd_embed(args):
    cover = normalize_path(args.cover)
    secret = normalize_path(args.secret)
    
    if args.split:
        return cmd_embed_split(args)
    
    if not os.path.exists(cover):
        print(f"Error: Cover file not found: {cover}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(secret):
        print(f"Error: Secret file not found: {secret}", file=sys.stderr)
        sys.exit(1)
    
    with open(secret, 'rb') as f:
        secret_data = f.read()
    
    if len(secret_data) == 0:
        print("Error: Secret file is empty", file=sys.stderr)
        sys.exit(1)
    
    fingerprint_seed = secrets.token_hex(16)
    fingerprint_bytes = generate_fingerprint(fingerprint_seed)
    
    if args.encrypt:
        if not args.recipient:
            print("Error: --recipient required with --encrypt", file=sys.stderr)
            sys.exit(1)
        
        recipient_key = normalize_path(args.recipient)
        
        if not os.path.exists(recipient_key):
            print(f"Error: Recipient public key not found: {recipient_key}", file=sys.stderr)
            sys.exit(1)
        
        crypto = BitcoinStyleCrypto()
        
        try:
            recipient_public = crypto.load_public_key(recipient_key)
        except Exception as e:
            print(f"Error loading recipient public key: {e}", file=sys.stderr)
            sys.exit(1)
        
        print("Encrypting with Bitcoin secp256k1...")
        secret_data = crypto.encrypt_file(secret_data, recipient_public)
        print(f"Encrypted: {len(secret_data)} bytes ({len(secret_data)/1024:.1f} KB)")
        
        encrypted_fingerprint = crypto.encrypt_file(fingerprint_seed.encode(), recipient_public)
        
        fingerprint_path = os.path.join(os.path.dirname(args.out) if args.out else 'images/stego_output', '.fingerprint.enc')
        os.makedirs(os.path.dirname(fingerprint_path) or '.', exist_ok=True)
        with open(fingerprint_path, 'wb') as f:
            f.write(encrypted_fingerprint)
        try:
            os.chmod(fingerprint_path, 0o600)
        except:
            pass
        print(f"Encrypted fingerprint: {fingerprint_path}")
    else:
        fingerprint_path = os.path.join(os.path.dirname(args.out) if args.out else 'images/stego_output', '.fingerprint')
        os.makedirs(os.path.dirname(fingerprint_path) or '.', exist_ok=True)
        with open(fingerprint_path, 'w') as f:
            f.write(fingerprint_seed)
        print(f"Fingerprint: {fingerprint_path}")
    
    capacity = get_file_capacity(cover)
    needed = len(secret_data)
    
    if needed > capacity:
        print(f"\n⚠️  ERROR: File too large for single cover!")
        print(f"   Need: {needed:,} bytes ({needed/1024:.1f} KB)")
        print(f"   Capacity: {capacity:,} bytes ({capacity/1024:.1f} KB)")
        
        num_files = math.ceil(needed / capacity)
        print(f"\n💡 SOLUTION: Use --split flag to distribute across multiple covers")
        print(f"   You need {num_files} cover files")
        print(f"\nExample:")
        print(f"   python3 main.py embed --cover '{cover}' --secret '{secret}' --split {num_files}")
        if args.encrypt:
            print(f"   (Add --encrypt --recipient '{args.recipient}' for encryption)")
        sys.exit(1)
    
    if not args.out:
        file_type = detect_file_type(cover)
        if file_type == 'image':
            if cover.lower().endswith(('.jpg', '.jpeg')):
                out = 'images/stego_output/stego.jpg'
            else:
                out = 'images/stego_output/stego.png'
        elif file_type == 'audio':
            out = 'images/stego_output/stego.flac'
        elif file_type == 'video':
            out = 'images/stego_output/stego.mp4'
        else:
            out = 'images/stego_output/stego.bin'
    else:
        out = normalize_path(args.out)
    
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    
    temp_file = '/tmp/temp_secret.bin'
    with open(temp_file, 'wb') as f:
        f.write(secret_data)
    
    file_type = detect_file_type(cover)
    
    if file_type == 'image' and not out.lower().endswith(('.jpg', '.jpeg')):
        print("\nBit position:")
        print("1 - LSB (least visible)")
        print("2 - Middle")
        print("3 - MSB (most visible)")
        choice = input("Choice (1/2/3): ").strip()
        bit_position = {"1": "lsb", "2": "mid", "3": "msb"}.get(choice, "lsb")
    else:
        bit_position = "lsb"
    
    try:
        embed_universal(cover, temp_file, out, bit_position=bit_position, fingerprint_bytes=fingerprint_bytes)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during embedding: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def cmd_embed_split(args):
    cover = normalize_path(args.cover)
    secret = normalize_path(args.secret)
    num_parts = args.split
    
    if not os.path.exists(cover):
        print(f"Error: Cover file not found: {cover}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(secret):
        print(f"Error: Secret file not found: {secret}", file=sys.stderr)
        sys.exit(1)
    
    with open(secret, 'rb') as f:
        secret_data = f.read()
    
    fingerprint_seed = secrets.token_hex(16)
    fingerprint_bytes = generate_fingerprint(fingerprint_seed)
    
    if args.encrypt:
        if not args.recipient:
            print("Error: --recipient required with --encrypt", file=sys.stderr)
            sys.exit(1)
        
        recipient_key = normalize_path(args.recipient)
        crypto = BitcoinStyleCrypto()
        recipient_public = crypto.load_public_key(recipient_key)
        
        print("Encrypting with Bitcoin secp256k1...")
        secret_data = crypto.encrypt_file(secret_data, recipient_public)
        print(f"Encrypted: {len(secret_data)} bytes")
        
        encrypted_fingerprint = crypto.encrypt_file(fingerprint_seed.encode(), recipient_public)
    
    print(f"Splitting {len(secret_data)} bytes into {num_parts} parts...")
    
    temp_encrypted = '/tmp/temp_encrypted.bin'
    with open(temp_encrypted, 'wb') as f:
        f.write(secret_data)
    
    parts = split_file(temp_encrypted, num_parts)
    
    capacity = get_file_capacity(cover)
    
    filename = os.path.basename(secret).encode('utf-8')
    
    output_dir = normalize_path(args.out) if args.out else 'images/stego_output'
    os.makedirs(output_dir, exist_ok=True)
    
    file_type = detect_file_type(cover)
    
    if file_type == 'image':
        if cover.lower().endswith(('.jpg', '.jpeg')):
            ext = '.jpg'
        else:
            ext = '.png'
    elif file_type == 'audio':
        ext = '.flac'
    else:
        ext = os.path.splitext(cover)[1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.encrypt:
        fingerprint_path = os.path.join(output_dir, '.fingerprint.enc')
        with open(fingerprint_path, 'wb') as f:
            f.write(encrypted_fingerprint)
        try:
            os.chmod(fingerprint_path, 0o600)
        except:
            pass
    else:
        fingerprint_path = os.path.join(output_dir, '.fingerprint')
        with open(fingerprint_path, 'w') as f:
            f.write(fingerprint_seed)
    
    print(f"Fingerprint saved: {fingerprint_path}")
    
    output_files = []
    
    for i, part_data in enumerate(parts):
        payload = create_split_payload(part_data, i, num_parts, filename, fingerprint_bytes)
        
        if len(payload) > capacity:
            print(f"\n⚠️  ERROR: Part {i} is {len(payload)} bytes but capacity is {capacity} bytes")
            print(f"   Try using a larger cover file or more parts (--split {num_parts + 10})")
            sys.exit(1)
        
        stego_name = generate_random_filename(ext)
        stego_path = os.path.join(output_dir, stego_name)
        
        temp_payload = f'/tmp/part_{i}.bin'
        with open(temp_payload, 'wb') as f:
            f.write(payload)
        
        try:
            embed_universal(cover, temp_payload, stego_path, fingerprint_bytes=fingerprint_bytes)
            output_files.append(stego_path)
            print(f"Part {i+1}/{num_parts}: {stego_path}")
        except Exception as e:
            print(f"Error embedding part {i}: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if os.path.exists(temp_payload):
                os.remove(temp_payload)
    
    if os.path.exists(temp_encrypted):
        os.remove(temp_encrypted)
    
    print(f"\n✓ Success! Created {len(output_files)} stego files in {output_dir}")
    print(f"✓ Fingerprint: {fingerprint_path}")

def cmd_merge(args):
    input_pattern = normalize_path(args.input)
    
    if os.path.isdir(input_pattern):
        pattern = os.path.join(input_pattern, '*')
    else:
        pattern = input_pattern
    
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith('.fingerprint') and not f.endswith('.fingerprint.enc')]
    
    if not files:
        print(f"Error: No files found matching pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(files)} stego files")
    
    fingerprint_path = None
    if args.fingerprint:
        fingerprint_path = normalize_path(args.fingerprint)
    else:
        dir_path = os.path.dirname(input_pattern) if not os.path.isdir(input_pattern) else input_pattern
        enc_fp = os.path.join(dir_path, '.fingerprint.enc')
        plain_fp = os.path.join(dir_path, '.fingerprint')
        
        if os.path.exists(enc_fp):
            fingerprint_path = enc_fp
        elif os.path.exists(plain_fp):
            fingerprint_path = plain_fp
    
    if not fingerprint_path or not os.path.exists(fingerprint_path):
        print("Error: Fingerprint file not found", file=sys.stderr)
        sys.exit(1)
    
    with open(fingerprint_path, 'rb') as f:
        fingerprint_data = f.read()
    
    is_encrypted = fingerprint_path.endswith('.enc')
    
    if is_encrypted:
        if not args.private:
            print("Error: --private required for encrypted fingerprint", file=sys.stderr)
            sys.exit(1)
        
        private_key_path = normalize_path(args.private)
        
        if not os.path.exists(private_key_path):
            print(f"Error: Private key not found: {private_key_path}", file=sys.stderr)
            sys.exit(1)
        
        crypto = BitcoinStyleCrypto()
        
        password = None
        if args.password:
            password = input("Private key password (or press Enter): ").strip()
            password = password if password else None
        
        try:
            private_key = crypto.load_private_key(private_key_path, password)
        except Exception as e:
            print(f"Error loading private key: {e}", file=sys.stderr)
            sys.exit(1)
        
        try:
            fingerprint_seed = crypto.decrypt_file(fingerprint_data, private_key).decode()
        except Exception as e:
            print(f"Error decrypting fingerprint: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            fingerprint_seed = fingerprint_data.decode()
        except:
            print("Error: Fingerprint file appears corrupted", file=sys.stderr)
            sys.exit(1)
    
    fingerprint_bytes = generate_fingerprint(fingerprint_seed)
    
    print(f"Using fingerprint: {fingerprint_path}")
    
    parts = {}
    total_parts = None
    filename = None
    
    for stego_file in files:
        try:
            file_type = detect_file_type(stego_file)
            
            if file_type == 'image':
                payload = extract_universal(stego_file, bit_position='lsb', fingerprint_bytes=fingerprint_bytes)
            elif file_type == 'audio':
                from steganography import extract_from_audio
                payload = extract_from_audio(stego_file, fingerprint_bytes)
            elif file_type == 'video':
                from steganography import extract_from_video
                payload = extract_from_video(stego_file, fingerprint_bytes)
            else:
                continue
            
            info = parse_split_payload(payload)
            
            if total_parts is None:
                total_parts = info['total_parts']
                filename = info['filename']
            
            parts[info['part_num']] = info['data']
            print(f"Extracted part {info['part_num']}/{info['total_parts']} from {os.path.basename(stego_file)}")
            
        except Exception as e:
            print(f"Warning: Could not extract from {stego_file}: {e}")
            continue
    
    if not parts:
        print("Error: No valid parts extracted", file=sys.stderr)
        sys.exit(1)
    
    if len(parts) != total_parts:
        print(f"Warning: Expected {total_parts} parts but found {len(parts)}", file=sys.stderr)
    
    merged_data = b''.join([parts[i] for i in sorted(parts.keys())])
    
    output_dir = normalize_path(args.out) if args.out else 'data/extracted'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    if args.decrypt:
        if not args.private:
            print("Error: --private required with --decrypt", file=sys.stderr)
            sys.exit(1)
        
        private_key_path = normalize_path(args.private)
        crypto = BitcoinStyleCrypto()
        
        password = None
        if args.password:
            password = input("Private key password (or press Enter): ").strip()
            password = password if password else None
        
        try:
            private_key = crypto.load_private_key(private_key_path, password)
        except Exception as e:
            print(f"Error loading private key: {e}", file=sys.stderr)
            sys.exit(1)
        
        print("Decrypting...")
        try:
            merged_data = crypto.decrypt_file(merged_data, private_key)
        except Exception as e:
            print(f"Error decrypting: {e}", file=sys.stderr)
            sys.exit(1)
    
    with open(output_path, 'wb') as f:
        f.write(merged_data)
    
    print(f"\n✓ Merged file: {output_path}")
    print(f"✓ Size: {len(merged_data):,} bytes ({len(merged_data)/1024:.1f} KB)")

def cmd_extract(args):
    stego = normalize_path(args.stego)
    
    if not os.path.exists(stego):
        print(f"Error: Stego file not found: {stego}", file=sys.stderr)
        sys.exit(1)
    
    fingerprint_path = None
    dir_path = os.path.dirname(stego)
    
    enc_fp = os.path.join(dir_path, '.fingerprint.enc')
    plain_fp = os.path.join(dir_path, '.fingerprint')
    
    if os.path.exists(enc_fp):
        fingerprint_path = enc_fp
    elif os.path.exists(plain_fp):
        fingerprint_path = plain_fp
    else:
        print("Error: Fingerprint file not found", file=sys.stderr)
        sys.exit(1)
    
    with open(fingerprint_path, 'rb') as f:
        fingerprint_data = f.read()
    
    is_encrypted = fingerprint_path.endswith('.enc')
    
    if is_encrypted:
        if not args.private:
            print("Error: --private required for encrypted fingerprint", file=sys.stderr)
            sys.exit(1)
        
        private_key_path = normalize_path(args.private)
        crypto = BitcoinStyleCrypto()
        
        password = None
        if args.password:
            password = input("Private key password (or press Enter): ").strip()
            password = password if password else None
        
        private_key = crypto.load_private_key(private_key_path, password)
        fingerprint_seed = crypto.decrypt_file(fingerprint_data, private_key).decode()
    else:
        fingerprint_seed = fingerprint_data.decode()
    
    fingerprint_bytes = generate_fingerprint(fingerprint_seed)
    
    output_folder = normalize_path(args.out) if args.out else 'data/extracted'
    
    file_type = detect_file_type(stego)
    
    if file_type == 'image':
        bit_position = 'lsb'
    else:
        bit_position = None
    
    try:
        extracted_path = extract_universal(stego, output_folder, bit_position=bit_position, fingerprint_bytes=fingerprint_bytes)
        
        if args.decrypt:
            if not args.private:
                print("Error: --private required with --decrypt", file=sys.stderr)
                sys.exit(1)
            
            private_key_path = normalize_path(args.private)
            crypto = BitcoinStyleCrypto()
            
            password = None
            if args.password:
                password = input("Private key password (or press Enter): ").strip()
                password = password if password else None
            
            private_key = crypto.load_private_key(private_key_path, password)
            
            with open(extracted_path, 'rb') as f:
                encrypted_data = f.read()
            
            print("Decrypting...")
            decrypted_data = crypto.decrypt_file(encrypted_data, private_key)
            
            decrypted_path = extracted_path + '.decrypted'
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            print(f"Decrypted: {decrypted_path}")
            os.remove(extracted_path)
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during extraction: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_info(args):
    cover = normalize_path(args.cover)
    if not os.path.exists(cover):
        print("Error: File not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        file_type = detect_file_type(cover)
        cap = get_file_capacity(cover)
        
        print(f"File: {cover}")
        print(f"Type: {file_type}")
        print(f"Size: {os.path.getsize(cover):,} bytes ({os.path.getsize(cover)/1024:.2f} KB)")
        print(f"Capacity: {cap:,} bytes ({cap/1024:.2f} KB / {cap/1024/1024:.2f} MB)")
        
        if args.secret:
            secret = normalize_path(args.secret)
            if os.path.exists(secret):
                secret_size = os.path.getsize(secret)
                print(f"\nSecret file: {secret}")
                print(f"Secret size: {secret_size:,} bytes ({secret_size/1024:.2f} KB)")
                
                if secret_size <= cap:
                    print(f"✓ Will fit! ({secret_size/cap*100:.1f}% of capacity)")
                else:
                    parts_needed = math.ceil(secret_size / cap)
                    print(f"✗ Too large! Need to split into {parts_needed} parts")
                    print(f"  Use: --split {parts_needed}")
        
    except Exception as e:
        print(f"Error analyzing file: {e}", file=sys.stderr)
        sys.exit(1)

def cmd_sign(args):
    file_path = normalize_path(args.file)
    private_key_path = normalize_path(args.private)
    sig_path = normalize_path(args.signature) if args.signature else file_path + '.sig'
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(private_key_path):
        print(f"Error: Private key not found: {private_key_path}", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs(os.path.dirname(sig_path) or '.', exist_ok=True)
    
    crypto = BitcoinStyleCrypto()
    
    password = None
    if args.password:
        password = input("Private key password (or press Enter): ").strip()
        password = password if password else None
    
    try:
        private_key = crypto.load_private_key(private_key_path, password)
    except Exception as e:
        print(f"Error loading private key: {e}", file=sys.stderr)
        sys.exit(1)
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print("Signing...")
    signature = crypto.sign_data(data, private_key)
    
    with open(sig_path, 'wb') as f:
        f.write(signature)
    
    print(f"Signature: {sig_path}")

def cmd_verify(args):
    file_path = normalize_path(args.file)
    sig_path = normalize_path(args.signature)
    public_key_path = normalize_path(args.public)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(sig_path):
        print(f"Error: Signature file not found: {sig_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(public_key_path):
        print(f"Error: Public key not found: {public_key_path}", file=sys.stderr)
        sys.exit(1)
    
    crypto = BitcoinStyleCrypto()
    
    try:
        public_key = crypto.load_public_key(public_key_path)
    except Exception as e:
        print(f"Error loading public key: {e}", file=sys.stderr)
        sys.exit(1)
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    with open(sig_path, 'rb') as f:
        signature = f.read()
    
    print("Verifying...")
    is_valid = crypto.verify_signature(data, signature, public_key)
    
    if is_valid:
        print("VALID signature")
        fingerprint = crypto.get_key_fingerprint(public_key)
        print(f"Signed by: {fingerprint}")
    else:
        print("INVALID signature")
        sys.exit(1)

def build_parser():
    p = argparse.ArgumentParser(
        description="Universal Bitcoin Steganography with F5, Phase Coding, and H.264",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Check capacity:
    %(prog)s info --cover photo.jpg --secret document.pdf
  
  Embed (single file):
    %(prog)s embed --cover photo.jpg --secret small.txt --encrypt --recipient bob_public.pem
  
  Embed (split with random names):
    %(prog)s embed --cover photo.jpg --secret large.pdf --split 200 --encrypt --recipient bob_public.pem
  
  Merge split files:
    %(prog)s merge --input 'output_dir/' --fingerprint 'output_dir/.fingerprint' --decrypt --private my_private.pem
        """
    )
    sub = p.add_subparsers(dest='cmd', help='Command to execute')
    
    pk = sub.add_parser('keygen', help='Generate SECP256K1 key pair')
    pk.add_argument('--private', help='Private key output path')
    pk.add_argument('--public', help='Public key output path')
    pk.add_argument('--password', action='store_true', help='Password-protect private key')
    
    pe = sub.add_parser('embed', help='Hide file in cover(s)')
    pe.add_argument('--cover', required=True, help='Cover file path')
    pe.add_argument('--secret', required=True, help='File to hide')
    pe.add_argument('--out', help='Output path or directory')
    pe.add_argument('--split', type=int, help='Split into N parts with random names')
    pe.add_argument('--encrypt', action='store_true', help='Encrypt before hiding')
    pe.add_argument('--recipient', help='Recipient public key')
    
    pm = sub.add_parser('merge', help='Merge split stego files')
    pm.add_argument('--input', required=True, help='Directory or pattern for split files')
    pm.add_argument('--fingerprint', help='Fingerprint file path')
    pm.add_argument('--out', help='Output directory')
    pm.add_argument('--decrypt', action='store_true', help='Decrypt after merging')
    pm.add_argument('--private', help='Private key for decryption')
    pm.add_argument('--password', action='store_true', help='Private key has password')
    
    px = sub.add_parser('extract', help='Extract from single stego file')
    px.add_argument('--stego', required=True, help='Stego file path')
    px.add_argument('--out', help='Output folder')
    px.add_argument('--decrypt', action='store_true', help='Decrypt after extraction')
    px.add_argument('--private', help='Private key')
    px.add_argument('--password', action='store_true', help='Private key has password')
    
    pi = sub.add_parser('info', help='Show capacity')
    pi.add_argument('--cover', required=True, help='Cover file')
    pi.add_argument('--secret', help='Check if secret fits')
    
    ps = sub.add_parser('sign', help='Sign file')
    ps.add_argument('--file', required=True, help='File to sign')
    ps.add_argument('--private', required=True, help='Private key')
    ps.add_argument('--signature', help='Output signature')
    ps.add_argument('--password', action='store_true', help='Private key has password')
    
    pv = sub.add_parser('verify', help='Verify signature')
    pv.add_argument('--file', required=True, help='File to verify')
    pv.add_argument('--signature', required=True, help='Signature file')
    pv.add_argument('--public', required=True, help='Signer public key')
    
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    
    if args.cmd == 'keygen':
        cmd_keygen(args)
    elif args.cmd == 'embed':
        cmd_embed(args)
    elif args.cmd == 'merge':
        cmd_merge(args)
    elif args.cmd == 'extract':
        cmd_extract(args)
    elif args.cmd == 'info':
        cmd_info(args)
    elif args.cmd == 'sign':
        cmd_sign(args)
    elif args.cmd == 'verify':
        cmd_verify(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()