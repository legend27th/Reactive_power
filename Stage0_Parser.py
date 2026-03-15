import struct
import glob
import os

input_dir = r"D:\Hackathon\dsn_raw_data"
output_bin = r"D:\Hackathon\voyager_baseband.bin"

def string_to_hex(value):
    """
    Convert string representation into a clean hex string.
    Handles cases like:
    '0x3F800000', '"3F800000"', '3F800000,' etc.
    """
    value = value.strip()              # remove whitespace
    value = value.replace(",", "")     # remove commas
    value = value.replace('"', "")     # remove quotes
    value = value.replace("0x", "")    # remove hex prefix if present
    return value.upper()


def process_data():
    
    file_pattern = os.path.join(input_dir, "dsn_capture_*.txt")
    file_list = sorted(glob.glob(file_pattern))

    if not file_list:
        print("No files found. Check your input directory path.")
        return

    print(f"Found {len(file_list)} files. Starting memory-efficient conversion...")

    with open(output_bin, 'wb') as out_f:
        for filepath in file_list:
            filename = os.path.basename(filepath)
            print(f"Processing: {filename}")

            with open(filepath, 'r') as in_f:
                for line in in_f:

                    parts = line.strip().split()

                    if len(parts) == 2:

                        # STEP 1: Convert string → clean hex
                        hex_i = string_to_hex(parts[0])
                        hex_q = string_to_hex(parts[1])

                        # STEP 2: hex → raw bytes
                        i_bytes = bytes.fromhex(hex_i)
                        q_bytes = bytes.fromhex(hex_q)

                        # STEP 3: unpack big-endian IEEE754 float
                        i_val = struct.unpack('>f', i_bytes)[0]
                        q_val = struct.unpack('>f', q_bytes)[0]

                        # STEP 4: store as little-endian floats
                        out_f.write(struct.pack('<f', i_val))
                        out_f.write(struct.pack('<f', q_val))

    print(f"\nSuccess! Compact binary file saved to: {output_bin}")


if __name__ == "__main__":
    process_data()