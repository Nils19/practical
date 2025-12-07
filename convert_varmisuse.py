#!/usr/bin/env python3
"""
Convert VarMisuse proto files to JSON format that tf-gnn-samples expects.
This reads the binary .gz files and converts them to .json.gz format.
"""
import gzip
import json
import os
import sys
from pathlib import Path

def convert_proto_to_json(proto_file, json_file):
    """
    Attempt to convert a proto file to JSON.
    This is a best-effort conversion - the actual format depends on the proto schema.
    """
    try:
        # Try reading as gzipped binary
        with gzip.open(proto_file, 'rb') as f:
            data = f.read()
        
        # The proto files need the actual protobuf schema to parse correctly
        # Without the .proto definition, we can't deserialize them properly
        print(f"Error: {proto_file} - Cannot convert without protobuf schema")
        return False
        
    except Exception as e:
        print(f"Error processing {proto_file}: {e}")
        return False

def main():
    varmisuse_dir = Path("/dss/dsshome1/lxc0B/apdl017/paper-05/nils/bottleneck/tf-gnn-samples/data/varmisuse")
    
    print("VarMisuse Conversion Issue:")
    print("=" * 70)
    print("The downloaded VarMisuse dataset uses Protocol Buffer format,")
    print("but tf-gnn-samples expects JSON format.")
    print()
    print("To properly convert, you need:")
    print("1. The .proto schema file from the original VarMisuse paper")
    print("2. protobuf Python library with the compiled schema")
    print("3. Custom conversion code to map proto fields to JSON")
    print()
    print("This is complex and not documented in the bottleneck paper.")
    print("=" * 70)
    print()
    print("RECOMMENDED: Skip VarMisuse - it's a secondary validation experiment.")
    print("Your Dictionary Lookup results already demonstrate the bottleneck phenomenon.")
    
    return 1

if __name__ == '__main__':
    sys.exit(main())
