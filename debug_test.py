#!/usr/bin/env python3
"""
Simple test script to debug the research aggregator
"""
import sys

print("Python version:", sys.version)
print("Starting imports...")

try:
    import asyncio

    print("✓ asyncio imported")

    import httpx

    print("✓ httpx imported")

    import torch

    print("✓ torch imported")

    import faiss

    print("✓ faiss imported")

    import numpy as np

    print("✓ numpy imported")

    from sentence_transformers import SentenceTransformer

    print("✓ sentence_transformers imported")

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded successfully!")

    print("Testing basic functionality...")
    test_text = ["This is a test sentence"]
    embeddings = model.encode(test_text)
    print(f"✓ Embedding shape: {embeddings.shape}")

    print("All tests passed!")

except Exception as e:
    print(f" Error during testing: {e}")
    import traceback

    traceback.print_exc()
