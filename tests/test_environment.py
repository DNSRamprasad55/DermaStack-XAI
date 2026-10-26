"""
Test Environment & Dependency Verification.
Verifies that all required packages, configurations, and paths are valid.
"""

import sys
import unittest
from pathlib import Path


class TestEnvironment(unittest.TestCase):
    """Test suite for environment and configuration."""

    def test_python_version(self):
        """Verify Python version is at least 3.10 and 64-bit."""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3)
        self.assertGreaterEqual(minor, 10)
        self.assertTrue(sys.maxsize > 2**32, "Python must be 64-bit.")

    def test_core_dependencies_import(self):
        """Verify core deep learning and ML libraries import properly."""
        import numpy as np
        import pandas as pd
        import sklearn
        import PIL
        import matplotlib
        import streamlit

        self.assertIsNotNone(np.__version__)
        self.assertIsNotNone(pd.__version__)
        self.assertIsNotNone(sklearn.__version__)
        self.assertIsNotNone(PIL.__version__)
        self.assertIsNotNone(matplotlib.__version__)
        self.assertIsNotNone(streamlit.__version__)

    def test_tensorflow_and_keras_import(self):
        """Verify TensorFlow and Keras are available."""
        import tensorflow as tf
        import keras

        self.assertTrue(tf.__version__.startswith("2."))
        self.assertIsNotNone(keras.__version__)

    def test_config_paths_and_classes(self):
        """Verify configuration module integrity."""
        from src.config import (
            DEFAULT_CONFIG,
            HAM10000_CLASSES,
            CLASS_NAMES_MAPPING,
            MEDICAL_DISCLAIMER
        )

        self.assertEqual(len(HAM10000_CLASSES), 7)
        self.assertEqual(len(CLASS_NAMES_MAPPING), 7)
        self.assertIn("mel", HAM10000_CLASSES)
        self.assertIn("nv", HAM10000_CLASSES)
        self.assertTrue(len(MEDICAL_DISCLAIMER) > 50)
        self.assertTrue(Path(DEFAULT_CONFIG.paths.root).exists())

    def test_seed_reproducibility(self):
        """Verify reproducibility helper works without error."""
        from src.utils.seed import set_seed
        import numpy as np

        set_seed(42)
        r1 = np.random.rand(5)
        set_seed(42)
        r2 = np.random.rand(5)
        np.testing.assert_allclose(r1, r2)


if __name__ == "__main__":
    unittest.main()
