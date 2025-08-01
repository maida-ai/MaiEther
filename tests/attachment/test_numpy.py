"""Tests for NumPy attachment functionality."""

import importlib
import sys
from collections.abc import Generator

import pytest

import ether
from ether import Ether
from ether.attachment import Attachment

# Import NumPy for testing
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
class TestAttachmentNumPy:
    """Test Attachment NumPy conversion methods."""

    def test_from_numpy_basic(self) -> None:
        """Test basic NumPy array conversion to attachment."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        att = Attachment.from_numpy(arr, id="test-0")

        assert att.id == "test-0"
        assert att.media_type == "application/x-raw-tensor"
        assert att.codec == "RAW_F32"
        assert att.shape == [3]
        assert att.dtype == "float32"
        assert att.size_bytes == 12  # 3 * 4 bytes per float32
        assert att.byte_order == "LE"
        assert att.inline_bytes == arr.tobytes()

    def test_from_numpy_with_uri(self) -> None:
        """Test NumPy array conversion with URI reference."""
        arr = np.array([1, 2, 3, 4], dtype=np.int32)
        att = Attachment.from_numpy(arr, id="test-1", uri="shm://data/12345")

        assert att.id == "test-1"
        assert att.uri == "shm://data/12345"
        assert att.inline_bytes is None  # Should not store inline when URI provided
        assert att.codec == "RAW_I32"
        assert att.shape == [4]
        assert att.dtype == "int32"
        assert att.size_bytes == 16  # 4 * 4 bytes per int32

    def test_from_numpy_multidimensional(self) -> None:
        """Test NumPy array conversion with multidimensional arrays."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        att = Attachment.from_numpy(arr, id="test-2")

        assert att.shape == [2, 2]
        assert att.dtype == "float64"
        assert att.codec == "RAW_F64"
        assert att.size_bytes == 32  # 4 * 8 bytes per float64

    def test_from_numpy_unsupported_dtype(self) -> None:
        """Test NumPy array conversion with unsupported dtype."""
        arr = np.array([1, 2, 3], dtype=np.complex64)
        att = Attachment.from_numpy(arr, id="test-3")

        assert att.codec == "RAW_BYTES"  # Generic codec for unsupported types
        assert att.dtype == "complex64"

    def test_from_numpy_error_not_ndarray(self) -> None:
        """Test that from_numpy raises error for non-ndarray."""
        with pytest.raises(ValueError, match="array must be a NumPy ndarray"):
            Attachment.from_numpy([1, 2, 3], id="test-4")

    def test_to_numpy_basic(self) -> None:
        """Test basic attachment to NumPy array conversion."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        att = Attachment.from_numpy(arr, id="test-5")
        restored = att.to_numpy()

        assert np.array_equal(arr, restored)
        assert restored.dtype == arr.dtype
        assert restored.shape == arr.shape

    def test_to_numpy_multidimensional(self) -> None:
        """Test attachment to NumPy array conversion with multidimensional arrays."""
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        att = Attachment.from_numpy(arr, id="test-6")
        restored = att.to_numpy()

        assert np.array_equal(arr, restored)
        assert restored.dtype == arr.dtype
        assert restored.shape == arr.shape

    def test_to_numpy_round_trip_various_dtypes(self) -> None:
        """Test round-trip conversion with various NumPy dtypes."""
        dtypes = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
        ]

        for dtype in dtypes:
            arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
            att = Attachment.from_numpy(arr, id=f"test-{dtype}")
            restored = att.to_numpy()

            assert np.array_equal(arr, restored)
            assert restored.dtype == arr.dtype

    def test_to_numpy_error_no_inline_bytes(self) -> None:
        """Test that to_numpy raises error when no inline bytes."""
        att = Attachment(
            id="test-7",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[3],
            dtype="float32",
        )

        with pytest.raises(RuntimeError, match="URI-based attachments not yet supported"):
            att.to_numpy()

    def test_to_numpy_error_no_data(self) -> None:
        att = Attachment(
            id="test-no-data",
            uri="file:///data.bin",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[3],
            dtype="float32",
        )
        att.uri = None
        with pytest.raises(ValueError, match="Attachment has no data to convert"):
            att.to_numpy()

    def test_to_numpy_error_missing_metadata(self) -> None:
        """Test that to_numpy raises error when missing shape or dtype."""
        att = Attachment(
            id="test-8",
            inline_bytes=b"\x00\x00\x80\x3f",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            # Missing shape and dtype
        )

        with pytest.raises(ValueError, match="Attachment missing shape or dtype"):
            att.to_numpy()

    def test_to_numpy_error_invalid_dtype(self) -> None:
        """Test that to_numpy raises error for invalid dtype."""
        att = Attachment(
            id="test-9",
            inline_bytes=b"\x00\x00\x80\x3f",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[1],
            dtype="invalid_dtype",
        )

        with pytest.raises(ValueError, match="Invalid dtype"):
            att.to_numpy()

    def test_to_numpy_error_data_mismatch(self) -> None:
        """Test that to_numpy raises error when data doesn't match shape/dtype."""
        att = Attachment(
            id="test-10",
            inline_bytes=b"\x00\x00\x80\x3f",  # 1 float32 value
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
            shape=[2],  # Expecting 2 values
            dtype="float32",
        )

        with pytest.raises(ValueError, match="Failed to reconstruct array"):
            att.to_numpy()

    def test_from_numpy_large_array(self) -> None:
        """Test NumPy array conversion with large arrays."""
        # Create a large array (1MB of data)
        arr = np.random.random(262144).astype(np.float32)  # 1MB / 4 bytes per float32
        att = Attachment.from_numpy(arr, id="test-11")

        assert att.size_bytes == 1048576  # 1MB
        assert att.shape == [262144]
        assert att.dtype == "float32"

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)

    def test_from_numpy_zero_dimensional(self) -> None:
        """Test NumPy array conversion with zero-dimensional arrays."""
        arr = np.array(42, dtype=np.int32)
        att = Attachment.from_numpy(arr, id="test-12")

        assert att.shape == []
        assert att.dtype == "int32"
        assert att.size_bytes == 4

        restored = att.to_numpy()
        assert np.array_equal(arr, restored)

    def test_from_numpy_empty_array(self) -> None:
        """Test NumPy array conversion with empty arrays."""
        arr = np.array([], dtype=np.float32)
        att = Attachment.from_numpy(arr, id="test-13")

        assert att.shape == [0]
        assert att.dtype == "float32"
        assert att.size_bytes == 0

        restored = att.to_numpy()
        assert np.array_equal(arr, restored)


class TestNumPyNotAvailable:
    """Test the cases where NumPy is not available."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch: pytest.MonkeyPatch) -> Generator:
        # Store original state
        original_numpy = sys.modules.get("numpy")
        original_flag = getattr(ether.attachment, "NUMPY_AVAILABLE", None)

        # Patch NumPy to be unavailable
        monkeypatch.setitem(sys.modules, "numpy", None)
        importlib.reload(sys.modules["ether.attachment"])

        yield

        # Restore original state
        if original_numpy is not None:
            sys.modules["numpy"] = original_numpy
        if original_flag is not None:
            ether.attachment.NUMPY_AVAILABLE = original_flag
        importlib.reload(sys.modules["ether.attachment"])

    def test_unavailable_flag(self) -> None:
        """Test that the global flag is set to False when NumPy is not available."""
        assert ether.attachment.NUMPY_AVAILABLE is False

    def test_from_raises(self) -> None:
        """Test that from_numpy raises error when NumPy is not available."""
        with pytest.raises(ImportError, match="NumPy is required.*?from_numpy"):
            Attachment.from_numpy(np.array([1, 2, 3]), id="test-14")

    def test_to_raises(self) -> None:
        """Test that to_numpy raises error when NumPy is not available."""
        att = Attachment(
            id="test-15",
            inline_bytes=b"\x00\x00\x80\x3f",
            media_type="application/x-raw-tensor",
            codec="RAW_F32",
        )
        with pytest.raises(ImportError, match="NumPy is required.*?to_numpy"):
            att.to_numpy()


class TestZeroCopy:
    """Test zero-copy functionality to ensure no accidental copies are introduced."""

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_from_numpy_zero_copy_inline(self) -> None:
        """Test that from_numpy doesn't copy array data when storing inline."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-zero-copy")

        # Verify the bytes are the same (not copied)
        assert att.inline_bytes == original_bytes
        assert id(att.inline_bytes) != id(original_bytes)  # Should be different objects but same content

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_from_numpy_zero_copy_uri(self) -> None:
        """Test that from_numpy doesn't store inline data when URI is provided."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        att = Attachment.from_numpy(arr, id="test-zero-copy-uri", uri="shm://data/12345")

        # Should not store inline data when URI is provided
        assert att.inline_bytes is None
        assert att.uri == "shm://data/12345"

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_to_numpy_zero_copy(self) -> None:
        """Test that to_numpy reconstructs array without copying the original data."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        att = Attachment.from_numpy(arr, id="test-zero-copy-to")

        restored = att.to_numpy()

        # Verify data integrity
        assert np.array_equal(arr, restored)
        assert restored.dtype == arr.dtype
        assert restored.shape == arr.shape

        # Verify it's a new array (not the same object)
        assert id(restored) != id(arr)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_large_array_zero_copy(self) -> None:
        """Test zero-copy with large arrays to ensure performance."""
        # Create a large array (1MB)
        arr = np.random.random(262144).astype(np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-large-zero-copy")

        # Verify no copy was made
        assert att.inline_bytes == original_bytes
        assert len(att.inline_bytes) == 1048576  # 1MB

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_multidimensional_zero_copy(self) -> None:
        """Test zero-copy with multidimensional arrays."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-multi-zero-copy")

        # Verify no copy was made
        assert att.inline_bytes == original_bytes

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)
        assert restored.shape == arr.shape

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_zero_dimensional_zero_copy(self) -> None:
        """Test zero-copy with zero-dimensional arrays."""
        arr = np.array(42, dtype=np.int32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-zero-dim-zero-copy")

        # Verify no copy was made
        assert att.inline_bytes == original_bytes

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_various_dtypes_zero_copy(self) -> None:
        """Test zero-copy with various NumPy dtypes."""
        dtypes = [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
        ]

        for dtype in dtypes:
            arr = np.array([1, 2, 3, 4, 5], dtype=dtype)
            original_bytes = arr.tobytes()

            att = Attachment.from_numpy(arr, id=f"test-{dtype}-zero-copy")

            # Verify no copy was made
            assert att.inline_bytes == original_bytes

            # Test round-trip
            restored = att.to_numpy()
            assert np.array_equal(arr, restored)
            assert restored.dtype == arr.dtype

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_empty_array_zero_copy(self) -> None:
        """Test zero-copy with empty arrays."""
        arr = np.array([], dtype=np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-empty-zero-copy")

        # Verify no copy was made
        assert att.inline_bytes == original_bytes

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_ether_attachment_zero_copy(self) -> None:
        """Test that Ether with attachments doesn't copy attachment data."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-ether-zero-copy")
        ether = Ether(kind="test", payload={"dim": 3}, metadata={"source": "test"}, attachments=[att])

        # Verify attachment data wasn't copied
        assert ether.attachments[0].inline_bytes == original_bytes

        # Test round-trip through Ether
        restored = ether.attachments[0].to_numpy()
        assert np.array_equal(arr, restored)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_serialization_zero_copy(self) -> None:
        """Test that serialization doesn't introduce copies."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original_bytes = arr.tobytes()

        att = Attachment.from_numpy(arr, id="test-serial-zero-copy")

        # Serialize and deserialize
        serialized = att.model_dump_json()
        deserialized = Attachment.model_validate_json(serialized)

        # Verify data integrity after serialization
        assert deserialized.inline_bytes == original_bytes

        # Test round-trip
        restored = deserialized.to_numpy()
        assert np.array_equal(arr, restored)

    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_memory_efficiency(self) -> None:
        """Test that we're not using excessive memory."""
        import sys

        # Create a large array
        arr = np.random.random(100000).astype(np.float32)
        original_size = sys.getsizeof(arr.tobytes())

        att = Attachment.from_numpy(arr, id="test-memory")

        # Verify we're not using significantly more memory
        attachment_size = sys.getsizeof(att.inline_bytes)
        assert attachment_size <= original_size * 1.1  # Allow 10% overhead for object metadata

        # Test round-trip
        restored = att.to_numpy()
        assert np.array_equal(arr, restored)
