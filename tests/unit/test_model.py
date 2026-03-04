from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

# Modern path yönetimi (pathlib)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "nyc_taxi_model.onnx"


@pytest.fixture(scope="module")
def shared_session() -> ort.InferenceSession:
    """
    ONNX modelini test modülü boyunca sadece 1 kez belleğe alır (Singleton mantığı).
    """
    if not MODEL_PATH.exists():
        pytest.fail(f"Model file not found! Path: {MODEL_PATH}")
    try:
        session = ort.InferenceSession(str(MODEL_PATH))
        return session
    except Exception as e:
        pytest.fail(f"Model failed to load during setup: {str(e)}")


class TestModelArtifact:
    """
    Model Artifact Tests:
    Tests the existence, loadability, and predictive ability of the trained .onnx model file.
    """

    def test_model_file_exists(self):
        """
        Test: Does the model file physically exist on the disk?
        """
        assert MODEL_PATH.exists(), f"Model file not found! Path: {MODEL_PATH}"
        assert MODEL_PATH.is_file(), "Path exists but is not a file!"

    def test_model_loading(self, shared_session: ort.InferenceSession):
        """
        Test: Can the ONNX Runtime load the model without errors?
        """
        assert shared_session is not None, "Model session could not be created."

    def test_model_metadata(self, shared_session: ort.InferenceSession):
        """
        Test: Are the model's input and output definitions correct?
        """
        inputs = shared_session.get_inputs()
        assert len(inputs) > 0, "The model has no input layer!"

        # Input tipinin Float32 olmasını bekleriz (ML modellerinde standarttır)
        assert inputs[0].type == "tensor(float)", (
            f"Unexpected input type: {inputs[0].type}"
        )

        outputs = shared_session.get_outputs()
        assert len(outputs) > 0, "The model has no output layer!"

    def test_prediction_flow(self, shared_session: ort.InferenceSession):
        """
        Test: Does the model generate valid predictions when fed with random data?
        """
        input_meta = shared_session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        # Beklenen feature sayısı (None, N_features)
        n_features = input_shape[1]

        # Dummy data oluştur (Batch Size = 1)
        dummy_input = np.random.rand(1, n_features).astype(np.float32)

        try:
            result = shared_session.run(None, {input_name: dummy_input})
            prediction = result[0]

            # 1. Boyut (Shape) Kontrolü: 1 satır veri yolladık, 1 adet sonuç dönmeli
            assert prediction.shape == (1, 1), (
                f"Expected shape (1, 1), got {prediction.shape}"
            )

            predicted_value = prediction[0][0]

            # 2. Tip ve Geçerlilik Kontrolü
            assert isinstance(predicted_value, (np.floating, float)), (
                "Output is not a float!"
            )
            assert not np.isnan(predicted_value), "Model predicted NaN!"
            assert not np.isinf(predicted_value), "Model predicted Infinity!"

        except Exception as e:
            pytest.fail(f"Error during inference: {str(e)}")
