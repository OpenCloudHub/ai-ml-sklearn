# ==============================================================================
# Wine Classifier API Integration Tests
# ==============================================================================
#
# Dummy test suite for the Wine Quality Classifier API.
#
# Tests:
#   - Health check endpoint
#   - Single and batch predictions with real DVC data
#   - Error handling for invalid inputs (wrong features, wrong types)
#
# Prerequisites:
#   - Ray Serve running with a deployed model
#   - Access to DVC data (MinIO credentials)
#
# Usage:
#   # Start the serving application first:
#   serve run src.serving.serve:app_builder model_uri="models:/wine-classifier/1"
#
#   # Run tests:
#   python tests/test_wine_classifier.py
#
# TODO: Migrate to pytest for better test organization
#
# ==============================================================================

#!/usr/bin/env python3
"""
Test script for Wine Quality Classifier FastAPI deployment
"""
# TODO: use pytest

import os
import sys

import requests

# Add src to path so we can import data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.data import load_data

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")

    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed")
        print(f"   Status: {data['status']}")
        if data.get("model_info"):
            print(f"   Model URI: {data['model_info']['model_uri']}")
            print(f"   Data version: {data['model_info']['data_version']}")
            print(f"   Features: {len(data['model_info']['expected_features'])}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_prediction_with_real_data():
    """Test the prediction endpoint with actual wine quality data from DVC"""
    print("🍷 Testing wine quality predictions with real data...")

    # Get data version from server
    health_response = requests.get(f"{BASE_URL}/")
    if health_response.status_code != 200:
        print("❌ Cannot get data version from server")
        return

    data_version = health_response.json().get("model_info", {}).get("data_version")
    if not data_version:
        print("❌ Server has no data version info")
        return

    print(f"   Loading data version: {data_version}")

    # Load actual validation data
    try:
        X_train, y_train, X_val, y_val, metadata = load_data(version=data_version)

        # Take first 5 samples from validation set
        test_samples = X_val.head(5).values.tolist()
        expected = y_val.head(5).tolist()

        print(f"   Testing with {len(test_samples)} validation samples")

        # Prepare request
        payload = {"features": test_samples}

        # Make prediction request
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            predictions = [p["quality_score"] for p in result["predictions"]]
            confidences = [p["confidence"] for p in result["predictions"]]

            print("✅ Predictions successful")
            print(f"   Model URI: {result['model_uri']}")
            print("\n   Comparison:")
            print(
                f"   {'Sample':<8} {'Predicted':<12} {'Actual':<10} {'Confidence':<12} {'Match'}"
            )
            print(f"   {'-' * 8} {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 5}")

            correct = 0
            for i, (pred, actual, conf) in enumerate(
                zip(predictions, expected, confidences)
            ):
                match = "✓" if pred == actual else "✗"
                if pred == actual:
                    correct += 1
                print(
                    f"   {i + 1:<8} {pred:<12} {actual:<10} {conf:.3f}        {match}"
                )

            accuracy = correct / len(expected) * 100
            print(f"\n   Accuracy: {accuracy:.1f}% ({correct}/{len(expected)})")
            print(f"   Avg Confidence: {sum(confidences) / len(confidences):.3f}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")

    print()


def test_single_prediction():
    """Test prediction with a single wine sample from real data"""
    print("🍷 Testing single wine prediction...")

    # Get data version
    health_response = requests.get(f"{BASE_URL}/")
    if health_response.status_code != 200:
        print("❌ Cannot get data version from server")
        return

    data_version = health_response.json().get("model_info", {}).get("data_version")
    if not data_version:
        print("❌ Server has no data version info")
        return

    try:
        X_train, y_train, X_val, y_val, metadata = load_data(version=data_version)

        # Take one sample
        single_sample = [X_val.iloc[0].tolist()]
        actual_quality = y_val.iloc[0]

        payload = {"features": single_sample}

        response = requests.post(f"{BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            pred = result["predictions"][0]
            print("✅ Single prediction successful")
            print(f"   Predicted Quality: {pred['quality_score']}")
            print(f"   Actual Quality: {actual_quality}")
            print(f"   Confidence: {pred['confidence']:.3f}")
            print(
                f"   Match: {'✓' if pred['quality_score'] == actual_quality else '✗'}"
            )
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")

    print()


def test_invalid_input_wrong_features():
    """Test error handling with wrong number of features"""
    print("⚠️  Testing error handling (wrong feature count)...")

    # Invalid payload (only 3 features instead of 12)
    invalid_payload = {"features": [[7.4, 0.7, 0.0]]}

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)

    print(f"   Invalid input response: {response.status_code}")
    if response.status_code == 400:
        print("✅ Error handling works correctly")
        error = response.json()
        print(f"   Error message: {error.get('detail', 'No detail')}")
    else:
        print(f"⚠️  Expected 400 error but got {response.status_code}")
    print()


def test_invalid_input_wrong_type():
    """Test error handling with invalid data types"""
    print("⚠️  Testing error handling (invalid data types)...")

    # Invalid payload (strings instead of numbers)
    invalid_payload = {
        "features": [
            [
                "invalid",
                "data",
                "here",
                1.9,
                0.076,
                11.0,
                34.0,
                0.9978,
                3.51,
                0.56,
                9.4,
                0,
            ]
        ]
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)

    print(f"   Invalid input response: {response.status_code}")
    if response.status_code in [400, 422]:
        print("✅ Error handling works correctly")
        error = response.json()
        print(f"   Error: {error.get('detail', 'Validation error')}")
    else:
        print(f"⚠️  Expected 400/422 error but got {response.status_code}")
    print()


def test_batch_prediction():
    """Test batch prediction with multiple samples from real data"""
    print("🍷 Testing batch predictions (10 samples from validation set)...")

    # Get data version
    health_response = requests.get(f"{BASE_URL}/")
    if health_response.status_code != 200:
        print("❌ Cannot get data version from server")
        return

    data_version = health_response.json().get("model_info", {}).get("data_version")
    if not data_version:
        print("❌ Server has no data version info")
        return

    try:
        X_train, y_train, X_val, y_val, metadata = load_data(version=data_version)

        # Take 10 samples
        batch_samples = X_val.head(10).values.tolist()
        expected = y_val.head(10).tolist()

        payload = {"features": batch_samples}

        response = requests.post(f"{BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            predictions = [p["quality_score"] for p in result["predictions"]]
            confidences = [p["confidence"] for p in result["predictions"]]

            print("✅ Batch predictions successful")
            print(f"   Predicted {len(result['predictions'])} samples")

            # Calculate accuracy
            correct = sum(1 for p, e in zip(predictions, expected) if p == e)
            accuracy = correct / len(expected) * 100
            print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(expected)})")

            # Show quality distribution
            quality_counts = {}
            for pred in predictions:
                quality_counts[pred] = quality_counts.get(pred, 0) + 1

            print(
                f"   Predicted quality distribution: {dict(sorted(quality_counts.items()))}"
            )

            # Show average confidence
            avg_confidence = sum(confidences) / len(confidences)
            print(f"   Average confidence: {avg_confidence:.3f}")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ Failed to load data: {e}")

    print()


def main():
    """Run all tests"""
    print("🧪 Starting Wine Quality Classifier API Tests")
    print("=" * 50)

    try:
        test_health_check()
        test_prediction_with_real_data()
        test_single_prediction()
        test_batch_prediction()
        test_invalid_input_wrong_features()
        test_invalid_input_wrong_type()

        print("🎉 All tests completed!")
        print("\n📖 API Documentation available at: http://localhost:8000/docs")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Ray Serve is running:")
        print(
            "   serve run src.serve:app_builder model_uri='models:/wine-classifier/1'"
        )


if __name__ == "__main__":
    main()
