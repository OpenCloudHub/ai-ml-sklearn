#!/usr/bin/env python3
"""
Test script for Wine Classifier FastAPI deployment
"""

import requests
from sklearn import datasets

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")

    if response.status_code == 200:
        print("✅ Health check passed")
        print(f"   Response: {response.json()}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_model_info():
    """Test the model info endpoint"""
    print("📊 Testing model info...")
    response = requests.get(f"{BASE_URL}/model/info")

    if response.status_code == 200:
        info = response.json()
        print("✅ Model info retrieved")
        print(f"   Model: {info['model_name']} v{info['model_version']}")
        print(f"   Expected features: {info['features_expected']}")
        print(f"   Output classes: {info['output_classes']}")
    else:
        print(f"❌ Model info failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_prediction():
    """Test the prediction endpoint with real wine data"""
    print("🍷 Testing wine prediction...")

    # Load sample wine data
    wine = datasets.load_wine()
    test_samples = wine.data[:3]  # Get 3 samples
    expected = wine.target[:3]

    # Prepare request
    payload = {"features": test_samples.tolist()}

    # Make prediction request
    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Predictions successful")
        print(f"   Predictions: {result['predictions']}")
        print(f"   Expected:    {expected.tolist()}")
        print(f"   Model:       {result['model_name']} v{result['model_version']}")

        # Check accuracy
        correct = sum(1 for p, e in zip(result["predictions"], expected) if p == e)
        accuracy = correct / len(expected) * 100
        print(f"   Accuracy:    {accuracy:.1f}% ({correct}/{len(expected)})")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_single_prediction():
    """Test prediction with a single wine sample"""
    print("🍷 Testing single wine prediction...")

    # Single wine sample (from wine dataset)
    single_sample = [
        [
            1.423e01,
            1.710e00,
            2.430e00,
            1.560e01,
            1.270e02,
            2.800e00,
            3.060e00,
            2.800e-01,
            2.290e00,
            5.640e00,
            1.040e00,
            3.920e00,
            1.065e03,
        ]
    ]

    payload = {"features": single_sample}

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ Single prediction successful")
        print(f"   Prediction: Class {result['predictions'][0]}")
        print(f"   Model: {result['model_name']} v{result['model_version']}")
    else:
        print(f"❌ Single prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_invalid_input():
    """Test error handling with invalid input"""
    print("⚠️  Testing error handling...")

    # Invalid payload (wrong number of features)
    invalid_payload = {
        "features": [[1.0, 2.0, 3.0]]  # Only 3 features instead of 13
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)

    print(f"   Invalid input response: {response.status_code}")
    if response.status_code != 200:
        print("✅ Error handling works correctly")
    else:
        print("⚠️  Expected error but got success")
    print()


def main():
    """Run all tests"""
    print("🧪 Starting Wine Classifier API Tests")
    print("=" * 50)

    try:
        test_health_check()
        test_model_info()
        test_prediction()
        test_single_prediction()
        test_invalid_input()

        print("🎉 All tests completed!")
        print("\n📖 API Documentation available at: http://localhost:8000/docs")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Ray Serve is running:")
        print(
            "   serve run --working-dir /workspace/project wine_fastapi_serve:wine_app"
        )


if __name__ == "__main__":
    main()
