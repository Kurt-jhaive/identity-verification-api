"""
Test client for Identity Verification API
Use this script to test the API with sample images
"""

import requests
import sys
from pathlib import Path


def test_verification(selfie_path: str, id_card_path: str, api_url: str = "http://localhost:8000"):
    """
    Test the identity verification API with two images.
    
    Args:
        selfie_path: Path to selfie image
        id_card_path: Path to ID card image
        api_url: Base URL of the API
    """
    endpoint = f"{api_url}/api/verify"
    
    # Check if files exist
    if not Path(selfie_path).exists():
        print(f"❌ Error: Selfie file not found: {selfie_path}")
        return
    
    if not Path(id_card_path).exists():
        print(f"❌ Error: ID card file not found: {id_card_path}")
        return
    
    print(f"📤 Sending request to {endpoint}")
    print(f"   Selfie: {selfie_path}")
    print(f"   ID Card: {id_card_path}")
    print()
    
    try:
        # Open files and create multipart form data
        with open(selfie_path, 'rb') as selfie_file, open(id_card_path, 'rb') as id_card_file:
            files = {
                'selfie': ('selfie.jpg', selfie_file, 'image/jpeg'),
                'id_card': ('id_card.jpg', id_card_file, 'image/jpeg')
            }
            
            # Send POST request
            response = requests.post(endpoint, files=files)
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"   Match: {result['match']}")
            print(f"   Confidence Score: {result['confidence_score']}%")
            print(f"   Message: {result['message']}")
        else:
            error = response.json()
            print("❌ ERROR!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Error: {error.get('error', 'Unknown')}")
            print(f"   Error Code: {error.get('error_code', 'Unknown')}")
            print(f"   Message: {error.get('message', 'Unknown')}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API.")
        print("   Make sure the API is running at:", api_url)
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


def test_health_check(api_url: str = "http://localhost:8000"):
    """Test the health check endpoint."""
    endpoint = f"{api_url}/health"
    
    print(f"🏥 Testing health check: {endpoint}")
    
    try:
        response = requests.get(endpoint)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API is healthy!")
            print(f"   Service: {result.get('service')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Threshold: {result.get('threshold')}")
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API.")
        print("   Make sure the API is running at:", api_url)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test health:       python test_client.py health")
        print("  Test verification: python test_client.py <selfie_path> <id_card_path> [api_url]")
        print()
        print("Example:")
        print("  python test_client.py health")
        print("  python test_client.py selfie.jpg id_card.jpg")
        print("  python test_client.py selfie.jpg id_card.jpg http://192.168.1.100:8000")
        sys.exit(1)
    
    if sys.argv[1] == "health":
        api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        test_health_check(api_url)
    elif len(sys.argv) >= 3:
        selfie = sys.argv[1]
        id_card = sys.argv[2]
        api_url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8000"
        test_verification(selfie, id_card, api_url)
    else:
        print("❌ Error: Please provide both selfie and ID card paths")
        sys.exit(1)
