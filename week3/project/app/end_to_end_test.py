import requests
import sys

headers = {'accept': 'application/json', 'Content-Type': 'application/json'}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} endpoint")
        sys.exit(0)
    
    endpoint = sys.argv[1].rstrip('/') + '/predict'
    print(f'sending requests to {endpoint}')

    with open('../data/requests.json', 'r') as f:
        test_payloads = f.readlines()

    for payload in test_payloads:
        print(f'sending {payload}'.rstrip('\n'))
        response = requests.post(endpoint, data=payload, headers=headers)
        print(response.status_code, response.text)

if __name__ == "__main__":
    main()
    