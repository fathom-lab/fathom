import requests
TOKEN = "REDACTED_SEE_SECRETS"
BASE = "https://zenodo.org/api"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Check orphan status
r = requests.get(f"{BASE}/deposit/depositions/19467650", headers=HEADERS)
print(f"GET status: {r.status_code}")
d = r.json()
print(f"State: {d.get('state')}")
print(f"Submitted: {d.get('submitted')}")
print(f"Title: {str(d.get('title', ''))[:60]}")

# Try delete
if d.get("state") == "unsubmitted":
    rd = requests.delete(f"{BASE}/deposit/depositions/19467650", headers=HEADERS)
    print(f"DELETE status: {rd.status_code}")
else:
    print("Record is published — cannot delete via API. Need Zenodo support.")
