def test_no_firebase_admin_in_requirements():
    with open('requirements.txt', 'r') as f:
        reqs = f.read()
    assert 'firebase-admin' not in reqs, "firebase-admin should be removed from requirements.txt"
