import os

def test_no_firebase_in_secrets():
    secrets_path = '.streamlit/secrets.toml'
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            content = f.read()
        assert '[firebase]' not in content, "Secrets file should not contain [firebase] section"
        assert 'apiKey' not in content, "Secrets file should not contain apiKey"
        assert 'authDomain' not in content, "Secrets file should not contain authDomain"

def test_no_firebase_in_envs():
    envs_path = 'scripts/envs.sh'
    if os.path.exists(envs_path):
        with open(envs_path, 'r') as f:
            content = f.read()
        assert 'FIREBASE' not in content.upper(), "envs.sh should not contain FIREBASE variables"
