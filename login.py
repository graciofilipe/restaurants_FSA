import streamlit as st
import streamlit.components.v1 as components
import json
from auth.firebase_auth import AuthManager

def login_page(auth_manager: AuthManager):
    """
    Renders the login page with a Google Sign-In button.
    """
    st.set_page_config(page_title="Login - FSA API Explorer", layout="centered")
    
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            margin-top: 10vh;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .login-title {
            margin-bottom: 1.5rem;
            color: #31333F;
            font-size: 2.5rem;
            font-weight: bold;
        }
        .login-text {
            margin-bottom: 2rem;
            color: #555;
            font-size: 1.1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">FSA API Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-text">Please sign in with your Google account to access the application.</div>', unsafe_allow_html=True)
    
    # We use a popup flow. The button will open a new window to ?mode=auth
    # Once auth is done there, it will postMessage back to this window.
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .google-btn {
                background-color: #4285F4;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-family: 'Roboto', sans-serif;
                font-size: 16px;
                font-weight: 500;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: background-color 0.2s, box-shadow 0.2s;
            }
            .google-btn:hover {
                background-color: #357ae8;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .google-btn:active {
                background-color: #3367d6;
            }
        </style>
    </head>
    <body>
        <button class="google-btn" id="login-btn">
            <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="20" height="20" alt="Google logo">
            Sign in with Google
        </button>

        <script>
            const loginBtn = document.getElementById('login-btn');
            loginBtn.addEventListener('click', () => {
                // Open the auth popup
                const width = 500;
                const height = 600;
                const left = (window.innerWidth / 2) - (width / 2);
                const top = (window.innerHeight / 2) - (height / 2);
                
                const authWindow = window.open(
                    window.location.origin + window.location.pathname + '?mode=auth',
                    'firebaseAuthPopup',
                    `width=${width},height=${height},top=${top},left=${left}`
                );
            });

            // Listen for the result from the popup
            window.addEventListener('message', (event) => {
                if (event.data.type === 'FIREBASE_AUTH_RESULT') {
                    if (event.data.success) {
                        const { token, email } = event.data.data;
                        // Send this to Streamlit via a query parameter or hidden input
                        // The easiest way to trigger a rerun with the token is query params
                        const url = new URL(window.location.href);
                        url.searchParams.set('token', token);
                        window.parent.location.href = url.toString();
                    } else {
                        console.error("Auth failed:", event.data.error);
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    
    components.html(html_content, height=100)
    st.markdown('</div>', unsafe_allow_html=True)