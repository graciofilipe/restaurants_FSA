import streamlit as st
from auth.firebase_auth import AuthManager

def login_page(auth_manager: AuthManager):
    # Check for existing session or cookie
    auth_manager.check_auth()
    
    if auth_manager.is_authenticated():
        st.success(f"Successfully signed in as {auth_manager.get_user_email()}")
        if st.button("Go to Application"):
            st.rerun()
        return

    st.title("FSA API Explorer - Login")
    st.markdown("Please sign in with your Google account to access the application.")
    
    auth_manager.login_button()

if __name__ == "__main__":
    login_page()
