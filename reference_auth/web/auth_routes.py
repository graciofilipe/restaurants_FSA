import os
from flask import Blueprint, request, session, jsonify, redirect, url_for, render_template
from web.auth import verify_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET'])
def login_page():
    """
    Render the login page.
    """
    if session.get('user'):
        return redirect(url_for('public.index'))
    return render_template('login.html')

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Exchange a Firebase ID token for a server-side session.
    """
    data = request.get_json()
    id_token = data.get('id_token')

    if not id_token:
        return jsonify({'error': 'Missing ID token'}), 400

    decoded_token = verify_token(id_token)
    
    if decoded_token:
        user_email = decoded_token.get('email')
        
        # --- Authorization Check ---
        allowed_emails_str = os.environ.get('ALLOWED_EMAILS', '')
        allowed_emails = [e.strip().lower() for e in allowed_emails_str.split(';') if e.strip()]

        allowed_domains_str = os.environ.get('ALLOWED_DOMAINS', '')
        allowed_domains = [d.strip().lower() for d in allowed_domains_str.split(';') if d.strip()]

        is_authorized = False
        if user_email:
            if user_email.lower() in allowed_emails:
                is_authorized = True
            elif '@' in user_email:
                user_domain = user_email.split('@')[1].lower()
                if user_domain in allowed_domains:
                    is_authorized = True
        
        # If no ALLOWED_EMAILS or ALLOWED_DOMAINS are set, allow all.
        # Otherwise, user must be in one of the allowlists.
        if (not allowed_emails and not allowed_domains) or is_authorized:
            session['user'] = {
                'uid': decoded_token['uid'],
                'email': user_email,
                'name': decoded_token.get('name'),
                'picture': decoded_token.get('picture')
            }
            return jsonify({'success': True, 'user': session['user']}), 200
        else:
            print(f"Unauthorized access attempt for email: {user_email}")
            return jsonify({'error': 'Unauthorized email address or domain'}), 403 # Forbidden
    else:
        return jsonify({'error': 'Invalid ID token'}), 401

@auth_bp.route('/logout', methods=['POST', 'GET'])
def logout():
    """
    Clear the server-side session.
    """
    session.pop('user', None)
    return redirect(url_for('public.index'))

@auth_bp.route('/me', methods=['GET'])
def me():
    """
    Return current user info.
    """
    user = session.get('user')
    if user:
        return jsonify(user), 200
    else:
        return jsonify({'error': 'Not logged in'}), 401
