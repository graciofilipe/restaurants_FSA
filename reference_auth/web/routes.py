import os
from flask import Blueprint, request, render_template, redirect, url_for
from services.storage import StorageService

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.route("/version", methods=["GET"])
def check_version():
    return "Version: 1.0.1 - Pagination & Color Test", 200

@admin_bp.route("/search", methods=["GET"])
def admin_search():
    # Consolidate search to the public root endpoint
    return redirect(url_for('public.index', **request.args))