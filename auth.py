
import sqlite3
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "aegis_users.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)


def register_user(username, email, password):
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Please fill in all fields."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO users (username, email, password_hash)
                   VALUES (?, ?, ?)""",
                (
                    username,
                    email,
                    generate_password_hash(password),
                ),
            )
        return True, "Account created! Please log in."

    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def login_user(email, password):
    with sqlite3.connect(DB_PATH) as conn:
        user = conn.execute(
            """SELECT id, username, email, password_hash
               FROM users WHERE email = ? COLLATE NOCASE""",
            (email.strip(),),
        ).fetchone()

    if user and check_password_hash(user[3], password):
        return {
            "id": user[0],
            "username": user[1],
            "email": user[2],
        }

    return None



def show_auth():
    # Center the authentication form
    left, center, right = st.columns([1, 1.1, 1])

    with center:
        st.markdown(
            """
            <div style="text-align:center; margin-bottom:18px;">
                <div style="font-size:42px;">🛡️</div>
                <h2 style="margin-bottom:4px;">Aegis</h2>
                <p style="color:gray; margin-top:0;">
                    AI Document Intelligence
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        login_tab, register_tab = st.tabs(["Login", "Register"])

        with login_tab:
            with st.form("aegis_login_form"):
                email = st.text_input(
                    "Email address",
                    key="auth_login_email",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    key="auth_login_password",
                )
                submitted = st.form_submit_button(
                    "Login",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                user = login_user(email, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        with register_tab:
            with st.form("aegis_register_form"):
                username = st.text_input(
                    "Username",
                    key="auth_register_username",
                )
                email = st.text_input(
                    "Email address",
                    key="auth_register_email",
                )
                password = st.text_input(
                    "Create password",
                    type="password",
                    key="auth_register_password",
                )
                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="auth_register_confirm",
                )
                submitted = st.form_submit_button(
                    "Create Account",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                    success, message = register_user(
                        username, email, password
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)



def require_login():
    init_db()

    if not st.session_state.get("authenticated", False):
        show_auth()
        st.stop()

    user = st.session_state.get("user")

    if not user:
        st.session_state["authenticated"] = False
        st.stop()

    with st.sidebar:
        st.write(f"👤 **{user['username']}**")
        st.caption(user["email"])

        if st.button("Logout", use_container_width=True):
            # Clear user-specific information on logout.
            for key in (
                "user",
                "authenticated",
                "aegis_results",
                "aegis_live_report",
                "aegis_comparison",
                "gemini_api_key",
            ):
                st.session_state.pop(key, None)

            st.rerun()
