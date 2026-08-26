"""
Email OTP auth via AWS SES, matching the pattern used across Sega's other internal Streamlit
tools (see e.g. narrative_qa.py) — signed token in st.query_params so login survives page
reloads, not just session_state.

Secrets required (see .streamlit/secrets.toml.example):
    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SES_REGION   — SES send permissions
    EMAIL_FROM             — verified SES sender address
    COOKIE_SIGNING_KEY     — long random string, signs the persistent login token
    ALLOWED_DOMAIN         — email domain allowed to sign in (default segaamerica.com)

This is the file to replace wholesale once Sega's SSO is ready — everything downstream reads
`st.session_state["auth_email"]` and doesn't care how it got set.
"""
import base64
import hashlib
import hmac
import time
from random import SystemRandom

import streamlit as st

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

ALLOWED_DOMAIN = None  # set at runtime from secrets in require_login(), see _allowed_domain()
OTP_EXPIRY_SECS = 600
TOKEN_EXPIRY_DAYS = 7
MAX_OTP_ATTEMPTS = 5


def _allowed_domain() -> str:
    return st.secrets.get("ALLOWED_DOMAIN", "segaamerica.com")


def _generate_otp() -> str:
    return f"{SystemRandom().randint(0, 999999):06d}"


def _send_otp_email(email: str, code: str) -> bool:
    if not HAS_BOTO3:
        st.error("boto3 is required for OTP email (pip install boto3).")
        return False
    try:
        ses = boto3.client(
            "ses",
            region_name=st.secrets.get("AWS_SES_REGION", "us-east-1"),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        )
        ses.send_email(
            Source=st.secrets.get("EMAIL_FROM", "metareview-tool@segaamerica.com"),
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": "Metareview Tool — Verification Code", "Charset": "UTF-8"},
                "Body": {"Text": {"Data": f"Your code is:\n\n    {code}\n\nExpires in "
                                           f"{OTP_EXPIRY_SECS // 60} minutes.", "Charset": "UTF-8"}},
            },
        )
        return True
    except KeyError:
        return False  # SES secrets not configured — caller falls back to dev mode
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to send email: {exc}")
        return False


def _sign_token(email: str) -> str:
    key = st.secrets.get("COOKIE_SIGNING_KEY", "dev-key-change-me").encode()
    ts = str(int(time.time()))
    payload = f"{email}|{ts}"
    sig = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def _verify_token(token: str):
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        email, ts_str, sig = raw.split("|")
        key = st.secrets.get("COOKIE_SIGNING_KEY", "dev-key-change-me").encode()
        expected = hmac.new(key, f"{email}|{ts_str}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        if time.time() - int(ts_str) > TOKEN_EXPIRY_DAYS * 86400:
            return None
        return email
    except Exception:  # noqa: BLE001
        return None


def _check_auth() -> bool:
    if st.session_state.get("auth_email"):
        return True
    token = st.query_params.get("t")
    if token:
        email = _verify_token(token)
        if email:
            st.session_state["auth_email"] = email
            return True
    return False


def require_login():
    """Call at the top of the app. Renders a login form and halts (st.stop()) until the
    user has a verified, non-expired session. Safe to call on every rerun."""
    if _check_auth():
        return

    st.title("Metareview Tool — Sign in")
    st.caption(
        f"OTP-based access for now — sign in with your @{_allowed_domain()} email. This "
        "moves to Sega's SSO once that's wired up; see auth.py for the swap-out point."
    )

    if not st.session_state.get("otp_code"):
        email = st.text_input("Work email", key="login_email")
        if st.button("Send verification code", type="primary"):
            email = (email or "").strip()
            if not email or not email.lower().endswith("@" + _allowed_domain().lower()):
                st.error(f"Only @{_allowed_domain()} addresses are allowed.")
            else:
                code = _generate_otp()
                sent = _send_otp_email(email, code)
                st.session_state.update(
                    otp_code=code, otp_time=time.time(), otp_email=email, otp_attempts=0,
                    otp_dev_mode=not sent,
                )
                st.rerun()
    else:
        email = st.session_state["otp_email"]
        st.write(f"Enter the 6-digit code sent to **{email}**.")
        if st.session_state.get("otp_dev_mode"):
            st.warning(
                f"No SES credentials configured (dev mode) — your code is **"
                f"{st.session_state['otp_code']}**. Set `AWS_ACCESS_KEY_ID` / "
                f"`AWS_SECRET_ACCESS_KEY` / `AWS_SES_REGION` / `EMAIL_FROM` in secrets before "
                f"using this beyond a local pilot."
            )
        code_in = st.text_input("6-digit code", max_chars=6, key="login_code")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify", type="primary"):
                att = st.session_state.get("otp_attempts", 0)
                if att >= MAX_OTP_ATTEMPTS:
                    st.error("Too many attempts. Request a new code.")
                elif time.time() - st.session_state.get("otp_time", 0) > OTP_EXPIRY_SECS:
                    st.error("Code expired. Request a new code.")
                elif code_in == st.session_state.get("otp_code"):
                    st.session_state["auth_email"] = email
                    for k in ("otp_code", "otp_time", "otp_email", "otp_attempts", "otp_dev_mode"):
                        st.session_state.pop(k, None)
                    st.query_params["t"] = _sign_token(email)
                    st.rerun()
                else:
                    st.session_state["otp_attempts"] = att + 1
                    rem = MAX_OTP_ATTEMPTS - st.session_state["otp_attempts"]
                    st.error(f"Incorrect code. {rem} attempt{'s' if rem != 1 else ''} left.")
        with col2:
            if st.button("Use a different email"):
                for k in ("otp_code", "otp_time", "otp_email", "otp_attempts", "otp_dev_mode"):
                    st.session_state.pop(k, None)
                st.rerun()

    st.stop()


def logout_button():
    if st.session_state.get("auth_email"):
        with st.sidebar:
            st.caption(f"Signed in as {st.session_state['auth_email']}")
            if st.button("Sign out"):
                st.session_state.clear()
                st.query_params.clear()
                st.rerun()
