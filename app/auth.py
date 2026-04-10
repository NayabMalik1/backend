import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.database import get_db
from app.db_models import User

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Plain text (no hashing) – ONLY for development
def verify_password(plain_password, stored_password):
    return plain_password == stored_password

def get_password_hash(password):
    return password  # store as plain text

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def send_reset_email(email, token):
    reset_link = f"http://localhost:3000/reset-password?token={token}"
    print(f"Password reset link for {email}: {reset_link}")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user
import smtplib
from email.message import EmailMessage
import os

def send_reset_email(email, token):
    # Construct reset link
    reset_link = f"http://localhost:3000/reset-password?token={token}"

    # Email content
    subject = "Password Reset Request"
    body = f"""
    Hello,

    You requested a password reset for your Android Malware Detection account.

    Click the link below to reset your password:
    {reset_link}

    This link will expire in 1 hour.

    If you did not request this, please ignore this email.

    Regards,
    Android Malware Detection Team
    """

    # Get email credentials from environment
    smtp_host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_PORT", 587))
    sender_email = os.getenv("EMAIL_HOST_USER")
    sender_password = os.getenv("EMAIL_HOST_PASSWORD")

    if not sender_email or not sender_password:
        print("Email credentials not set. Reset link would have been sent to:", email)
        print("Reset link:", reset_link)
        return

    # Create email message
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    # Send email
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"Password reset email sent to {email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        # Fallback: print to console for debugging
        print(f"Reset link for {email}: {reset_link}")