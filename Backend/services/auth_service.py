"""
Enhanced JWT Authentication Service - CORRECTED
Fixed JWT, bcrypt, database path issues, and proper error handling
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import bcrypt
import jwt  # PyJWT
from dotenv import load_dotenv
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Load environment variables
load_dotenv("enhanced.env")

logger = logging.getLogger(__name__)

# FIXED: Correct JWT Configuration with environment variables
JWT_SECRET = os.getenv("SECRET_KEY", "biorag_production_secret_key_change_immediately_in_production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours default
DATABASE_PATH = os.getenv("DATABASE_PATH", "users.db")  # FIXED: Use env variable

security = HTTPBearer()

class EnhancedAuthService:
    """Enhanced Authentication service - CORRECTED VERSION"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_PATH  # FIXED: Use environment variable
        self._ensure_database()

    def _ensure_database(self):
        """Initialize SQLite database with proper configuration"""
        # CRITICAL FIX: Create directory if it doesn't exist
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"✅ Created database directory: {db_dir}")
        
        # FIXED: Add check_same_thread=False for FastAPI compatibility
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
        except sqlite3.OperationalError as e:
            logger.error(f"❌ Failed to open database at {self.db_path}: {e}")
            # Fallback to current directory
            self.db_path = "users.db"
            logger.info(f"⚠️ Falling back to: {os.path.abspath(self.db_path)}")
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()

        # Users table with enhanced fields
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                preferences TEXT DEFAULT '{}',
                is_active BOOLEAN DEFAULT 1,
                query_count INTEGER DEFAULT 0,
                total_processing_time REAL DEFAULT 0.0
            )
        """)

        # Chat history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                preview TEXT NOT NULL,
                query_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                citations_count INTEGER DEFAULT 0,
                search_results_count INTEGER DEFAULT 0,
                query_analysis TEXT DEFAULT '{}',
                enhancement_applied BOOLEAN DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Evaluation history table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                results TEXT NOT NULL,
                num_questions INTEGER DEFAULT 0,
                citation_accuracy REAL DEFAULT 0.0,
                answer_relevance REAL DEFAULT 0.0,
                biomedical_f1 REAL DEFAULT 0.0,
                comprehensive_score REAL DEFAULT 0.0,
                benchmark_comparison TEXT DEFAULT '{}',
                duration_seconds REAL DEFAULT 0.0
            )
        """)

        # User feedback table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                chat_history_id INTEGER,
                rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                feedback_text TEXT,
                feedback_type TEXT DEFAULT 'general',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (chat_history_id) REFERENCES chat_history (id)
            )
        """)

        # System metrics table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        """)

        # Create demo user
        demo_email = "user@example.com"  # Changed to match frontend
        demo_password = "password123"
        cur.execute("SELECT id FROM users WHERE email = ?", (demo_email,))
        if not cur.fetchone():
            # FIXED: Proper bcrypt usage with rounds parameter
            pw_hash = bcrypt.hashpw(demo_password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
            cur.execute(
                "INSERT INTO users (email, password_hash, name, preferences) VALUES (?, ?, ?, ?)",
                (demo_email, pw_hash, "Demo User", json.dumps({
                    "preferred_answer_length": "comprehensive",
                    "citation_style": "numbered",
                    "domain_preference": "general",
                    "evaluation_notifications": True
                })),
            )
            logger.info(f"✅ Created demo user: {demo_email} / {demo_password}")

        # Create admin user
        admin_email = "admin@biochat.ai"
        admin_password = "admin2024"
        cur.execute("SELECT id FROM users WHERE email = ?", (admin_email,))
        if not cur.fetchone():
            pw_hash = bcrypt.hashpw(admin_password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
            cur.execute(
                "INSERT INTO users (email, password_hash, name, preferences) VALUES (?, ?, ?, ?)",
                (admin_email, pw_hash, "System Administrator", json.dumps({
                    "role": "admin",
                    "access_evaluation": True,
                    "access_metrics": True,
                    "notification_level": "all"
                })),
            )
            logger.info(f"✅ Created admin user: {admin_email} / {admin_password}")

        conn.commit()
        conn.close()
        logger.info(f"✅ Database initialized at {os.path.abspath(self.db_path)}")

    async def create_user(
        self, email: str, password: str, name: Optional[str] = None, preferences: Optional[Dict] = None
    ) -> Tuple[Dict, str]:
        """Create new user with validation"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")

        # FIXED: Proper database connection with thread safety
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()

        # Check if user exists
        cur.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cur.fetchone():
            conn.close()
            raise ValueError("User already exists")

        # FIXED: Secure password hashing
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
        
        if not name:
            name = email.split("@")[0].title()

        default_prefs = {
            "preferred_answer_length": "comprehensive",
            "citation_style": "numbered",
            "domain_preference": "general",
            "evaluation_notifications": False,
            "theme": "light",
        }
        if preferences:
            default_prefs.update(preferences)

        cur.execute(
            "INSERT INTO users (email, password_hash, name, preferences) VALUES (?, ?, ?, ?)",
            (email, pw_hash, name, json.dumps(default_prefs)),
        )
        user_id = cur.lastrowid
        conn.commit()
        conn.close()

        user = {
            "id": user_id,
            "email": email,
            "name": name,
            "preferences": default_prefs,
            "created_at": datetime.utcnow().isoformat(),
            "query_count": 0,
        }
        token = self.create_access_token(user_id)
        logger.info(f"✅ Created user: {email} (id={user_id})")
        return user, token

    async def authenticate_user(self, email: str, password: str) -> Tuple[Dict, str]:
        """Authenticate user with proper verification"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        
        cur.execute(
            """SELECT id, email, password_hash, name, preferences, query_count, total_processing_time, last_login
               FROM users WHERE email = ? AND is_active = 1""",
            (email,),
        )
        row = cur.fetchone()
        
        if not row:
            conn.close()
            raise ValueError("Invalid credentials or account disabled")

        user_id, db_email, stored_hash, name, prefs_json, query_count, total_time, last_login = row
        
        # FIXED: Proper bcrypt verification
        if not bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
            conn.close()
            raise ValueError("Invalid credentials")

        # Update last login
        cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

        try:
            prefs = json.loads(prefs_json) if prefs_json else {}
        except json.JSONDecodeError:
            prefs = {}

        user = {
            "id": user_id,
            "email": db_email,
            "name": name,
            "preferences": prefs,
            "query_count": query_count or 0,
            "total_processing_time": total_time or 0.0,
            "last_login": last_login,
        }
        token = self.create_access_token(user_id)
        logger.info(f"✅ User authenticated: {email}")
        return user, token

    def create_access_token(self, user_id: int) -> str:
        """Create JWT token with proper expiration"""
        expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "user_id": user_id,
            "iat": datetime.utcnow(),
            "exp": expires,
            "version": "2.0",
            "features": ["enhanced_rag", "citation_validation", "evaluation"],
        }
        # FIXED: Proper JWT encoding
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user info"""
        try:
            # FIXED: Proper JWT decoding with error handling
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("user_id")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")

            # Get user info from database
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = conn.cursor()
            cur.execute(
                """SELECT id, email, name, preferences, query_count, total_processing_time, is_active
                   FROM users WHERE id = ?""",
                (user_id,),
            )
            row = cur.fetchone()
            conn.close()

            if not row or not row[6]:  # Check is_active
                raise HTTPException(status_code=401, detail="User not found or inactive")

            try:
                prefs = json.loads(row[3]) if row[3] else {}
            except json.JSONDecodeError:
                prefs = {}

            return {
                "id": row[0],
                "email": row[1],
                "name": row[2],
                "preferences": prefs,
                "query_count": row[4] or 0,
                "total_processing_time": row[5] or 0.0,
                "token_version": payload.get("version", "1.0"),
                "features": payload.get("features", []),
            }

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(status_code=401, detail="Token verification failed")

    async def save_chat_history(
        self,
        user_id: int,
        title: str,
        preview: str,
        query_text: str,
        response_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save chat history with metadata"""
        metadata = metadata or {}
        confidence_score = float(metadata.get("confidence_score", 0.0))
        processing_time = float(metadata.get("processing_time", 0.0))
        citations_count = int(len(metadata.get("citations", [])))
        search_results_count = int(metadata.get("search_results_count", 0))
        query_analysis = json.dumps(metadata.get("query_analysis", {}))
        enhancement_applied = 1 if metadata.get("enhancement_applied", False) else 0

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO chat_history
               (user_id, title, preview, query_text, response_text, confidence_score,
                processing_time, citations_count, search_results_count, query_analysis,
                enhancement_applied, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, title, preview, query_text, response_text, confidence_score,
                processing_time, citations_count, search_results_count, query_analysis,
                enhancement_applied, json.dumps(metadata),
            ),
        )
        chat_id = cur.lastrowid
        
        # Update user stats
        cur.execute(
            """UPDATE users SET query_count = query_count + 1, total_processing_time = total_processing_time + ? WHERE id = ?""",
            (processing_time, user_id),
        )
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Chat history saved for user {user_id} (id={chat_id})")
        return chat_id

    async def get_user_history(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's chat history"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()
        cur.execute(
            """SELECT id, title, preview, timestamp, confidence_score, processing_time,
                      citations_count, search_results_count, enhancement_applied
               FROM chat_history WHERE user_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (user_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
        
        history: List[Dict[str, Any]] = []
        for r in rows:
            history.append({
                "id": r[0],
                "title": r[1],
                "preview": r[2],
                "timestamp": r[3],
                "confidence_score": r[4] or 0.0,
                "processing_time": r[5] or 0.0,
                "citations_count": r[6] or 0,
                "search_results_count": r[7] or 0,
                "enhancement_applied": bool(r[8]),
                "user_id": user_id,
            })
        return history

    async def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cur = conn.cursor()

        # User basic stats
        cur.execute(
            "SELECT query_count, total_processing_time, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
        )
        user_stats = cur.fetchone()

        # Chat history stats
        cur.execute(
            """SELECT COUNT(*) as total_chats, AVG(confidence_score) as avg_confidence,
                      AVG(processing_time) as avg_processing_time, SUM(citations_count) as total_citations,
                      COUNT(CASE WHEN enhancement_applied = 1 THEN 1 END) as enhanced_answers
               FROM chat_history WHERE user_id = ?""",
            (user_id,),
        )
        chat_stats = cur.fetchone()

        # Recent activity
        cur.execute(
            "SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND timestamp > datetime('now', '-7 days')",
            (user_id,),
        )
        recent_activity = cur.fetchone()[0]
        conn.close()

        if not user_stats:
            return {}

        total_chats = chat_stats[0] or 0
        enhanced_answers = chat_stats[4] or 0

        return {
            "total_queries": user_stats[0] or 0,
            "total_processing_time": user_stats[1] or 0.0,
            "member_since": user_stats[2],
            "last_login": user_stats[3],
            "total_chats": total_chats,
            "average_confidence": chat_stats[1] or 0.0,
            "average_processing_time": chat_stats[2] or 0.0,
            "total_citations_received": chat_stats[3] or 0,
            "enhanced_answers": enhanced_answers,
            "recent_activity_7days": recent_activity or 0,
            "enhancement_rate": (enhanced_answers / max(total_chats, 1)) if total_chats else 0.0,
        }

# FIXED: Proper singleton pattern for service instance
_auth_service_instance: Optional[EnhancedAuthService] = None

def get_auth_service() -> EnhancedAuthService:
    """Get auth service singleton"""
    global _auth_service_instance
    if _auth_service_instance is None:
        _auth_service_instance = EnhancedAuthService()
    return _auth_service_instance

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    auth = get_auth_service()
    user = auth.verify_token(credentials.credentials)
    logger.debug(f"User {user.get('email')} accessed system (queries: {user.get('query_count')})")
    return user

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[Dict[str, Any]]:
    """Dependency to get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    try:
        auth = get_auth_service()
        return auth.verify_token(credentials.credentials)
    except HTTPException:
        return None

# Backward compatibility alias
AuthService = EnhancedAuthService