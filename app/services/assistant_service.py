from flask import g, request, jsonify
from ..utils.database import get_db
from ..models.law.assistant import get_legal_response
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_assistant_response():
    # This endpoint can be used for a simple health check or to indicate the assistant is live.
    return {'message': 'assistant is live'}, 200


def process_legal_query_service():
    """
    Service function to process a legal query from a POST request.
    Returns (response_dict, status_code)
    """
    try:
        if not request.is_json:
            logger.warning("Invalid request: Content-Type must be application/json")
            return {
                'error': 'Content-Type must be application/json'
            }, 400
        data = request.get_json()
        query = data.get('query')
        session_id = data.get('session_id')
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Invalid request: Query must be a non-empty string")
            return {
                'error': 'Query must be a non-empty string'
            }, 400
        db = get_db()
        cursor = db.cursor(dictionary=True)
        today = datetime.now().date()
        cursor.execute("""
            SELECT s.max_queries_per_day, t.user_id, t.subscription_id
            FROM transactions t
            JOIN subscriptions s ON t.subscription_id = s.id
            WHERE t.user_id = %s AND s.status = 1 AND t.status = 1
            ORDER BY t.created_at DESC LIMIT 1
        """, (g.user_id,))
        sub = cursor.fetchone()
        if not sub:
            # fallback: try to get the latest transaction for the user (even if not active)
            cursor.execute("""
                SELECT s.max_queries_per_day, t.user_id, t.subscription_id
                FROM transactions t
                JOIN subscriptions s ON t.subscription_id = s.id
                WHERE t.user_id = %s
                ORDER BY t.created_at DESC LIMIT 1
            """, (g.user_id,))
            sub = cursor.fetchone()
            if not sub:
                return {'error': 'No active subscription found'}, 403
        # Count queries used today from messages table
        cursor.execute("""
            SELECT COUNT(*) as used FROM messages m
            JOIN chat_sessions cs ON m.session_id = cs.id
            WHERE cs.user_id = %s AND DATE(m.timestamp) = %s AND m.sender = 'user'
        """, (g.user_id, today))
        used = cursor.fetchone()['used'] if cursor.rowcount else 0
        if used >= sub['max_queries_per_day']:
            return {'error': 'Daily query limit reached'}, 403
        # --- Chat session and message saving logic ---
        import uuid
        # If session_id is not provided or does not exist, create a new session
        if not session_id:
            session_id = str(uuid.uuid4())
            try:
                cursor.execute("""
                    INSERT INTO chat_sessions (id, user_id, session_name, status, created_at, last_active)
                    VALUES (%s, %s, %s, TRUE, NOW(), NOW())
                """, (session_id, g.user_id, f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
                db.commit()
            except Exception as e:
                db.rollback()  # Ignore duplicate key error
        else:
            # Check if session exists for this user
            cursor.execute("SELECT id FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, g.user_id))
            existing = cursor.fetchone()
            if not existing:
                # If session_id provided but not found, create it
                try:
                    cursor.execute("""
                        INSERT INTO chat_sessions (id, user_id, session_name, status, created_at, last_active)
                        VALUES (%s, %s, %s, TRUE, NOW(), NOW())
                    """, (session_id, g.user_id, f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
                    db.commit()
                except Exception as e:
                    db.rollback()  # Ignore duplicate key error
        # Save user message
        user_message_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO messages (id, session_id, sender, message, stream_chunk, status, timestamp)
            VALUES (%s, %s, 'user', %s, FALSE, TRUE, NOW())
        """, (user_message_id, session_id, query))
        db.commit()
        logger.info(f"Processing query: {query}")
        response = get_legal_response(query)
        logger.info(f"Query processed successfully: {query}")
        # Save bot response
        bot_message_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO messages (id, session_id, sender, message, stream_chunk, status, timestamp)
            VALUES (%s, %s, 'bot', %s, FALSE, TRUE, NOW())
        """, (bot_message_id, session_id, response))
        db.commit()
        # After model response, update remaining_queries for user
        # Reset remaining_queries if it's a new day
        cursor.execute("SELECT updated_at, remaining_queries FROM users WHERE id = %s", (g.user_id,))
        user_row = cursor.fetchone()
        last_update = user_row['updated_at'].date() if user_row and user_row['updated_at'] else None
        if last_update != today:
            # Reset to max_queries_per_day for new day
            cursor.execute("UPDATE users SET remaining_queries = %s, updated_at = NOW() WHERE id = %s", (sub['max_queries_per_day'] - 1, g.user_id))
        else:
            # Decrement for same day
            cursor.execute("UPDATE users SET remaining_queries = GREATEST(remaining_queries - 1, 0), updated_at = NOW() WHERE id = %s", (g.user_id,))
        db.commit()
        return {
            'query': query,
            'response': response,
            'status': 'success',
            'queries_left_today': sub['max_queries_per_day'] - used - 1,
            'session_id': session_id,
            'max_queries_per_day': sub['max_queries_per_day'],
            'used_queries_today': used + 1
        }, 200
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            'error': f'An error occurred: {str(e)}',
            'status': 'error'
        }, 500


def get_session_messages(session_id):
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, sender, message, stream_chunk, status, timestamp
        FROM messages
        WHERE session_id = %s
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = cursor.fetchall()
    return {'session_id': session_id, 'messages': messages}, 200

