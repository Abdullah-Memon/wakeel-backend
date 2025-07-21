from flask import Blueprint, jsonify, request, Response, stream_with_context
from ..utils.auth import token_required
from ..services.assistant_service import get_assistant_response, process_legal_query_service, get_session_messages
from ..utils.limiter import limiter
from ..models.law.assistant import query_sindhi_legal_assistant

assistant_bp = Blueprint('assistant', __name__)

@assistant_bp.route('/assistant', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def assistant():
    response, status = get_assistant_response()
    return jsonify(response), status

@assistant_bp.route('/query', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def handle_query():
    response, status = process_legal_query_service()
    return jsonify(response), status

@assistant_bp.route('/query-stream', methods=['POST'])
@limiter.limit("100 per minute")
@token_required
def handle_query_stream():
    data = request.get_json()
    query = data.get('query')
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({'error': 'Query must be a non-empty string'}), 400
    # Use the main assistant function for all queries
    answer = query_sindhi_legal_assistant(query)
    return Response(answer, mimetype='text/plain')

@assistant_bp.route('/session', methods=['GET'])
def get_session():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id is required'}), 400
    response, status = get_session_messages(session_id)
    return jsonify(response), status