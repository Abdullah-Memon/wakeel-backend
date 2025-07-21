import mysql.connector
from flask import g, current_app
from ..models.schema import get_schema

def get_db():
    if 'db' not in g:
        g.db = mysql.connector.connect(**current_app.config['DB_CONFIG'])
    return g.db

def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    cursor = db.cursor()
    schemas = get_schema()
    
    for table_name, schema in schemas.items():
        cursor.execute(schema)
    db.commit()