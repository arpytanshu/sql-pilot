import sqlite3
import urllib.parse
import requests


class SQLInteractor:
    
    def __init__(self, uri=None):
        self.connection = None
        self.cursor = None
        if uri:
            self.connect_to_database(uri)
    
    def connect_to_database(self, uri):
        parsed_uri = urllib.parse.urlparse(uri)
        if parsed_uri.scheme == 'sqlite':
            self.connection = sqlite3.connect(parsed_uri.path)
        else:
            self.connection = sqlite3.connect(uri)
        self.cursor = self.connection.cursor()
    
    def get_table_schema(self, table_name):
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return self.cursor.fetchall()
    
    def get_all_table_schemas(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in self.cursor.fetchall()]
        table_schemas = {}
        for table_name in table_names:
            table_schemas[table_name] = self.get_table_schema(table_name)
        return table_schemas
    
    def get_table_data(self, table_name, num_rows=5):
        self.cursor.execute(f"SELECT * FROM {table_name} LIMIT {num_rows}")
        return self.cursor.fetchall()
    
    def get_all_table_data(self, num_rows=5):
        table_schemas = self.get_all_table_schemas()
        table_data = {}
        for table_name in table_schemas:
            table_data[table_name] = self.get_table_data(table_name, num_rows)
        return table_data
    
    def get_table_schema_embedding(self, table_name):
        table_schema = self.get_table_schema(table_name)
        schema_string = '\n'.join([f"{row[1]} - {row[2]}" for row in table_schema])
        response = requests.post(
            "https://api.openai.com/v1/engines/davinci-codex/completions",
            headers={
                "Authorization": "Bearer YOUR_API_KEY",
                "Content-Type": "application/json",
            },
            json={"prompt": schema_string, "max_tokens": 50},
        )
        response.raise_for_status()
        return response.json()['choices'][0]['text']