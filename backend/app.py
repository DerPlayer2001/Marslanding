from flask import Flask, request
from flask_cors import CORS
import psycopg2
import os

app = Flask(__name__)
CORS(app)

conn = psycopg2.connect(
    database="postgres",
    user="web",
    password="web",
    host=os.environ["DB_URL"],
    port="5432",
)


@app.route("/", methods=["GET", "POST"])
def index():
    cursor = conn.cursor()
    print(request.method)
    match request.method:
        case "GET":
            cursor.execute("SELECT player, score FROM scores ORDER BY score DESC")
            return cursor.fetchall()
        case "POST":
            cursor = conn.cursor()
            json = request.get_json()
            cursor.execute(
                "INSERT INTO scores (player, score) VALUES (%s, %s)",
                (json["player"], json["score"]),
            )
            conn.commit()
            return "OK"
        case _:
            return "Invalid method"


if __name__ == "__main__":
    app.run(debug=True)
