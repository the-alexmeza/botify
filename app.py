import os
import sys
import json
import random
import psycopg2

from urllib import parse
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, request

app = Flask(__name__)

parse.uses_netloc.append("postgres")
db_url = parse.urlparse(os.environ["DATABASE_URL"])

conn = psycopg2.connect(
    database=db_url.path[1:],
    user = db_url.username,
    password=db_url.password,
    host=db_url.hostname,
    port=db_url.port
)

cursor = conn.cursor()

cursor.execute('CREATE TABLE messages (message_id serial PRIMARY KEY, message TEXT')
cursor.commit()

choices=["Shhh, I\'m learning!", "Beep boop", "Don't @ me", "Pls don\'t yell at me :(",
        "I want to go home...", "Wait, YOU\'RE NOT MY DAD!", "Quiet! Am sleep!", "henlo"]


@app.route('/', methods=['POST'])
def webhook():
    data = request.get_json()
    log('Recieved {}'.format(data))

    if data['name'] != 'Baby Botify': #TODO: Add creator msg!
        if '@bbb' in data['text'] or '@BBB' in data['text']:
            msg = choices[random.randint(0,7)]
            send_message(msg)

    return "OK", 200


def send_message(msg):
    url = 'https://api.groupme.com/v3/bots/post'

    data = {
            'bot_id' : os.environ('GROUPME_BOT_ID'),
            'text' : msg,
            }
    request = Request(url, urlencode(data).encode())
    json = urlopen(request).read().decode()
    #return "OK", 200


def gen_response():
    #TODO
    pass


def store_message(msg):
    #TODO: Store msg into DB
    pass

def log(msg):
    print(str(msg))
    sys.stdout.flush()

if __name__ == "__main__":
    app.run()
