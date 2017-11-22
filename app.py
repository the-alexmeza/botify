import os
import sys
import json
import random

from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import Flask, request

app = Flask(__name__)

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
            'bot_id' : 'b7fc39f3a81ce8245b2a6ed00f',
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
