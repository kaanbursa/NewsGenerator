import os
from flask import Flask, request, send_file, render_template
from model import extend



app = Flask(__name__)
app.static_folder = 'static'


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    data = request.form.get('text')
    return extend(data)

if __name__ == "__main__":
    print('Port is running')
    app.run('0.0.0.0', os.environ.get('PORT',8080))
