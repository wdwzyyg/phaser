from pathlib import Path
import logging

from quart import Quart, render_template

app = Quart(
    __name__,
    static_url_path="/static",
    static_folder="dist",
)

def run() -> None:
    logging.basicConfig(level=logging.INFO)
    app.run()

@app.get("/")
async def index():
    return await render_template("manager.html")

@app.get("/client")
async def client():
    return await render_template("dashboard.html")