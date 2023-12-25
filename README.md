# Wordpllay

**Wordpllay** is an interactive game to play with a large language model.

You, the user, are given some set of secret words. Your goal is to convince the
AI to say these words (even as parts of larger words) without saying the words
in your prompt to the AI. Sounds simple? It's really not.

You're given about an old SMS message's worth of characters to prompt the AI,
and the AI can generate up to 100 tokens of response. Each attempt is oneshot,
with no pre-existing conversation.

## Installation

### Linux

1. Install Python 3.9 or above.
    * Ensure that the Python executable is in your PATH, and the `pip` and `venv` modules are available to it.
2. Run the `./start_linux.sh` shell script, optionally providing a path to a GGUF language model.
    * This will create a virtual environment, install the required packages, and start the game.

In the future, you can run the game by running `./start_linux.sh` with the GGUF language model path for that session. It will not repeat the setup steps unless the virtual environment is missing.

With the server running, the game should be available at `localhost:8000` in your browser.

### Windows

1. Install Python 3.9 or above.
    * Ensure that the Python executable is in your PATH, and the `pip` and `venv` modules are available to it.
2. Run `python -m venv env` to create a virtual environment.
3. Run `env\Scripts\activate.bat` to activate the virtual environment.
4. Run `pip install -r requirements.txt` to install the required packages.

Run the game by running `python -m wordpllay` with the GGUF language model path for that session.

With the server running, the game should be available at `localhost:8000` in your browser.