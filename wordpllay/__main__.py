"""
The wordpllay server.
"""
import asyncio
import json
from pathlib import Path
import logging
from typing import (
    Any,
    Callable,
    Coroutine,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
)
import numpy.random
from aiohttp import web
import jinja2


PROMPT_LIMIT: int = 200  # characters


_WORDLIST: Optional[Sequence[str]] = None


def get_random_words(seed: int) -> Sequence[str]:
    """
    Get a random number of words from the wordlist, based on a session seed.
    """
    # pylint: disable-next=global-statement
    global _WORDLIST
    if _WORDLIST is None:
        with (SERVER_ROOT / "words.txt").open("r", encoding="ascii") as f:
            _WORDLIST = f.read().splitlines()

    gen = numpy.random.default_rng(seed)
    word_count = gen.integers(1, 3)
    return gen.choice(_WORDLIST, word_count, replace=False)


def success(a: Any) -> asyncio.Future[Any]:
    """
    Create a future that is already resolved with the given value.
    """
    fut = asyncio.Future()
    fut.set_result(a)
    return fut


async def index_handler(_) -> web.FileResponse:
    """
    Serve the index page.
    """
    return web.FileResponse(SERVER_ROOT / "static" / "index.html")


async def dumb_llm(
    prompt: str, chunk_writer: Callable[[str], Any] = None
) -> str:
    """
    Run the language model on a prompt.
    """
    output = f'Looking at the prompt "{prompt}", I think that I should respond with... uh... idk.'

    for word in output.split():
        if chunk_writer:
            await chunk_writer(" " + word)
        await asyncio.sleep(0.2)

    return output


class SingleTask:
    """
    A coroutine holder that can only run one task at a time.
    """

    __slots__ = ("task",)

    def __init__(self, task: Optional[asyncio.Task] = None) -> None:
        self.task = task

    async def start(self, task: asyncio.Task) -> None:
        """
        Schedule a task to run, cancelling any existing task.
        """
        if self.task is not None:
            await self.stop("New task started.")
        self.task = task

    async def stop(self, msg: Optional[str] = None) -> None:
        """
        Cancel the currently running task.
        """
        if self.task is not None:
            self.task.cancel(msg)
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


DIFFICULTY_LIST = ["easy", "normal", "hard"]

DifficultyType = Literal["easy", "normal", "hard"]


class GameSession(NamedTuple):
    """
    Handle a single websocket connection from a client.
    """

    app: web.Application
    ws: web.WebSocketResponse
    sid: int
    random_words: Sequence[str]
    prompt_session: SingleTask

    @property
    def jinja_env(self) -> jinja2.Environment:
        """
        Extracts and returns the jinja2 environment from the current app.

        Shortcut method.
        """
        return self.app["jinja2_env"]

    @staticmethod
    async def from_request(request: web.Request) -> "GameSession":
        """
        Create a game session from an incoming websocket upgrade request.
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        return GameSession.from_websocket(request.app, ws)

    @staticmethod
    def from_websocket(
        app: web.Application,
        ws: web.WebSocketResponse,
    ) -> "GameSession":
        """
        Create a game session from an established websocket connection.
        """
        sid = numpy.random.randint(0, 2**32)
        random_words = get_random_words(sid)
        logging.info(
            "Session %s: Starting with random words: %s", sid, random_words
        )
        return GameSession(app, ws, sid, random_words, SingleTask())

    def send_gamearea(self) -> Coroutine[None, None, None]:
        """
        Send the textarea form to the client.
        """
        try:
            rendered_template = self.jinja_env.get_template(
                "gamearea.html"
            ).render(
                random_words=self.random_words,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Session %s: Exception while rendering template: %s",
                self.sid,
                exc,
            )
            raise exc

        logging.info("Session %s: Sending rendered template", self.sid)

        return self.ws.send_str(rendered_template)

    def send_output(
        self,
        msg: str,
        state: Literal["winner", "loser", "processing", "ready"],
        stopped: str = "",
        classes: str = "",
    ) -> Coroutine[None, None, None]:
        """
        Send a completed output message to the client.
        """
        try:
            rendered_template = self.jinja_env.get_template(
                "gameoutput.html"
            ).render(
                text=msg,
                classes=classes,
                stopped=stopped,
                state=state,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Session %s: Exception while rendering template: %s",
                self.sid,
                exc,
            )
            raise exc

        logging.info("Session %s: Sending output template", self.sid)

        return self.ws.send_str(rendered_template)

    def send_err(self, msg: str) -> Coroutine[None, None, None]:
        """
        Send an error message to the client.
        """
        return self.send_output(msg, state="loser", classes="error_output")

    def send_output_chunk(self, chunk: str) -> Coroutine[None, None, None]:
        """
        Send a chunk of output to the client.
        """
        try:
            return self.ws.send_str(
                self.jinja_env.get_template("gamepartialoutput.html").render(
                    text=chunk
                )
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Session %s: Exception while rendering partialoutput: %s",
                self.sid,
                exc,
            )
            return success(None)

    async def execute_prompt(
        self, prompt: str, difficulty: DifficultyType
    ) -> None:
        """
        Execute a prompt from the client.
        """

        stopped_msg = ""
        if difficulty not in DIFFICULTY_LIST:
            logging.info(
                'Session %s: Difficulty %r unrecognized, assumed "easy"',
                self.sid,
                prompt,
            )
            stopped_msg = (
                f'Difficulty "{difficulty}" not recognized. Assuming "easy"...'
            )
            difficulty = "easy"

        if difficulty != "easy":
            contained_words = [
                word for word in self.random_words if word in prompt
            ]
            if contained_words:
                await self.send_output(
                    "Nope.",
                    state="loser",
                    stopped=stopped_msg
                    + f"Your prompt contains {contained_words} which are in"
                    f' your target words! On "{difficulty}" difficulty,'
                    " this is not allowed.",
                )
                return

        await self.send_output("", state="processing", stopped=stopped_msg)

        logging.info("Session %s: Executing prompt: %s", self.sid, prompt)

        collected_chunks = []

        async def chunk_writer(chunk: str) -> None:
            """
            Collect chunks of output.
            """
            chunk = chunk.replace("\n", "<br/>")
            collected_chunks.append(chunk)
            await self.send_output_chunk(chunk)
            await asyncio.sleep(0)

        error_message = ""
        stopped = ""

        try:
            result = (
                await SERVER_LLM.generate_async(
                    prompt,
                    chunk_writer=chunk_writer,
                    max_tokens=100,
                )
            ).replace("\n", "<br/>")
        except asyncio.CancelledError as exc:
            logging.info(
                "Session %s: Prompt execution cancelled: %s", self.sid, exc
            )
            stopped = str(exc)
            result = "".join(collected_chunks)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception(
                "Session %s: Exception while executing prompt: %s",
                self.sid,
                exc,
            )
            error_message = f"<br/>ERROR: {exc}"
            result = "".join(collected_chunks)

        won = all(word in result for word in self.random_words)

        await self.send_output(
            result + error_message,
            state="winner" if won else "loser",
            stopped=stopped,
            classes="error_output" if error_message else "",
        )

    def handle_form_message(
        self, data: str
    ) -> Coroutine[None, None, Optional[Literal[True]]]:
        """
        Handle a form message from the client.
        """
        logging.info("Session %s: Received message:\n%s", self.sid, data)

        try:
            recieved = json.loads(data)
        except json.decoder.JSONDecodeError as err:
            logging.warning(
                "Session %s: Unable to decode JSON sent by client. Error: %s",
                self.sid,
                err,
            )
            return self.send_err(
                "ERROR: Unable to decode JSON sent by client."
            )

        if not isinstance(recieved, dict):
            logging.warning(
                "Session %s: Client sent non-object JSON: %s",
                self.sid,
                recieved,
            )
            return self.send_err(
                "ERROR: Unable to decode JSON sent by client."
            )

        if "stop" in recieved:
            logging.info("Session %s: Client requested stop.", self.sid)
            return self.prompt_session.stop("Client requested stop.")
        if "next_game" in recieved:
            logging.info("Session %s: Client requested new game.", self.sid)
            return success(True)
        if "reset_output" in recieved:
            logging.info(
                "Session %s: Client requested output reset.", self.sid
            )
            return self.send_output("", state="ready")
        if "new_game" in recieved:
            logging.info("Session %s: Client requested new game.", self.sid)
            return success(True)

        if (prompt := recieved.get("prompt")) is None:
            logging.info(
                "Session %s: Client sent JSON without a prompt: %s",
                self.sid,
                recieved,
            )
            return self.send_err(
                "ERROR: JSON sent by client does not contain a prompt."
            )

        if len(prompt) > 200:
            logging.info(
                "Session %s: Client sent prompt that is too long: %s",
                self.sid,
                recieved,
            )
            return self.send_err(
                "ERROR: Prompt sent by client is too long. "
                "Please keep under 200 characters."
            )

        difficulty = recieved.get("difficulty")

        return self.prompt_session.start(
            asyncio.create_task(
                self.execute_prompt(prompt, difficulty), name="prompt_exec"
            )
        )

    async def receive_messages(self) -> None:
        """
        Retrieve messages until the game session is completed or the
        """
        logging.info("Session %s: Waiting for messages", self.sid)
        async for msg in self.ws:
            if msg.type == web.WSMsgType.TEXT:
                new_game = await self.handle_form_message(msg.data)
                if new_game:
                    break
            elif msg.type == web.WSMsgType.ERROR:
                logging.warning(
                    "Session %s: WS connection closed with exception %s",
                    self.sid,
                    self.ws.exception(),
                )
                break
            elif msg.type == web.WSMsgType.CLOSE:
                logging.info(
                    "Session %s: Closing normally by client request.", self.sid
                )
                await self.ws.close()
                break
            elif msg.type == web.WSMsgType.CLOSED:
                logging.info(
                    "Session %s: Attempted to read message on closed connection."
                )
                break
            else:
                logging.warning(
                    "Session %s: Unhandled websocket message type: %s",
                    self.sid,
                    msg.type,
                )


async def websocket_handler(request) -> web.WebSocketResponse:
    """
    Serve the wordpllay game via a websocket.
    """
    game = await GameSession.from_request(request)

    while not game.ws.closed:
        await game.send_gamearea()
        await game.receive_messages()
        if not game.ws.closed:
            game = GameSession.from_websocket(request.app, game.ws)

    return game.ws


def init_app(server_root: Path) -> web.Application:
    """
    Initialize the wordpllay application object.
    """
    app = web.Application()
    app.add_routes(
        [
            web.static("/static", server_root / "static"),
            web.get("/", index_handler),
            web.get("/ws", websocket_handler),
        ]
    )
    app["jinja2_env"] = jinja2.Environment(
        loader=jinja2.FileSystemLoader(server_root / "templates")
    )
    return app


def main() -> None:
    "Entry point."
    app = init_app(SERVER_ROOT)
    web.run_app(app, host="localhost", port=8000)


if __name__ == "__main__":
    SERVER_ROOT = Path(__file__).parent
    logging.basicConfig(level=logging.INFO)

    import sys

    if len(sys.argv) < 2:
        print(
            "WARNING: No language model specified. Using an extremely dumb llm."
        )
        SERVER_LLM = dumb_llm
    else:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"ERROR: Model path {model_path} does not exist.")
            sys.exit(1)

        from wordpllay.llm import LLM

        SERVER_LLM = LLM(model_path)

    main()
