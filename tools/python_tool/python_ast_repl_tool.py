"""A tool for running python code in a REPL."""

import ast
import asyncio
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field


import functools
import logging
import multiprocessing
import sys
from io import StringIO
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: Optional[Dict] = Field(default_factory=dict, alias="_globals")
    locals: Optional[Dict] = Field(default_factory=dict, alias="_locals")

    @classmethod
    def worker(
        cls,
        command: str,
        globals: Optional[Dict],
        locals: Optional[Dict],
        queue: multiprocessing.Queue,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(
                target=self.worker, args=(command, self.globals, self.locals, queue)
            )

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, self.globals, self.locals, queue)
        # get the result from the worker function
        return queue.get()



def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.
    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    return query


class PythonREPLTool(BaseTool):
    """A tool for running python code in a REPL."""

    name: str = "Python_REPL"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "If you want to see the output of a value, you should print it out "
        "with `print(...)`."
    )
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def _run(
        self,
        query: str
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
        return self.python_repl.run(query)

    async def _arun(
        self,
        query: str,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.sanitize_input:
            query = sanitize_input(query)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.run, query)

        return result


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


class PythonAstREPLTool(BaseTool):
    """A tool for running python code in a REPL."""

    name: str = "python_repl_ast"
    description: str = (
        "A Python shell. Use this to execute python commands. "
        "Input should be a valid python command. "
        "When using this tool, sometimes output is abbreviated - "
        "make sure it does not look abbreviated before using it in your answer."
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs


    def _run(
        self,
        query: str
    ) -> str:
        """Use the tool."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        query: str
    ) -> Any:
        """Use the tool asynchronously."""

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._run, query)

        return result
