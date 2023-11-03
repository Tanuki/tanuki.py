import asyncio
import os
from time import time
from typing import AsyncIterable, Generator

import openai
from dotenv import load_dotenv

from monkey_patch.monkey import Monkey as monkey

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@monkey.patch
async def iter_presidents() -> AsyncIterable[str]:
    """List the presidents of the United States"""


@monkey.patch
async def iter_prime_ministers() -> AsyncIterable[str]:
    """List the prime ministers of the UK"""


@monkey.patch
async def tell_me_more_about(topic: str) -> str:
    """"""


async def describe_presidents():
    # For each president listed, generate a description concurrently
    start_time = time()
    print(start_time)
    tasks = []
    iter = iter_prime_ministers()
    async for president in iter:
        print(f"Generating description for {president}")
        #task = asyncio.create_task(tell_me_more_about(president))
        #tasks.append(task)

    #descriptions = await asyncio.gather(*tasks)

    #print(f"Generated {len(descriptions)} descriptions in {time() - start_time} seconds")
#    return descriptions


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(describe_presidents())
    loop.close()

if __name__ == '__main__':
    main()