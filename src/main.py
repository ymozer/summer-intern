import asyncio
import sys
import os
import aiofiles
import redis.asyncio as redis
# Redis exceptions
from redis.exceptions import ConnectionError,TimeoutError
import uvloop
import argparse

from Agent import Agent
from Master import Master
from Slave import Slave
from log.color import LogColor

log=LogColor()

def remove_last_comma_from_file(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Check if the last line contains a comma
    if lines and lines[-1].strip().endswith(','):
        lines[-1] = lines[-1].strip()[:-1]  # Remove the last comma

    with open(file_path, 'w') as f:
        f.writelines(lines)

if __name__ == '__main__':
    slave_list=['agent2','agent3']

    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--id', type=str, default='agent_1', help='Agent ID')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Redis IP')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')

    # create output dir
    if not os.path.exists('output'):
        os.makedirs('output')

    async def main():
        args = parser.parse_args()
        log.p_ok(f"Agent ID: {args.id}")
        log.p_ok(f"Redis IP: {args.ip}")
        log.p_ok(f"Redis port: {args.port}")

        master = Master('agent1', args.ip, args.port, 'stream_1', 'group_1')
        slave = Slave(slave_list[0], args.ip, args.port, 'stream_1', 'group_2')
        slave2 = Slave(slave_list[1], args.ip, args.port, 'stream_1', 'group_3')
        await asyncio.gather(master.main(), slave.main(),slave2.main())


    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        try:
            runner.run(main())
        except KeyboardInterrupt:
            log.p_fail("KeyboardInterrupt")
            for i in slave_list:
                with open(f"output/{i}.json", mode='a') as f:
                    # The last line of the JSON file must not have a comma, delete it
                    f.seek(f.tell() - 1, os.SEEK_SET)
                    remove_last_comma_from_file(f"output/{i}.json")
                    # code exits, so write the last line as ]
                    f.write("\n]")
            sys.exit(1)
