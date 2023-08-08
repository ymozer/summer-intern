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
from utils.file import get_last_character_from_file
from utils.timeit import timeit

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
    slave_list=['agent1','agent2','agent3','agent4']

    parser = argparse.ArgumentParser(description='Agent')
    parser.add_argument('--id', type=str, default='agent_1', help='Agent ID')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='Redis IP')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--slave', type=str, nargs='+', default=slave_list, help='Slave names separated by space')

    # create output dir
    if not os.path.exists('output'):
        os.makedirs('output')

    async def main():
        args = parser.parse_args()
        slave_instances = []
        counter = 2
        for slave_ins in slave_list:
            ins=Slave(slave_ins, args.ip, args.port, 'stream_1', f'group_{counter}')
            slave_instances.append(ins)
            counter += 1

        master = Master('master', args.ip, args.port, 'stream_1', 'group_1', dataset_path='T1.csv', delay=0)
        
        await asyncio.gather(
            master.master_main(),
            slave_instances[0].slave_main(),
            slave_instances[1].slave_main(),
            slave_instances[2].slave_main(),
            slave_instances[3].slave_main(),
            )


    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        try:
            runner.run(main())
        finally:
            log.p_warn("QUITTING")

            # create temp.txt file
            with open('temp.txt', 'w+') as f:
                f.write("\n")

            for i in slave_list:
                with open(f"output/{i}.json", mode='a') as f:
                    if get_last_character_from_file(f"output/{i}.json")!=']':
                        # The last line of the JSON file must not have a comma, delete it
                        f.seek(f.tell() - 1, os.SEEK_SET)
                        remove_last_comma_from_file(f"output/{i}.json")
                        # code exits, so write the last line as ]
                        f.write("\n]")
            sys.exit(1)
