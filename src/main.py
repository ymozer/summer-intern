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
        for slave_ins in args.slave:
            ins=Slave(slave_ins, args.ip, args.port, 'stream_1', f'group_{counter}')
            slave_instances.append(ins.slave_main())
            counter += 1

        master = Master('master', args.ip, args.port, 'stream_1', 'group_1', dataset_path='T1.csv', delay=0)
        
        await asyncio.gather(
            master.master_main(),
            *slave_instances
            )
        
        import json
        info = await Agent.r.execute_command('XINFO', 'GROUPS', Agent.stream_name)
        dictt = Master.decode_list_of_bytes(info)
        json_data = json.dumps(dictt, indent=4)
        # pretty print
        log.p_ok(f"data: {json_data}")



    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        try:
            runner.run(main())
        finally:
            log.p_warn("QUITTING")

            # create temp.txt file
            with open('temp.txt', 'w+') as f:
                f.write("\n")
            
            # list all files in output dir
            output_files=os.listdir('output')
            # add prefix output/
            output_files=['output/'+i for i in output_files]
            print("output files: ", output_files)
            
            for i in slave_list:
                for j in output_files:
                    with open(j, mode='a') as f:
                        if get_last_character_from_file(j)!=']':
                            remove_last_comma_from_file(j)
                            # code exits, so write the last line as ]
                            f.write("\n]")

            sys.exit(1)
