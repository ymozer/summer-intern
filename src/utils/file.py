import os

def file_control(file_path: str, is_file_opened: bool):
    if os.stat(file_path).st_size != 0 and not is_file_opened:
        # remove the ] from the last line
        with open(file_path, 'r+') as f:
            lines = list(f)  # Convert the generator to a list
            lines[-1] = lines[-1].strip()[:-1]  # Remove the last comma
            if lines[-1].endswith('\n'):
                lines[-1] = lines[-1].rstrip('\n')  # Remove the new line character
            f.seek(0)
            f.writelines(lines)
            # seek to the last escape character and write a comma
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.write('')
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 2, os.SEEK_SET)
            f.write(',\n')