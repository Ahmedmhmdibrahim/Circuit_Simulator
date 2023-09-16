import os
from typing import List, Dict

def read_file(file_path: str, output_type: str) -> List[str]:
    """
        This Function take file_path as argument and output_type which determine the type of output
        be list or string.
    """
    if os.path.isfile(file_path) == True:
        file = open(file=file_path, mode='r')

        content = file.readlines()

        if output_type == list:
            content = "".join(content).strip()
            L = content.split('\n')
            content = []
            for i in L:
                if i.strip() != "":
                    content.append(i.strip())
            return content
        elif output_type == str:
            return "".join(content).strip()
    else:
        print("File doesn't exist!")
