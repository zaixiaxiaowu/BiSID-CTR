import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amazon_process import run

if __name__ == "__main__":
    run(os.path.dirname(os.path.abspath(__file__)), "gift")
