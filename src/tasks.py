import os
import sys
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append("/coc/pskynet6/dhe83/mice/src")
from utils import *

from pathlib import Path
from typing import Tuple
import sys
import torch


