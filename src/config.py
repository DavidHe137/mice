import os
project_root = "/coc/pskynet6/dhe83/mice" #NOTE: adjust for different environemtns
data = os.path.join(project_root, "data")
experiments = os.path.join(project_root, "experiments")
logs = os.path.join(project_root, "logs")
scripts = os.path.join(project_root, "scripts")
src = os.path.join(project_root, "src")
outputs = os.path.join(project_root, "outputs")
llama = "/srv/nlprx-lab/share6/nghia6/llama"
tests_per_gpu = 25
delim = "|"
tasks = ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WiC", "WSC", "Winograd"]
