import pyrenn
from shutil import copyfile

dst = pyrenn.__file__
src = 'pyrenn_correct.py'
copyfile(src, dst)